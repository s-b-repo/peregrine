[« Docs index](README.md)

# Adaptive runtime

The 3-lane scheduler is stateless per forward; the adaptive runtime layers a
**telemetry → tuner → placement feedback loop** on top of it. Every knob is
env-gated, defaults to the historical behavior, and is **correctness-neutral**:
only latency and residency change, never the reduced values — bit-identical
when off.

The cycle, each forward:

1. `moe_forward_concurrent` brackets its three lanes and the reduce phase with
   wall-time counters, consults the `LaneBalancer` for per-expert placement
   using a live heat snapshot, and applies the co-activation affinity order.
2. After the forward, `Model::publish_lane_timings` swap-resets the
   accumulator, updates the tuners, steps the sensor governors, folds the
   routing-entropy EWMA, rewards and re-chooses the learned scheduler, feeds
   the co-activation tracker, and applies any changed io_uring worker cap.
3. On the next forward, `build_balancer` reads the fresh bias and
   `forward_hidden` applies the staged prefetch-distance nudges.

All adaptive state lives on `Model` behind atomics and mutexes, safe to
scrape from `&self`.

## Lane telemetry

`LaneTimings` (`lane.rs`) accumulates per-lane time (io / cpu / gpu / reduce)
inside `moe_forward_concurrent` via atomic counters on a shared accumulator
threaded through `ForwardCtx`. `PlanOptimizer::tick` (`telemetry.rs`) folds
`LaneTimings`, `BubbleTuner`, and `IoTuner` snapshots into one
`RuntimeTelemetry` value per forward.

### Those four counters are summed over threads, not wall clock

This is the single most important thing to know before reading them. The io lane
runs **one thread per ring** and the CPU lane a pool, so each counter is a sum of
busy time across several threads. **`io_us` of 17 s is equally "four rings busy for
4.3 s" and "one ring busy for 17 s"** — and on their own the counters cannot tell
you which.

`LaneTimings::lane_wall_us` is the denominator that resolves it: the **wall** clock
of the 3-lane region itself. With it, the shutdown report prints duty cycles:

```
[lane] 3 forwards: io 54.7s (73%) cpu 19.5s (26%) gpu 0.0s (0%) reduce 1.0s (1%)
[lane] moe wall 58.2s over 3 forwards (19.38s each); io duty 24% of 4 rings, cpu 0.3 workers busy
[lane] cpu-lane bandwidth 1.75 GB/s over 34.0 GB of expert slabs
```

`io duty 24% of 4 rings` says only ~0.94 rings were doing anything. **That number
found a bug that had halved decode throughput** — the percentages on the first line
read "I/O dominates at 73 %", which was true and useless, because it is a share of
*busy* time and three of the four rings were never busy. See
[the concurrent scheduler](concurrent-scheduler.md#the-three-lanes) for the fix and
[Measurement discipline](measurement.md#3-thread-summed-counters-are-not-wall-time)
for how to read the block.

### Two accumulators, on purpose

`Model` keeps the per-forward accumulator that `publish_lane_timings`
**resets** each forward — that is the sample the tuner needs — and a second,
run-lifetime `lane_totals` that is never reset, which is what the `[lane]` report
prints. Without the second one there is no way to ask "where did this *run* go":
the tuner's sample is gone by the time anyone could look at it. Both are folded from
the same `sample`, so the report and the controller can never disagree.

## Bubble tuner & lane balancer

`BubbleTuner` maintains an EWMA per lane (α = 0.3) and declares a
`Bias::Toward{Cpu,Gpu,Io}` only when the top lane exceeds **1.5×** the max of
the others for **3 consecutive forwards** — hysteresis defeats one-off spikes.

`LaneBalancer::choose(gpu_resident, heat)` consumes the bias inside the
scheduler (gate: `COLI_LANE_BALANCE=1`):

- `Bias::TowardGpu` → a **cold** GPU-resident expert downgrades to the CPU lane.
- `Bias::TowardCpu` → a hot streamed expert is a candidate to spill onto the
  GPU (the spill-upgrade path is reserved — it needs on-demand upload).

**Runtime expert replication** (`COLI_REPLICATE_K=<K>`) composes with this:
`Model::enqueue_expert_replicas`, called from `reheat`, prefetches the top-K
hottest VRAM-resident experts into the CPU warm cache too — so a `TowardGpu`
downgrade serves the expert straight from RAM, paying no disk read.

## IoTuner

EWMA over per-forward `io_us` plus SQ-full deltas (counted at every ring push,
drained per forward) drive grow/halve of the io_uring `iowq_max_workers` cap,
applied via `Reactor::set_iowq_max_workers` on every ring and deduped against
the last-applied value. Gate: `COLI_IO_TUNE` (default on). The last applied
cap is exposed as `Model::last_iowq()`.

## Sensor governors

Three governors (`peregrine-io/src/sensors.rs`, stepped from
`publish_lane_timings`) write one shared governor-adjustable worker count with
**shrink-wins arbitration**:

| Governor | Gate | Behavior |
|---|---|---|
| Thermal | `COLI_THERMAL_LIMIT_C=<°C>` | `/sys/class/thermal` sampled every 16 forwards; shrink workers above the limit, regrow 8 °C below |
| Power | `COLI_POWER_CAP_W=<W>` | wrap-aware RAPL meter (`/sys/class/powercap`); shrink above the cap, regrow below 80 % of it |
| Memory bandwidth | `COLI_BW_GOVERNOR=1` | CPU-lane GB/s EWMA (slab bytes ÷ cpu time); shrink on a plateau, periodic probe regrows |

## Routing entropy & phase detection

- **Entropy-adaptive prefetch** (`COLI_ENTROPY_ADAPT=1`, needs
  `COLI_PREFETCH_TUNE`): normalized Shannon entropy of the routed distribution
  over the K-deep history, EWMA'd per forward — narrow prefetch breadth when
  routing is repetitive, widen when dispersed.
- **Phase detection** (`predict.rs`): `PredictSource::PhaseAware` compares the
  newest two frames' Jaccard distance against `COLI_PHASE_THRESHOLD`
  (default 0.6) and, above it, folds a dominating vote onto the newest frame's
  experts. The weight comes from `predict::phase_boost(depth)`, not a constant:
  it shipped as a hardcoded `2` until 2026-08-08, which at the default depth
  only *tied* an expert that had just dropped out, so the shift response was
  inert while its tests passed on a hand-built `boost: 100`.
  `PhaseTracker` (`workload.rs`) keeps the stateful form — an EWMA plus a
  post-shift window — and **has no production caller**; until 2026-08-08 it was
  also the only reader of `COLI_PHASE_THRESHOLD`, so that knob governed nothing.
- **Workload classes**: the HTTP handler classifies each request's prompt tail
  (`workload::classify_str` → `Prose | Code | Json | Math | Mixed`) and the
  engine resolves per-class prefetch breadth via
  `COLI_PREFETCH_WARM_PATHS_<CLASS>` / `COLI_PREFETCH_HINT_PATHS_<CLASS>`.

## Learned schedulers

Both persist their policy in `route_stats.json` and are deterministic
(seeded LCG); if both are enabled the bandit wins.

- **ε-greedy bandit** (`COLI_LEARN_SCHED=1`, `learn.rs::BanditScheduler`):
  arms are knob configurations (prefetch distance × workers); reward is an
  EWMA of 1/decode-µs.
- **Tabular Q-learning** (`COLI_RL_SCHED=1`, `learn.rs::QScheduler`): states
  are (bias × stability), actions are knob deltas, reward is latency
  improvement; the Q-table persists.

## Cross-session persistence

With `COLI_ROUTE_STATS_PERSIST` (default on), `Model` saves
`<model-dir>/route_stats.json` on `Drop` — routing history, expert heat, the
co-activation tracker, and any learned policy — and auto-loads a matching one
at `Model::load`, so prefetch and residency start warm on the previous
session's routing. Artifacts are config-tag guarded against stale checkpoints.
See [Model format & artifacts](model-format.md) for the file inventory.

## Adaptive batching (serve layer)

The batching engine (`peregrine-serve/src/batch.rs`) contributes its own
loop — a two-tier priority queue (`X-Peregrine-Priority: high` drains first),
a latency-SLA working-cap governor (`COLI_BATCH_SLA_MS`), and a decode-heavy
admission window (`COLI_ADAPTIVE_WINDOW=N` runs prefill every Nth tick).
Details in [Serving](serving.md).

## Hardware counters

`peregrine_io::PerfCounter` is a real `perf_event_open(2)` LLC-miss counter
(thread-following, user-space-only, hand-declared `PERF_ATTR_SIZE_VER0` attr
layout). `telemetry::open_l3_miss_counter` gates it on `COLI_PERF_COUNTERS=1`. It **is**
wired: `peregrine-engine`'s `serve` opens it on the decode thread (`main.rs`) and
prints the total at shutdown. Opening it there is deliberate —
`perf_event_open(2)` follows the *calling* thread, so the figure covers attention
and the deterministic reduce, **not** the io_uring workers or the `peregrine-par`
pool. A whole-process number needs one counter per thread, and presenting this
one as that is how a number stops meaning anything.
Every constructor degrades to `None` when the kernel refuses (containers,
`perf_event_paranoid ≥ 3`, no PMU) — the counter is an optimization input,
never a dependency.

`COLI_PERF_PREFETCH_FEEDBACK=1` additionally lets the per-forward miss delta
steer the prefetch distance (rising misses widen it), which is the consumer this
page and `telemetry.rs` described for months before it existed. It is a
**second** opt-in on top of `COLI_PERF_COUNTERS`, because a measurement becoming
a control loop should require saying so. **The direction is a hypothesis.** The
counter follows the decode thread, so it cannot see the workers that stream and
compute experts; a rising miss rate there most plausibly tracks a growing KV
cache, which prefetch breadth does not address. The control law
(`model.rs::llc_trend` — seeding holds, ±10 % dead band) is pure and unit-tested
precisely because a live-counter test would pass by not running on any host that
refuses the syscall.

## Knob reference

The complete env-var table lives in [Configuration](configuration.md).
