[« Docs index](README.md)

# Benchmark pass — 2026-08-01 (post-improvement re-measure)

This is the re-run [`benchmarks.md`](benchmarks.md) called for: *"the study's
measurements predate the adaptive-runtime and completion sweeps … re-running
with the knobs live is the next benchmarking pass."*

Engine at `c23f7b5` (workspace defect audit / tokenizer fast paths), the same
GLM-5.2 744B int4 model and the same box as
[the original study](peregrine-vs-colibri.md).

**What this pass predates.** The knobs measured here are the §1–§11 set — all of
which change *how* expert bytes are fetched. The work that followed this pass acts
on *how many*: adaptive prefill chunking, the cross-request prefix cache, adaptive
top-k (`COLI_ROUTE_MIN_SHARE`) and the int2 producer, tracked in
[`todo.md`](../todo.md) §12–§13. None of them is measured below, and the finding
that motivated opening that axis is the one in the first TL;DR bullet.

## TL;DR

- **The adaptive/I-O knob bundle produces no measurable throughput gain.**
  At B=16 it is **1.00×** baseline (0.225 vs 0.224 tok/s, 3 reps each). Disk-read
  counts are byte-identical between the two, which corroborates it: the knobs
  change *how* the bytes are fetched, not how many, and on this disk-bound
  workload how doesn't matter.
- **The CUDA lane is a real but modest win: 1.09× at B=16** (0.244 vs 0.224,
  tight across 3 reps). These are the **first GPU-lane numbers ever measured**
  for peregrine — the original study was CPU-streaming only.
- **`peregrine bench 1 4 16` is not a sound harness for comparing configs.**
  Running the batch points in one process biases the later ones badly: baseline
  B=16 reads **0.143 tok/s in-sweep vs 0.224 isolated (+57%)**. Every number
  below that matters was re-measured one batch size per fresh process.
- Batching itself reproduces the published headline: **4.4× aggregate at B=16**
  (0.056 → 0.244 tok/s).

## Setup

| | |
|---|---|
| Engine | `c23f7b5`, `--release` (fat LTO); GPU arm additionally `--features cuda` (nvcc 13.3, links `libcudart.so.13`) |
| Model | `GLM-5.2-colibri-int4-with-int8-mtp`, 358 GB on disk, 78 layers, vocab 154 880 |
| Box | Ryzen 5 5500 (6C/12T, AVX2, no VNNI), 46 GB DDR4, RTX 3060 12 GB (sm_86), LUKS-encrypted NVMe |
| Shared env | `MALLOC_ARENA_MAX=2`, `COLI_ECACHE_GB=2`, `COLI_ROUTE_STATS_PERSIST=0`, `COLI_BENCH_STEPS=3` |
| Containment | each run in a `systemd-run --scope -p MemoryMax=24G` cgroup |

`COLI_ROUTE_STATS_PERSIST=0` matters for validity: persistence is on by default
and would have let each arm warm the next through `route_stats.json` in the
model directory. Every arm here starts cold.

**Cache is 2 GB, not the study's 10–12 GB.** The box was hosting a live VM, and
the study records an OOM kill at 24.7 GB RSS. Peak RSS came in at **16.4–17.3 GB**
across every run, so the cap was never approached. Shrinking the cache costs
almost nothing here — measured cross-token expert hit rate is 0.6 %.

### Arms

| Arm | Knobs on top of defaults |
|---|---|
| `baseline` | none — every adaptive knob defaults to historical behavior, so plain defaults *are* the published study's config |
| `improved` | `COLI_DIRECT`² `COLI_REGBUF`¹ `COLI_IO_TUNE` `COLI_LANE_BALANCE` `COLI_SHAPE_SPECIALIZE` `COLI_HYPER_SCHED` `COLI_PREFETCH_TUNE` `COLI_ENTROPY_ADAPT` `COLI_REPLICATE_K=8` |
| `gpu` | `improved` + `COLI_GPU=1` on the `cuda` build |

¹ **`COLI_REGBUF` turned out to be inert.** Audited after this pass:
`register_read_buffers()` / `IORING_OP_READ_FIXED` are implemented and tested,
but nothing calls them outside tests and no code reads the variable — the
streaming path always takes the plain read. So the `improved` arm was eight live
knobs and one no-op. It does not change the conclusion (the bundle was already
1.00×, with byte-identical disk reads), but the arm description would otherwise
overstate what was exercised. Tracked in [`todo.md`](../todo.md) §4.

² **`COLI_DIRECT` was crippled, which this pass could not have known.**
Discovered 2026-08-02: `Reactor::read_direct_aligned` — the call the streaming
lane makes under `COLI_DIRECT=1` — submitted **one region at a time**, so the
O_DIRECT lane ran at queue depth 1 regardless of `COLI_IO_BATCH` or
`COLI_IO_RINGS`, while the buffered sibling submitted all 96 regions of an expert
batch at once. Direct reads were consequently *slower* than buffered ones.

So the nine-knob bundle above was really **eight live knobs, one no-op, and one
running at a fraction of its intended depth** — and the headline **1.004×** rests
on that. Unlike footnote ¹ this one *may* change the conclusion, because
`COLI_DIRECT` is the knob most directly aimed at the disk-bandwidth bound the
whole §1–§11 program targets. The fix measured 1.2–1.3× on the I/O lane in
isolation on a LUKS laptop; its effect on end-to-end decode is unmeasured.

**Re-running this pass is the first thing to do on hardware that can load the
model** — see the [validation runbook](validation-runbook.md#2-re-run-the-2026-08-01-knob-pass--its-conclusion-is-suspect).
Until then, treat 1.004× as provisional rather than as settled evidence that
faster byte movement is exhausted.

## Results — B=16, isolated, 3 repeats

The headline measurement. One batch size per process, fresh load each time.

| Arm | rep 1 | rep 2 | rep 3 | **median** | vs baseline |
|---|---|---|---|---|---|
| baseline | 0.225 | 0.224 | 0.209 | **0.224** | 1.00× |
| improved | 0.225 | 0.225 | 0.237 | **0.225** | **1.004×** |
| gpu | 0.241 | 0.246 | 0.244 | **0.244** | **1.089×** |

Aggregate tok/s. Spread is tight (±3 % worst case) — B=16 runs are long enough
to average out host contention.

`improved` and `baseline` agree to three digits on two of three reps. Treating
the knob bundle as a throughput lever on this workload is not supported.

## Results — B=1, isolated, 2 repeats

| Variant | rep 1 | rep 2 | median | note |
|---|---|---|---|---|
| baseline | 0.057 | 0.054 | 0.056 | |
| `direct_only` (O_DIRECT alone) | 0.063 | 0.065 | **0.064** | tight; only lead worth chasing |
| `bundle_nodirect` (bundle − O_DIRECT) | 0.058 | 0.040 | 0.049 | ±31 % |
| improved | 0.062 | 0.049 | 0.056 | ±23 % |
| gpu | 0.067 | 0.037 | 0.052 | ±45 % |

**B=1 is too noisy to rank anything.** Single-sequence runs are ~50 s, short
enough that host contention dominates; the same config swung 45 % between
repeats. The one suggestive result is `direct_only` — O_DIRECT *by itself*
measured 1.15× baseline with a tight spread on both reps, while the full bundle
containing O_DIRECT did not. That is a lead, not a finding: n=2.

## Why the first sweep looked so different

The initial single-process sweep produced numbers that did not survive
isolation:

| B | baseline in-sweep | baseline isolated | |
|---|---|---|---|
| 1 | 0.045 | 0.056 | +24 % |
| 16 | 0.143 | 0.224 | **+57 %** |

In-sweep, B=16 runs third in the same process, after B=1 and B=4 have already
churned the allocator and the warm cache. That penalty landed on whichever arm
was measured that way, and it manufactured an apparent **1.45× for `improved`
and 1.64× for `gpu`** that vanished under isolation. It also manufactured an
apparent B=1 *regression* for `improved` (0.028 tok/s) that no later run
reproduced — the closest repeat was 0.062.

`docs/benchmarks.md` currently recommends `peregrine bench 1 4 16` for exactly
this comparison. It is fine for observing batch scaling within one config; it
should not be used to compare configs against each other.

## GPU lane

First measurement of this lane. `COLI_GPU=1` on the `cuda` build:

```
[CUDA] device 0: NVIDIA GeForce RTX 3060, 12.5 GB VRAM, sm_86
peregrine: GPU tier holds 62 experts in VRAM
```

62 residents against ~19 200 experts routed per B=16 step is **0.3 % residency**,
and the disk-read counters show it: 7867 reads vs 7957 for the CPU arms, a 1.1 %
reduction. A 1.09× throughput gain from a 1.1 % byte reduction means the win is
not residency — it is the GPU lane overlapping compute that the CPU arms
serialize against I/O. That is the concurrent-scheduler thesis showing up in a
measurement for the first time, just at a scale this GPU can barely express.

**Caveat:** the GPU arm is *not* token-identical. Per the study's limitations,
"GPU experts compute in f32 (not bit-exact vs int4 CPU) by design." `baseline`
vs `improved` is a clean correctness-neutral A/B; `gpu` trades exactness for
speed and should be read as a different configuration, not a free win.

## Threats to validity

- **Contended host.** An OSX-KVM VM (~80 % CPU) and its VNC helper (~100 % CPU)
  ran throughout. This is the dominant noise source and the reason B=1 is
  unusable. B=16 medians are stable regardless. Package temp stayed at 31 °C, so
  thermal throttling is ruled out.
- **Page cache not dropped between runs** (no passwordless sudo). With a 358 GB
  model against ~28 GB free, carryover is a small fraction of the working set,
  and the O_DIRECT arms bypass the page cache entirely.
- **2 GB expert cache**, so absolute tok/s is below the study's 10–12 GB config.
  Cross-arm comparisons are unaffected — every arm used the same cache.
- **n=3 at B=16, n=2 at B=1.** Enough to kill the large claimed effects; not
  enough to resolve effects under ~5 %.
- **Short runs** (3 decode steps) to keep wall-clock tractable at ~0.2 tok/s.

## Reproducing

```bash
# headline: one batch size per process, repeated
BATCH=16 VARIANTS="baseline improved gpu" scripts/bench-b1-isolate.sh out 3

# B=1 knob isolation (adds direct_only / bundle_nodirect variants)
BATCH=1 scripts/bench-b1-isolate.sh out 2

# the original 3-arm sweep (batch scaling within a config only)
MEMMAX=24G COLI_ECACHE_GB=2 scripts/bench-arms.sh out baseline improved gpu
```

Raw output for this pass is committed under
[`bench-data/2026-08-01/`](../bench-data/2026-08-01) — `sweep/` (the original
3-arm in-process sweep), `b1-isolate/` and `b16-repeat/`. Every run contributes
the engine's stdout table (`.out`), its stderr including the `[ecache]`
disk-read counters (`.err`), and wall/RSS stats (`.stat`). `sweep/` additionally
captures each arm's exact environment (`.env`); for the isolate runs the
environment is the `variant_env` table in `scripts/bench-b1-isolate.sh`, and
the per-variant tallies are in `results.csv`.
