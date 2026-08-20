[« Docs index](README.md)

# External audit: what it found, and what has since been measured

An external audit of this repository ([issue #6](https://github.com/s-b-repo/peregrine/issues/6))
worked through the roadmap, `todo.md` and the benchmark study and produced a
21-section review with its own Tier 1/2/3 priority ranking. Its central thesis
is correct and worth repeating:

> a lot of optimizations have already been implemented, but many have not yet
> been demonstrated to improve real GLM-5.2 throughput.

That is still the single most accurate sentence anyone has written about this
project.

This page exists because **the audit's priority list cannot be acted on as
written**. It was assembled from documents that predate several measurements,
and at least two of its Tier 1 items are things the repo has since either
*shipped by default* or *measured and rejected*. Following the list literally
would spend scarce machine time re-deriving closed results — which is the exact
failure `measurement.md` was written to prevent.

Below: every actionable item, with its current status and the evidence.

---

## Already answered since the audit was written

**Do not re-propose these without new evidence.**

| Audit item | Status now | Evidence |
|---|---|---|
| §9 "route pruning / minimum-share gating … one of the most interesting remaining optimizations" | **Measured and rejected.** `COLI_ROUTE_MIN_SHARE` stays off | τ=0.05 flips **27.9 %** of top-1 predictions for a ~12.5 % read saving; τ=0.02 flips 20.7 %. The tail the gate-mass counters called negligible is negligible in *mass*, not in *effect* (`bench-data/2026-08-13-route-min-share/`) |
| §13 "prefill and decode as separate forwards … desired improvement is to fuse" | **Shipped, and on by default** | `COLI_FUSE_PREFILL=1` since 2026-08-13. Output-neutral, asserted twice: `a_fused_chunk_is_indistinguishable_from_two_separate_forwards` (bit-identical logits) and `fused_prefill_emits_the_same_tokens_as_the_two_forward_tick` (identical token stream) |
| §13 "the old fixed 64-token prefill chunk … adaptive chunking was added" | **Shipped and default** | `COLI_PREFILL_CHUNK_DIV=4` |
| §15 "DSA exists in the code but wasn't reachable in the production execution path" | **Wired**, both paths | Single-sequence 2026-08-14; the batched server core 2026-08-20 (`batched_dsa_rows_get_what_each_sequence_would_get_alone`). The audit's snapshot was accurate when taken |
| §10 "INT3 … still interesting … untested point" | **Measured, failed its gate, twice** | uniform int3-g64 `flip_rate = 0.514`; asymmetric (`--down keep --keep-last-layers 6`) `0.447`. The data-free RTN ladder is closed at every measured point |
| §12 "KV is currently f32" | **Shipped** | `COLI_KV_DTYPE=f16` |
| §12 "paged KV / block-pooled KV" | **Closed** | `KvBuf` grows in bounded blocks and `KvPool` recycles a retired sequence's buffers process-wide; `attention.rs:67` calls this "the last open half of paged/block-pooled KV" and closes it. Recycling lives in `Drop`, not at engine retirement points, because there are ≥6 places a `SeqKv` dies |
| §12 "capacity is constrained by count rather than byte-efficient paging" | **Closed** | `COLI_KV_BUDGET_MB` is a byte ceiling admission respects alongside `--max-batch`, and it charges a shared prefix **once** per distinct allocation, not once per holder |
| §14 "speculation that reduces total bytes/token, not merely more candidate tokens" | **Adopted as the standing rule**, and acted on | `COLI_SPEC_CONF=0.65` + `COLI_DRAFT=5` measured **+37 % tok/s and −22 % disk reads** at B=16, REPEATS=3 — precisely by pruning drafts that would not repay their verify rows. See [speculative-decoding-alternatives.md](speculative-decoding-alternatives.md) |
| §7 persistent CUDA kernels | **Declined on evidence**, and the audit agrees | They would delete the shipped CUDA-graph cache to solve the same problem and monopolize the one non-blocking stream |
| §8 `cudaMallocAsync` defrag pool | **Closed on measurement** | After worst-case churn of the two expert block sizes, **96.7 %** of free VRAM is still one block. The probe was built instead of the pool |
| §20.4 "several historical benchmark cells were single runs" | **Fixed in the harness** | `scripts/bench-serve-envarms.sh` rotates arm order across repeats and reports medians; `iobench` defaults to `REPS=5` and refuses to resolve a gap smaller than the measured spread |
| §19 "dead/unreachable code was a real gap" | **Now a permanent gate** | `[R]` is a counted class in `scripts/audit-bad-patterns.sh`, reported on every run |

Two audit claims are simply superseded by later data rather than by work:

- **§1 "single-stream decode slower than colibrì (0.062 vs 0.077 tok/s)."** On the
  second box the comparison *inverts* — peregrine ~980 MB/s against colibrì's
  ~870 MB/s after the O_DIRECT and parallel-ring work. There is no stable
  "colibrì is faster" fact to chase; see `benchmarks.md`.
- **§4 "the repository initially attributed the low hit rate to routing entropy;
  that was later disproven."** Correct, and the audit states the replacement
  explanation accurately: one token routes ~11 GB against a ~363 GB pool, so the
  cache cannot hold a pass at any eviction order. That closure is why
  "bigger warm cache" is a recorded closed negative — with the explicit
  exception that it does **not** cover a budget comparable to a whole pass
  (`COLI_ECACHE_GB=auto`).

---

## Open, and the audit is right

### Reduce bytes per token (audit Tier 1 #1)

Unchanged as the fundamental constraint, and unchallenged as the top priority:
**10.85 GB of routed expert reads per decoded token** against a ~370 GB working
set, io duty 93 %. Every ranked idea in
`ideas-tokens-per-sec-2026-08-15.md` is scored in GB/token saved or delivered
GB/s gained for this reason.

The audit's own framing is the one to keep: *making the I/O engine faster has
diminishing returns* — peregrine reached ~980 MB/s raw reads and remained
disk-bound.

### Better sub-4-bit quantization: vector quantization, AQLM, QuIP# (Tier 2 #7)

**The strongest genuinely-open idea in the audit, and it is right for the right
reason.** Every failure on this ladder — int2-g64 at `flip_rate 1.000`,
int3-g64 at 0.514, asymmetric int3 at 0.447 — used *data-free* round-to-nearest.
That is not evidence against data-aware codebooks, and llama.cpp's sub-4-bit
formats survive only because an importance matrix protects the salient channels.

Partially prepared: `peregrine-requantize --calib` and `peregrine calib-capture`
are code-complete for importance-weighted rounding, with the pooled-per-layer
variant specced first (per-expert statistics are noise at realistic trace
lengths). The measurement night was shelved 2026-08-16 when Qwen resident
serving took priority. VQ/AQLM-style codebooks are a further step and a genuine
research project, not a knob.

### Multi-GPU residency, GPUDirect Storage, distributed sharding (Tier 1 #2, Tier 3)

Correctly identified, and correctly described by the audit as *hardware-gated
rather than missing design work*. Each has a design naming its `file:line` seam
in [scale-out-design.md](scale-out-design.md). The audit's observation that the
3-lane advantage is **latent** while everything is disk-cold is the sharpest
architectural point it makes.

### A fresh end-to-end benchmark (§20.2, and the audit's bottom line)

> The single most valuable missing piece now is a fresh, controlled benchmark
> after all current changes.

Still true, still owed, and still blocked on machine time rather than design.
The harness exists (`bench-serve-envarms.sh`, `bench-measure.sh`,
`bench-serve-lanes.py`, `bench-serve-gaps.py`) and the archive convention is
established (`bench-data/<date>-<topic>/`).

---

## What this response changed

Two of the audit's measurement gaps were real, absent, and buildable without the
checkpoint. Both are now instruments rather than arguments.

### Queue time (§20.7)

The audit asks for "request/s, latency, TTFT, p50/p95/p99, **queue time**,
continuous arrival workloads". TTFT and inter-token percentiles already existed
(`bench-serve-lanes.py`, `bench-serve-gaps.py`) — but **every one of those
instruments starts counting once a request is already being served.** Time spent
waiting for admission was therefore indistinguishable from slow decode, which on
a server that sheds at `COLI_QUEUE_DEPTH` is exactly the difference between "at
capacity" and "over capacity".

`/metrics` now reports `queue`: `wait_us`, `admits`, `max_us`, `mean_us`. Counted
per *admission*, not per submit — a request refused at the door never waited for
anything, and averaging refusals in as zeros would flatter the mean precisely
when the server is most overloaded.

### Joules per token (§20.5)

The audit notes "very little emphasis on joules/token, watts, performance/watt",
and it is right: `EnergyMeter` existed but was read only under the opt-in
`COLI_POWER_CAP_W` governor and never surfaced.

`/metrics` now reports cumulative `energy_uj`, so `Δenergy_uj ÷
Δdecode.tokens_emitted` is microjoules per token. **Read it with both caveats:**

- **RAPL covers the CPU package, not the machine.** On this box the domains are
  `package-0` and `core` — no DRAM domain and certainly no SSD. On an engine
  whose bottleneck is 10.85 GB of expert reads per token, the component doing
  the most work is the one RAPL cannot see. This is a *floor* on system energy,
  not an estimate of it, and closing that gap needs a wall meter.
- **`energy_uj` is root-only on current kernels** (the PLATYPUS side-channel
  mitigation), so it reads `null` for an unprivileged server. Grant the counter
  rather than running the server as root:

  ```
  SUBSYSTEM=="powercap", ACTION=="add", \
    RUN+="/bin/chmod g+r /sys/class/powercap/%k/energy_uj"
  ```

  `null` is the expected reading on a stock host, and is deliberately not `0` —
  zero energy and no permission are different facts.

### Still not built, deliberately

- **Continuous-arrival (open-loop) workloads** (§20.7). Both existing clients are
  closed-loop. An open-loop arrival process is what makes queue time mean
  something; the counter above is the prerequisite, not the study.
- **Long-context scaling sweeps** (§20.6). Needs the checkpoint.
- **Multi-engine comparison** (§20.1). The audit is right that this repo
  demonstrates *peregrine vs colibrì*, not "peregrine is the best engine". A
  same-hardware llama.cpp/ktransformers comparison is a real gap, and an honest
  one to leave open rather than half-answer.

---

## On the engine comparison in the issue body

The issue body ranks peregrine against vLLM, SGLang, llama.cpp, ExLlamaV3,
TensorRT-LLM, ktransformers and Lsglang. The framing to keep is its own:

> If by "better" you mean fastest normal LLM server, peregrine isn't the leader.
> If you mean "I have a gigantic MoE model, only a fraction of experts fit in
> VRAM, and I want CPU + GPU + system RAM + NVMe all working simultaneously" —
> then the field becomes much smaller.

Two of the named engines are worth reading rather than benchmarking against:
**ktransformers** for router-aware hot-expert caching, and **Lsglang** for
GPU + NUMA hybrid MoE scheduling. Both attack the same bottleneck from the
residency side. Neither claim should be cited as a number here without a
same-hardware run — the audit's own §20.1 complaint applies to its own table.
