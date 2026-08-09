[« Docs index](README.md)

# The concurrent 3-lane scheduler

The centerpiece of peregrine (`crates/peregrine-model/src/concurrent.rs`,
plus the CPU∥SSD core in `crates/peregrine-sched`): a completion-driven
scheduler where the GPU lane, CPU lane, and io_uring SSD lane all drain the
same MoE layer's experts at once.

## Task model

Per MoE layer, after routing and batch-union dedup, each **unique** routed
expert becomes exactly one `ExpertTask`. This structurally enforces the
"compute each expert once, apply to all its rows" invariant. Tasks are
classified O(1) by residency:

| Residency | Lane |
|---|---|
| VRAM-resident | GPU lane (batched `expert_group`) |
| Warm-cache hit | CPU lane, served from RAM |
| Disk | I/O lane streams it, then hands it to a compute lane |

With `COLI_LANE_BALANCE=1` the [`LaneBalancer`](adaptive-runtime.md#bubble-tuner--lane-balancer)
can override the static decision — e.g. downgrade a cold GPU-resident expert
to the CPU lane when telemetry shows the GPU is the bottleneck.

## The three lanes

**I/O lane.** `COLI_IO_RINGS` io_uring rings (default 4), each on its own
thread. Rings atomically claim expert batches off a shared `AtomicUsize`
cursor (`io_work.fetch_add` — lock-free work stealing) and issue a deep
batched submit of up to `COLI_IO_BATCH` experts × 6 regions per expert
(≈ 96 concurrent reads at the default 16). Contiguous regions are merged
before submit (`read_many` / `read_experts_batched`). On each completion the
lane stamps the task's slab and routes the now-ready expert to a compute lane
— it never blocks on a CQE.

**The claim size is an upper bound, not the claim.** The lane ceil-divides the
layer's work across the rings:

```rust
let batch = experts_per_batch().min(n_plans.div_ceil(n_rings)).max(1);
```

This matters because the two phases have very different work lists. A prefill
chunk's per-layer routed union is ~69 experts, so it claims the full 16. A
**decode** token routes only **8 experts per layer** — fewer than one claim.
With a fixed batch of 16, ring 0's `fetch_add(16)` returned start 0 and took
all 8, while rings 1–3 got starts 16/32/48, every one `>= n_plans`, and broke
out **without issuing a single read**. One ring did four rings' work on every
sparse layer of every decode token: measured at **24 % io duty across 4 rings**,
and ~0.6 GB/s where the same device gives 1.12 GB/s at 4 rings under `iobench`.

Ceil-dividing gives decode `ceil(8/4) = 2` so all four rings run, and leaves
prefill at its full 16. Measured effect at defaults: decode **21.83 → 16.08
s/tok**, io duty **24 % → 84 %**
([`bench-data/2026-08-09-decode-levers/`](../bench-data/2026-08-09-decode-levers/README.md)).
The lesson generalises: **any fixed claim size larger than the work list
serialises a work-stealing loop onto one worker**, silently, and the
thread-summed lane counters cannot see it — see
[Measurement discipline](measurement.md).

**CPU lane.** A worker pool computes each streamed expert's SwiGLU as bytes
land: fused gate+up → silu·up → down → weighted scatter. A streamed expert's
rows tile across the persistent `peregrine-par` pool
(`Mlp::swiglu` → `QtWeight::apply_vec` → `par_chunks_mut`); nesting is safe
because an expert's matmul inside a parallel MoE runs serial via a
thread-local guard.

**GPU lane** (feature = `cuda`). Ready VRAM-resident tasks coalesce into
batched `expert_group` calls on a persistent non-blocking stream with
double-buffered pinned staging, so upload, kernel, and download overlap:
H2D(n+1) ∥ kernel(n) ∥ D2H(n−1). See [GPU / CUDA](gpu-cuda.md).

## Deterministic accumulation

Top-K routing means two experts can write the same output row, so raw scatter
would race. peregrine stages per-expert results and merges them with a single
**fixed-order reduce** keyed by position — not by completion order — so the
output is bit-identical to the sequential path no matter how the lanes
interleave. This is asserted by tests (concurrent output == sequential).

## Dispatch-order shaping

Before the lanes start, the layer's task order can be shaped — all
correctness-neutral (the reduce is position-keyed):

- **Layout schedule** — if `<model-dir>/schedule.json` exists (from
  [`peregrine-layout-reorg`](layout-tools.md)) and `COLI_LAYOUT_SCHEDULE` is on
  (default), streamed expert plans sort by the schedule's disk-order rank so
  the batched submit issues contiguous-offset reads first.
- **Co-activation fusion** — pairs of experts that co-fire at ≥
  `COLI_FUSE_THRESHOLD` (default 0.9) are kept adjacent in dispatch
  (same io-claim window / same GPU batch) via `apply_affinity_order`.
- **Hypergraph grouping** — under `COLI_HYPER_SCHED=1`, union-find components
  over co-activation pairs act as hyperedges; same-component experts land in
  one claim window.

## Failure handling

On a batched-read failure the buffered path re-issues each region via
`Reactor::read_exact_retry` (linear backoff; transient `EIO`/`EAGAIN`/`EINTR`),
gated on `COLI_IO_RECOVERY` (default on). A panic-free discipline applies
throughout — errors propagate as `peregrine_core::Error`, never a crash
mid-inference.

## The streaming core (`peregrine-sched`)

`moe_streamed` is the simpler two-lane ancestor (io_uring streaming ∥ CPU
compute) kept as a correctness oracle and building block; `reconstruct.rs`
holds the deterministic re-assembly. The N-ring three-lane scheduler in
`concurrent.rs` is the production path.

## Related pages

- [Adaptive runtime](adaptive-runtime.md) — the telemetry → tuner → placement
  feedback loop layered on top of this scheduler.
- [Prefetch & caching](prefetch-and-caching.md) — how the next layer's/token's
  experts get warmed before the lanes ask for them.
- [I/O & storage](io-and-storage.md) — the reactor the I/O lane is built on.
