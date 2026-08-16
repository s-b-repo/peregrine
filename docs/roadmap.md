[« Docs index](README.md)

# Roadmap & status

The audited, per-item roadmap is [`todo.md`](../todo.md) — 137 tracked items
with file:line evidence, status verified against the codebase. This page is
the summary.

## Where things stand (2026-08-13)

| Scope | ✅ Done | 🟡 Partial | ⬜ Not started | Completion |
|---|---:|---:|---:|---:|
| **Full roadmap** (137 items) | 128 | 1 | 8 | **~93 % strict · ~94 % weighted** |
| **Priority shortlist** (20) | 17 | 0 | 3 | **85 % strict** |

Per-section: Prefetch **11/11** · GPU 9/10 · Caching **13/13** · I/O 10/11 ·
Memory/NUMA **8/8** · Scheduling 16/18 · Disk-layout **10/10** · Workload
**5/5** · Compilation **5/5** · Self-optimizing **10/10** · Multi-GPU 0/4 ·
Attention/serving **7/7** · Workload-reduction **15/16**.

What remains open splits four ways, and only the first group is blocked here
(the 2026-07-30 revision of this page said ten items needed hardware; five of
those have since shipped or closed on this box — the full audit trail is in
`todo.md`):

**Needs hardware this box lacks (5)** — ≥ 2 GPUs, a GPUDirect Storage driver
stack, or multiple hosts. Each has a design naming its `file:line` seam in
[scale-out-design.md](scale-out-design.md): GPUDirect Storage; multi-GPU
expert ownership/migration; NVLink-aware placement; VRAM replication;
distributed sharding across hosts.

**CUDA work this box could do — resolved by evidence instead (3).** Persistent
kernels are declined (they would delete the shipped CUDA-graph cache to solve
the same problem, monopolize the one non-blocking stream, and defeat the
`scratch_gen` guard); the `cudaMallocAsync` defrag pool **closed on
measurement 2026-08-13** (the probe was built instead: after worst-case churn
of the two expert block sizes, 96.7 % of free VRAM is still one block);
engine-idle GPU warming has an in-tree negative result, and the mid-forward
spill half is blocked on a `&mut GpuTier` seam, not on CUDA.

**Open by choice (1)** — CPU/GPU split GEMM: timing-derived split points would
make logits depend on machine timing, and it cuts zero bytes per token. Its
named replacement, `COLI_ROUTE_MIN_SHARE`, was **gated 2026-08-13 and failed
at the setting worth having**: τ=0.05 flips 27.9 % of top-1 predictions for a
~12.5 % read saving ([bench-data](../bench-data/2026-08-13-route-min-share/README.md)).

**Machine time (1)** — int2-g64 converted and **measured: flip_rate 1.000, a
closed negative** (2-bit round-to-nearest is not enough for this model; the
container decodes correctly). int3-g64 (12.7 % smaller than int4) is the one
untested rung on the precision ladder; the requantizer and the flip-rate gate
are both ready for it.

## What shipped, in waves

1. **Foundation** — CUDA backend FFI, io_uring with registered files, the
   3-lane concurrent scheduler, continuous batching, the bit-identical
   fork-join pool, the 3-tier memory hierarchy, INT4/INT8 quantization.
2. **Adaptive-runtime wave** (2026-07-30) — lane telemetry + bubble
   detection + CPU/GPU balancer, adaptive io_uring worker cap, transparent
   zstd (disk + RAM), the offline layout tool (greedy/Louvain/spectral),
   cross-session routing persistence, two-priority admission, NUMA pinning,
   heat-gated cache admission, a real `perf_event_open` LLC-miss counter,
   idle recompression, per-workload-class prefetch.
3. **Completion sweep** (same day) — sensor governors (thermal/RAPL/
   bandwidth), entropy-adaptive prefetch, NUMA-bound allocation +
   hierarchical dispatch, expert fusion + hypergraph scheduling, macro-state
   routing compression, the `galactic` one-shot pass, Hilbert/2-opt/tier
   layouts, physical checkpoint self-rewrite (`--apply`), online bandit +
   Q-learning schedulers, per-shape dispatch specialization, kblock layout
   auto-conversion, the `compile-plan` execution plan.
4. **Tokenizer fast path** — the vendored gigatoken BPE engine as the sole
   runtime tokenizer, parity-gated, 34× measured.

Items marked **pragmatic** in `todo.md` were completed in user-approved
pragmatic forms (e.g. profile-guided execution *planning* rather than
compiler-style codegen; heuristic rather than trained cache admission).

## Deprioritized

Fast matrix multiplication (Strassen-class) — asymptotically cheaper but
almost never wins for LLM inference; excluded from the percentages.

## Next measurement pass

5. **VRAM-residency pass** — fixed defects the roadmap's own bookkeeping hid:
   the heat/bytes residency knapsack was marked done and documented as live
   while having zero production callers; f32 promotion collapsed the resident
   set ~3×; one non-per-row-int4 expert truncated the whole tier. Plus
   format-split expert dispatch and a PCIe upload budget.
6. **Attention & serving pass** — opened §12–§13, an axis the original eleven
   sections had no category for. Adaptive prefill chunking (a fixed 64-token
   chunk made prefill quadratic in prompt length), a cross-request KV prefix
   cache, adaptive top-k, and the first int2 producer — plus the two measurement
   tools (`COLI_GATE_STATS`, `Model::prediction_flip_rate`) that make lossy work
   assessable at all.

The [benchmark study](peregrine-vs-colibri.md) predates the adaptive waves;
re-running it with the new knobs live (and on residency-capable hardware) is
the standing next step. Synthetic-model tests catch correctness; throughput
impact of the "many small adaptive knobs" pattern needs a real model,
evaluated combined.
