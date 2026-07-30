[« Docs index](README.md)

# Roadmap & status

The audited, per-item roadmap is [`todo.md`](../todo.md) — 95 tracked items
with file:line evidence, status verified against the codebase. This page is
the summary.

## Where things stand (2026-07-30)

| Scope | ✅ Done | 🟡 Partial | ⬜ Not started | Completion |
|---|---:|---:|---:|---:|
| **Full roadmap** (95 items) | 83 | 2 | 10 | **~87 % strict · ~88 % weighted** |
| **Priority shortlist** (19) | 15 | 1 | 3 | **~79 % strict · ~82 % weighted** |

Per-section: Prefetch **9/9** · GPU 4/8 · Caching **10/10** · I/O 10/11 ·
Memory/NUMA **5/5** · Scheduling 15/18 · Disk-layout **10/10** · Workload
**5/5** · Compilation **5/5** · Self-optimizing **10/10** · Multi-GPU 0/4.

**Every remaining open item is hardware-gated** — it needs `nvcc` + an NVIDIA
GPU, ≥ 2 GPUs, a GPUDirect Storage driver stack, or multiple hosts, none of
which this workspace has:

- Persistent CUDA kernels (threadblocks looping dequeue → compute → enqueue)
- CUDA Graphs wired into the decode loop (capture/replay itself is built and tested)
- GPU-side fused reduce; `cudaMallocAsync` pool for `reheat` churn
- CPU/GPU split GEMM; idle-cycle GPU compute; PCIe bandwidth scheduling
- GPUDirect Storage (SSD → VRAM direct)
- Multi-GPU expert ownership/migration, NVLink-aware placement, VRAM
  replication (the tier is hardcoded to `device=0` today)
- Distributed inference across hosts with expert sharding

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

The [benchmark study](peregrine-vs-colibri.md) predates the adaptive waves;
re-running it with the new knobs live (and on residency-capable hardware) is
the standing next step. Synthetic-model tests catch correctness; throughput
impact of the "many small adaptive knobs" pattern needs a real model,
evaluated combined.
