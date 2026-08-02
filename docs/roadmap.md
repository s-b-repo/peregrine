[« Docs index](README.md)

# Roadmap & status

The audited, per-item roadmap is [`todo.md`](../todo.md) — 108 tracked items
with file:line evidence, status verified against the codebase. This page is
the summary.

## Where things stand (2026-07-30)

| Scope | ✅ Done | 🟡 Partial | ⬜ Not started | Completion |
|---|---:|---:|---:|---:|
| **Full roadmap** (108 items) | 89 | 6 | 13 | **~82 % strict · ~85 % weighted** |
| **Priority shortlist** (19) | 15 | 1 | 3 | **~79 % strict · ~82 % weighted** |

Per-section: Prefetch **9/9** · GPU 5/9 · Caching **12/12** · I/O 9/11 ·
Memory/NUMA **4/5** · Scheduling 16/18 · Disk-layout **10/10** · Workload
**5/5** · Compilation **5/5** · Self-optimizing 9/10 · Multi-GPU 0/4 ·
Attention/serving 2/5 · Workload-reduction 3/5.

The 19 open items split three ways, and only the first group is blocked here.

**Needs hardware (10)** — `nvcc` + an NVIDIA GPU, ≥ 2 GPUs, a GPUDirect Storage
driver stack, or multiple hosts:

- Persistent CUDA kernels (threadblocks looping dequeue → compute → enqueue)
- CUDA Graphs wired into the decode loop (capture/replay itself is built and tested)
- GPU-side fused reduce; `cudaMallocAsync` pool for `reheat` churn
- Idle-cycle GPU compute (PCIe bandwidth scheduling shipped: `COLI_PCIE_BUDGET_MB`)
- GPUDirect Storage (SSD → VRAM direct)
- Multi-GPU expert ownership/migration, NVLink-aware placement, VRAM
  replication (the tier is hardcoded to `device=0` today)
- Distributed inference across hosts with expert sharding

**Open by choice (1)** — CPU/GPU split GEMM: the CPU half computes int4 and the
GPU half f32, and a timing-derived split point would make low-order output bits
depend on machine timing, so the same prompt would give different logits run to
run.

**Pure CPU work, blocked only by size (8)** — and this is where the remaining
throughput is, since caching plateaus at this capacity ratio (0.6% warm-cache
hit rate on a 10 GB cache; see the §5.2 correction) and §1–§11 all optimize how
fast bytes move rather than how many:

- Fuse prefill rows into the decode batch (two disjoint forwards today, each
  streaming its own expert union)
- KV cache quantization (the KV is f32; ~180 MB per 1k tokens)
- Paged / block-pooled KV (per-sequence contiguous `Vec`s, capped by count not bytes)
- int2 checkpoint conversion — **the converter shipped** (`peregrine-requantize`, measured 2.69 GB → 1.35 GB on a real GLM-5.2 shard); what remains is a full-checkpoint run and a flip-rate measurement
- Heat-tiered on-disk precision (hot int4 / cold int2 — needs no loader change)
- Wire three features that ship tested but unreachable, found by the [R]
  reachability pass: the registered-buffer read path (`COLI_REGBUF` is read by no
  code), the `perf_event_open` LLC-miss counter (`COLI_PERF_COUNTERS` is inert —
  nothing calls the opener), and the slab pool's generation tagging
  (`checkout_tagged`/`checkin_tagged` have zero callers)

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
