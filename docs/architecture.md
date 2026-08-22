[« Docs index](README.md)

# Architecture

peregrine is a from-scratch Rust MoE inference engine — a spin-off of
[colibrì](https://github.com/JustVugg/colibri) — built around one thesis: on a
streamed Mixture-of-Experts model, the **CPU lane, GPU lane, and io_uring SSD
lane should drain the same MoE layer concurrently**, instead of running as
sequential phases. Target platform: **Linux + NVIDIA CUDA** (the default build
is CPU-only and needs no GPU).

## Why it exists: the phased-vs-concurrent gap

colibrì's CUDA MoE loop is *phased*: VRAM-resident experts are deferred,
RAM/disk experts compute inline on the CPU, and the GPU expert group is
dispatched only after that loop finishes — so CPU-expert and GPU-expert compute
never overlap on the same layer. An in-code note upstream measures the waste:
*"9343 experts in VRAM sat unused during prefill — 81 s of expert-matmul all on
CPU, GPU groups 21 ms total."*

For a decode block with 3 VRAM + 2 RAM + 3 disk experts:

- phased wall-clock ≈ `max(disk_chain, cpu_5_experts) + gpu_3_experts`
- peregrine wall-clock ≈ `max(gpu_lane, cpu_lane, disk_lane)` — the slowest
  single lane, not the sum.

The full design rationale, milestones, and C-source cross-references live in
[`DESIGN.md`](DESIGN.md).

## Workspace map

```
crates/
  peregrine-core     formats: Cfg, safetensors index (zstd-aware), QT quant detect,
                     dtype, pack, compress
  peregrine-kernels  std::arch int8/int4 dots + matmuls (scalar reference + AVX2/AVX-VNNI)
  peregrine-model    MLA attention, router, MoE, sampler, MTP, prefetch prediction,
                     lane telemetry + bubble tuner + lane balancer, IoTuner, PhaseTracker,
                     WmmaTuner, PlanOptimizer, the N-ring concurrent lane (concurrent.rs),
                     top-level Model
  peregrine-io       io_uring Reactor (registered files, O_DIRECT, fadvise, batched hint),
                     priority-weighted LRU cache, warm cache (Bloom + optional zstd),
                     mem hints (hugepages, NUMA pinning), topology probe, perf counters,
                     aligned slab pool
  peregrine-cuda     FFI to cuda/backend_cuda.cu (feature = "cuda")
  peregrine-sched    concurrent MoE scheduler: io_uring streaming ∥ CPU compute
  peregrine-par      persistent scoped worker pool, bit-identical to serial (std-only)
  peregrine-engine   binary `peregrine`: stdio serve protocol, demo, bench, offline artifacts
  peregrine-serve    binary `peregrine-serve`: OpenAI-compatible HTTP server + continuous batching
  peregrine-tools    lib + binary `peregrine-layout-reorg`: offline expert re-layout
  peregrine-token    vendored gigatoken BPE subset (MIT): the runtime tokenizer
cuda/                vendored CUDA kernels from colibrì (backend_cuda.cu / .h)
```

Per-crate deep dives: [concurrent scheduler](concurrent-scheduler.md) ·
[adaptive runtime](adaptive-runtime.md) · [prefetch & caching](prefetch-and-caching.md) ·
[I/O & storage](io-and-storage.md) · [GPU / CUDA](gpu-cuda.md) ·
[tokenizer](tokenizer.md) · [serving](serving.md) · [layout tools](layout-tools.md).

## The memory hierarchy

A 744B-class MoE activates only a fraction of its parameters per token. The
dense sub-model (attention, shared expert, embeddings) is reused every token
and stays resident; the routed experts change token to token and are treated as
one three-tier hierarchy:

```
GPU VRAM  ──  hottest experts, heat-ranked, re-selected by reheat() every 256 steps
RAM       ──  warm expert cache (priority-weighted LRU + Bloom filter, optional zstd)
SSD       ──  everything else, streamed per token over io_uring
```

Residency is decided by a greedy heat/bytes knapsack (`gpu.rs::solve_residency_greedy`),
falling back to round-robin on a cold heat table. The
[`LaneBalancer`](adaptive-runtime.md) can override static residency at dispatch
time when telemetry shows one lane is the bottleneck.

## Forward pass: one MoE layer

Per MoE layer, after routing (Phase A) and batch-union dedup (Phase B), each
**unique** expert becomes exactly one `ExpertTask` — structurally enforcing
"compute each expert once, apply it to all its rows". Tasks classify O(1) by
residency into three lanes that run as concurrent actors:

- **GPU lane** — batched `expert_group` calls on a persistent non-blocking
  CUDA stream, double-buffered pinned staging so H2D(n+1) ∥ kernel(n) ∥ D2H(n−1).
  Weight *upload* runs the same way: an int4-resident expert is read by io_uring
  straight into a `cudaHostRegister`'d aligned buffer and copied to VRAM
  asynchronously, so the disk→GPU path is two DMAs with no userspace copy
  (`COLI_GPU_PINNED`).
- **CPU lane** — a persistent worker pool (`peregrine-par`); each worker runs
  one expert's SwiGLU (fused gate+up → silu·up → down → weighted scatter), with
  inner row tiling staying SIMD.
- **I/O lane** — `COLI_IO_RINGS` io_uring rings (default 4), each on its own
  thread, atomically claiming expert batches off a lock-free cursor
  (`io_work.fetch_add`) and issuing deep batched submits (~96 reads in flight).
  On completion the ready expert routes to a compute lane — it never blocks.

**Accumulation is deterministic.** Top-K routing means two experts can write
the same output row, so results are staged and merged by a single fixed-order
reduce — output is bit-identical to the sequential path, regardless of lane
completion order.

## The five layers of concurrency

1. **All I/O is io_uring** — and as of the 2026-08-22 pass this is literal
   rather than aspirational. Config, safetensors headers, all weight loading,
   `tokenizer.json`, every optional JSON artifact, the KV-checkpoint reads *and
   writes*, `write_atomic`, and the `/proc`+`/sys` probes all go through the
   reactor; per-token expert streaming is the hot path. The ring gained write,
   fsync and read-until-EOF ops to make the write and synthetic-file halves
   possible. Two documented exceptions remain, both because io_uring has no such
   opcode: directory enumeration (`read_dir`) and `readlink`. The synthetic-file
   half is reversible with `COLI_IO_PROCFS=direct`, because it is the one place
   the invariant costs rather than saves — see
   [configuration](configuration.md#coli_io_procfs).
2. **The streaming MoE lane** is N parallel io_uring rings with lock-free
   work-stealing, a CPU worker pool consuming as bytes land, an optional GPU
   lane, and one deterministic reduce (`peregrine-model/src/concurrent.rs`).
3. **The resident compute path is data-parallel** on a persistent pool
   (`peregrine-par`): rmsnorm, resident MoE, per-row attention, and every
   matmul. Bit-identical to serial (`f32::to_bits`-exact tests), work-gated
   (tiny matrices stay serial), and nesting-safe.
4. **The GPU backend is async** — pinned staging + persistent non-blocking
   stream, with CUDA graph capture/replay available (`peregrine-cuda`).
5. **The serving engine interleaves prefill with decode** — chunked prefill
   (64-token chunks) advances round-robin *between* batched decode steps, so a
   long prompt never stalls the in-flight batch
   (`peregrine-serve/src/batch.rs`).

Expert reads are **zero-copy into the weight**: a landing region is a
`peregrine_io::Bytes` the streamed `QtWeight` moves in, and kernels read it as
`&[u8]`. The O_DIRECT lane DMAs each region's 4096-aligned superset straight
into an owned aligned buffer — bulk weight bytes are never memcpy'd in
userspace on either path.

## Design invariants

- **Bit-identical when off.** Every adaptive/optimization knob is env-gated
  and, when disabled, the token stream is byte-for-byte the historical one.
  Correctness-neutral subsystems (prefetch, eviction, layout schedules) can
  never change output — a wrong prediction just re-streams identical bytes.
- **Token-exactness anchors.** The scalar integer-dot kernels are the
  reference; SIMD variants are checked bit-for-bit against them. The
  concurrent scheduler's output equals the sequential path's.
- **No panics on the hot path.** Engine crates deny
  `unwrap`/`expect`/`panic`; every fallible public API returns
  `peregrine_core::Error`. Enforced workspace-wide by
  [`scripts/audit-bad-patterns.sh`](BAD_PATTERNS.md).
- **`unsafe` is confined** to `peregrine-io` (io_uring, madvise/mbind, perf),
  `peregrine-kernels` (SIMD), `peregrine-cuda` (FFI), and the vendored
  `peregrine-token`; `peregrine-core` is `#![forbid(unsafe_code)]`.

## Lineage

peregrine ports colibrì's numerics faithfully; the vendored CUDA kernels under
`cuda/` come from upstream's `backend_cuda.cu`. Upstream:
[JustVugg/colibri](https://github.com/JustVugg/colibri) · fork:
[s-b-repo/colibri](https://github.com/s-b-repo/colibri). See the
[benchmark study](peregrine-vs-colibri.md) for a same-hardware comparison.
