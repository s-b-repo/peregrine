[« Docs index](README.md)

# GPU / CUDA lane

The GPU lane is FFI to the **vendored, validated CUDA kernels** from colibrì
(`cuda/backend_cuda.cu` / `.h`) — fused quantized matmuls, Tensor Core WMMA
(W4A16 / INT4), SwiGLU, attention+RoPE — behind the `cuda` cargo feature of
`peregrine-cuda`. Rewriting those kernels in Rust was deliberately deferred:
they are already token-exactness-validated upstream, and re-wrapping them
would re-open that validation.

## Building

```bash
# default build: the CUDA backend is a stub — no nvcc, no GPU needed
cargo build --release

# on an NVIDIA host with CUDA installed:
cargo build -p peregrine-cuda --features cuda     # build.rs runs nvcc on cuda/backend_cuda.cu
cargo test  -p peregrine-cuda --features cuda     # GPU-gated tests, incl. graph capture
cargo build --release --features cuda -p peregrine-engine
```

`build.rs` compiles `cuda/backend_cuda.cu` with `nvcc` and links
`-lcudart -lstdc++`. The FFI is a flat `extern "C"` ABI over opaque
`ColiCudaTensor*` handles, with added `_async` non-syncing stream variants for
the concurrent scheduler.

## Runtime gates

Building with the feature does not enable the GPU — runtime is opt-in:

| Var | Effect |
|---|---|
| `COLI_GPU=1` | enable the GPU lane (the VRAM expert tier is built only when set) |
| `COLI_GPU_INT4=1` | prefer int4 VRAM residents (~8x denser). Needs per-row int4 sources; a grouped-int4 or int8 expert falls back to f32 on its own rather than failing the tier |
| `COLI_GPU_F32_FRAC=<0..1>` | adaptive per-expert precision: the hottest fraction of residents is promoted to f32 at `reheat`, the rest stay int4; format tracked per expert with re-upload on change |
| `COLI_PCIE_BUDGET_MB=<n>` | cap on bytes one `reheat` generation may upload across PCIe. Residency churn is otherwise unbounded (~18.9 MB int4 / ~151 MB f32 per expert, every 256 decode steps); the coldest deferred experts are reconsidered next generation. Watch `[gpu] transfer_frac` to size it |
| `COLI_CUDA_PROFILE=1` | accumulate per-call `h2d_ms` / `kernel_ms` / `d2h_ms` timings |
| `COLI_CUDA_TC_*` | Tensor Core dispatch knobs (int4 / W4A16 gates, min-row thresholds) |

## What the lane does

- **Batched expert compute**: ready VRAM-resident experts coalesce into
  `expert_group` calls (fused gate/up/silu/down on GPU); layer-level
  accumulation stays on the deterministic host-side reduce.
- **Pinned staging + async copies**: `cudaMallocHost` staging with
  `cudaMemcpyAsync` on a persistent non-blocking stream, double-buffered so
  H2D(n+1) ∥ kernel(n) ∥ D2H(n−1).
- **Persistent scratch pools**: pre-allocated scratch slots reused across
  layers — no per-layer allocation churn.
- **CUDA graph capture/replay**: implemented and tested (including
  multi-kernel graphs); wiring it into the decode loop is an open
  hardware-gated item.
- **Dynamic residency**: `reheat()` re-selects the hottest experts every 256
  steps from the routing heat table. Initial placement is a greedy heat/bytes
  knapsack (`solve_residency_sized`) seeded from the previous session's
  `route_stats.json`, so a warm start places by last session's routing; with no
  usable stats it falls back to the deterministic round-robin spread.
  `COLI_REPLICATE_K` additionally mirrors the hottest residents into the CPU
  warm cache so a lane-balancer downgrade pays no disk read.
- **Mixed-precision residency**: `COLI_GPU_F32_FRAC` promotes the hottest
  fraction of residents to f32. The resident count and that fraction are sized
  *together* (`plan_precision_fitted`) — an f32 resident costs ~8× an int4 one,
  so sizing the set from the int4 footprint first and promoting afterwards let
  the promotions eat the whole budget before the int4 tail was reached (~67
  residents instead of ~204 at `frac=0.25` on a 10 GB budget). A promoted
  expert that no longer fits falls back to int4 rather than being evicted.
- **Mixed-format sources**: int4 residency is a preference. An expert whose
  source is grouped-int4 or int8 falls back to the f32 path on its own, instead
  of failing the whole tier at the first non-per-row-int4 expert. Such an expert
  is asked once, not once per generation — the tier remembers that its source
  cannot be int4, otherwise it re-uploaded forever.
- **Format-split dispatch**: the kernel picks its path from `all_s4` over the
  *whole* call, so one f32 resident would drop every expert in that call onto
  the generic scalar path. `compute` issues one `expert_group` per residency
  format and restores job order before returning, leaving the position-keyed
  reduce untouched. (This alone does not guarantee Tensor Cores — the TC path
  also requires every job in a call to clear the min-row threshold.)
- **PCIe budget**: `COLI_PCIE_BUDGET_MB` bounds how many bytes one `reheat`
  generation may upload, deferring the coldest experts to the next generation
  rather than bursting gigabytes into the lane it is feeding.

## Autotuning

`WmmaTuner` (`peregrine-model/src/wmma_tune.rs`) records per-shape
`(D, I, count, max_rows) → TileConfig` kernel-time EWMAs and picks the winning
tile per shape. The CUDA-side dispatch selector that would consume the
persisted table (`kernel_tuning.json`) is a CUDA-only follow-up — the tuner
and its serialization exist, the file is not yet written/read by any code
path.

## What still needs hardware

These need `nvcc` + real hardware this workspace lacks (see
[Roadmap](roadmap.md)): persistent CUDA kernels (threadblocks looping
dequeue→compute→enqueue), CUDA Graphs wired into the decode loop, GPU-side
fused reduce, a `cudaMallocAsync` pool for `reheat` churn, idle-cycle GPU
compute, GPUDirect Storage, and all multi-GPU work (expert
ownership/migration, NVLink placement, VRAM replication — the tier is
currently hardcoded to `device=0`).

Three clarifications, because this list used to be wrong in both directions:

- **PCIe bandwidth scheduling is done**, not hardware-gated — see
  `COLI_PCIE_BUDGET_MB` above. The byte costs follow from the residency
  format, so the policy needs no device measurement.
- **CPU/GPU split GEMM is open by choice, not by hardware.** Splitting one
  expert's rows across lanes is small plumbing, but the CPU half computes int4
  and the GPU half f32, and a split point derived from the bubble tuner's
  wall-clock EWMA would make low-order output bits depend on machine timing.
- **CUDA Graphs into decode is much larger than "wiring".** Every step of
  `forward_layer` is host-resident and only MoE expert weights ever reach VRAM,
  so it needs device-resident attention/norm/router/embed weights and a device
  KV cache first.

## Why this matters

On a single consumer box both engines are disk-bound and the GPU lane is
mostly latent. The concurrent design pays off in the **residency regime** —
enough VRAM to hold the expert working set — which is exactly where colibrì
reports 6.84 tok/s on 6× RTX 5090 with its *phased* loop leaving the overlap
on the table. See [Benchmarks](benchmarks.md).
