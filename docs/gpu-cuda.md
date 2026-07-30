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
| `COLI_GPU_INT4=1` | keep VRAM residents int4 (needs per-row int4 experts; grouped formats are rejected with a descriptive error) |
| `COLI_GPU_F32_FRAC=<0..1>` | adaptive per-expert precision: the hottest fraction of residents is promoted to f32 at `reheat`, the rest stay int4; format tracked per expert with re-upload on change |
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
  steps from the routing heat table; initial placement is a greedy heat/bytes
  knapsack. `COLI_REPLICATE_K` additionally mirrors the hottest residents
  into the CPU warm cache so a lane-balancer downgrade pays no disk read.

## Autotuning

`WmmaTuner` (`peregrine-model/src/wmma_tune.rs`) records per-shape
`(D, I, count, max_rows) → TileConfig` kernel-time EWMAs and picks the winning
tile per shape. The CUDA-side dispatch selector that would consume the
persisted table (`kernel_tuning.json`) is a CUDA-only follow-up — the tuner
and its serialization exist, the file is not yet written/read by any code
path.

## What still needs hardware

Every remaining roadmap item on the GPU side is gated on `nvcc` + real
hardware this workspace lacks (see [Roadmap](roadmap.md)): persistent CUDA
kernels (threadblocks looping dequeue→compute→enqueue), CUDA Graphs wired
into the decode loop, GPU-side fused reduce, a `cudaMallocAsync` pool for
`reheat` churn, CPU/GPU split GEMM, idle-cycle GPU compute, PCIe bandwidth
scheduling, GPUDirect Storage, and all multi-GPU work (expert
ownership/migration, NVLink placement, VRAM replication — the tier is
currently hardcoded to `device=0`).

## Why this matters

On a single consumer box both engines are disk-bound and the GPU lane is
mostly latent. The concurrent design pays off in the **residency regime** —
enough VRAM to hold the expert working set — which is exactly where colibrì
reports 6.84 tok/s on 6× RTX 5090 with its *phased* loop leaving the overlap
on the table. See [Benchmarks](benchmarks.md).
