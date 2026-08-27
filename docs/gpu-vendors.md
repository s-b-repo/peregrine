[« Docs index](README.md)

# GPU vendors: one kernel source, NVIDIA and AMD

peregrine's GPU lane is one kernel source — `cuda/backend_cuda.cu` behind the
vendor-neutral host ABI in `backend_cuda.h` — compiled by whichever toolchain
the build host has. The Rust FFI and everything above it never know the
vendor; only `crates/peregrine-cuda/build.rs` and the linked runtime differ.

| Vendor | Status |
|---|---|
| **NVIDIA (CUDA)** | Production. `nvcc` compiles the source directly; every measured number in this repo comes from this path (RTX 3060). |
| **AMD (ROCm/HIP)** | **Compiles-by-construction, never yet run on hardware.** `hipify-perl` translates the source mechanically, the two CUDA-only construct families are bracketed (below), and the launch gates route around them — but no AMD GPU has executed these kernels. Treat the first run on real hardware as a validation exercise, not a deployment. |
| **Intel** | Designed, not ported: the kernels have no SYCL/Level Zero form. `PEREGRINE_GPU_BACKEND=intel` is recognized and **rejected** at build time, because silently building nothing would read as support. |

## Building

```sh
# vendor selection (build time)
PEREGRINE_GPU_BACKEND=cuda|hip|auto   # auto (default): nvcc if present, else hipcc
CUDA_HOME=/opt/cuda                    # NVIDIA toolkit root (auto-detected)
ROCM_PATH=/opt/rocm                    # ROCm root (default /opt/rocm)
CUDA_ARCH=native                       # nvcc -arch (default native)
HIPC_ARCH=gfx1100                      # hipcc --offload-arch; `;`-separated for several

cargo build --release --features cuda  # the feature means "GPU lane", not "NVIDIA"
```

The HIP branch runs `hipify-perl` into `OUT_DIR` (the vendored `.cu` is never
touched), compiles with `hipcc`, and links `amdhip64`. When `HIPC_ARCH` is
unset it asks `rocm_agent_enumerator` for the present agents; on a ROCm host
with **no** AMD device (the compile-verification case) it falls back to a
broad CDNA + RDNA set (`gfx90a;gfx942;gfx1030;gfx1100`) rather than `native`,
which hipcc rejects without a device to inspect.

Missing toolchain is a build **warning** and no backend (a pure-CPU host must
still build the workspace); a present toolchain that rejects the source is a
hard build **error** — the two facts must never produce the same outcome.

At runtime, `peregrine_cuda::linked_backend()` reports which vendor the
binary actually carries (`"CUDA (NVIDIA)"` / `"HIP (AMD ROCm)"` / `""`), and
`devices::Vendor::backend_compiled` matches a present card's PCI vendor
against it — a HIP build drives AMD and cannot drive NVIDIA, so the answer is
per-vendor, not "was the feature on". `COLI_GPU_DEVICES` accepts vendor
prefixes (`cuda:0`, `hip:0`/`amd:0`); a request naming a vendor this binary
cannot drive is dropped with an advisory naming the rebuild, never silently.

## What is bracketed on AMD, and why it stays correct

Two construct families in the kernel source have no HIP form:

1. **NVIDIA WMMA (tensor-core) intrinsics** — the fp16 `w4a16_matmul_t` /
   `w4a16_gate_up_t` tiles and the int-s4 `grouped_s4_wmma` kernel. Their
   device code is guarded by `__CUDA_ARCH__` (undefined under hipcc), so on
   AMD they preprocess to *empty kernels*. The trap is the host side: an
   empty kernel launches successfully and returns zeros, and AMD's
   `hipDeviceProp_t::major` reports the gfx generation (11 on RDNA3), which
   would sail past the historical `compute_major >= 7` gate. Every launch
   gate therefore also consults `COLI_HAS_WMMA` (a compile-time constant, 0
   under `__HIP_PLATFORM_AMD__`), routing AMD to the portable arms instead:
   the GEMV dense path, `ARM_W4_PACKED`/`ARM_GENERIC` for grouped experts.
   Consequences on AMD: `COLI_CUDA_TC_INT4` / `COLI_CUDA_TC_W4A16` are inert,
   and the batched dense-MLP entry reports "unsupported" (the caller's CPU
   fallback serves it) — slower, never wrong.
2. **Masked warp shuffles** (`__shfl_down_sync`) — AMD wavefronts (64-wide;
   the reductions already use `warpSize`) execute in lockstep and HIP ships
   only the unmasked form. `COLI_SHFL_DOWN` expands to the historical masked
   call on NVIDIA and `__shfl_down` on AMD.

Verified without AMD hardware, and honestly labelled as such:

- `nvcc` compiles the bracketed source bit-for-bit equivalently (the macros
  expand to the historical code; the added `!COLI_HAS_WMMA` gate terms fold
  to constants).
- `hipify-perl` (fetched standalone) translates the full runtime API surface
  with zero warnings beyond the deliberate `__shfl_down` use, and a
  preprocessor-simulation pass asserts **no** CUDA-only token
  (`wmma::`, `mma.h`, `__shfl_down_sync`, `__syncwarp`, `nvcuda`) survives in
  any region reachable under `__HIP_PLATFORM_AMD__`.

## First run on a ROCm host — the validation runbook

The cortix box can be that host: alongside the RTX 3060 it carries a
**Navi 24 (Radeon RX 6400/6500-class, gfx1034)** — RDNA2, ~4 GB VRAM. ROCm
does not list gfx1034, but it executes gfx1030 code objects: build with
`HIPC_ARCH=gfx1030` and run with `HSA_OVERRIDE_GFX_VERSION=10.3.0` (the
established Navi 24 path). 4 GB is too small for a serious expert tier and
plenty for the whole kernel test suite.

1. `pacman -S rocm-hip-sdk` (Arch) or the distro equivalent; `hipcc` and
   `hipify-perl` must land under `$ROCM_PATH/bin`.
2. `PEREGRINE_GPU_BACKEND=hip cargo build --release --features cuda` — the
   boot banner must say `HIP (AMD ROCm)`.
3. `cargo test -p peregrine-cuda --features cuda` and
   `cargo test -p peregrine-model --features cuda` — the GPU tests init the
   device and self-skip only when none is present, so on an AMD box they
   must actually run (check `nvidia-smi`-style vacuity: a green suite on a
   box whose device never initialized proves nothing — the tests print what
   they placed).
4. The bit-identity gates that matter: `gpu_resident_layers_decode_like_cpu_layers`,
   the grouped-expert arm tests, and `peregrine flip-rate` against a CPU-only
   run of the same container.
5. Expect the portable arms only (no WMMA): performance work on AMD —
   rocWMMA for the fp16 tiles, `__builtin_amdgcn_*` dot products for s4 —
   starts *after* correctness is measured, per this repo's standing rule.

## Known limits

- The HIP path has **never executed on an AMD GPU**. Every claim above is
  compile-time construction plus audit, and says so.
- CUDA graphs hipify to hipGraphs (ROCm ≥ 5.3); if an older ROCm rejects
  them, `COLI_CUDA_GRAPHS=0` disables capture without touching correctness.
- `warpSize` on AMD is 64: the shared-memory reduction buffers are sized for
  ≤ 32 warps/block and every launch here uses ≤ 256 threads (4 waves), so
  the existing bounds hold with room.
- Glm5Next (GLM-5.3-Flash) refuses the GPU expert tier on *both* vendors
  until the kernels implement its SwiGLU clamp — see
  [deployment-glm53-flash.md](deployment-glm53-flash.md).
