# peregrine

**The fastest bird.** A from-scratch **Rust** MoE inference engine that drives
**CPU, GPU, RAM, and SSD concurrently** — a spin-off of
[colibrì](https://github.com/JustVugg/colibri) reimagined for true heterogeneous
concurrency and minimal syscalls.

> colibrì is the *hummingbird* — a tiny, elegant, dependency-free C engine that
> streams a 744B-parameter model from disk. **peregrine** is its falcon: the same
> idea, rebuilt in Rust to make every resource work at once. (The peregrine
> falcon dives at ~390 km/h — the fastest animal on Earth.)

## Why a spin-off

colibrì's C engine is already excellent, but its CUDA MoE path is **phased, not
concurrent**: VRAM-resident experts are deferred, RAM/disk experts compute on the
CPU inline, and the GPU expert group is dispatched *only after* that finishes
(`glm.c` MoE loop) — so on the same layer, CPU-expert and GPU-expert compute
never overlap. An in-code note there measures the waste: *"9343 experts in VRAM
sat unused during prefill — 81s of expert-matmul all on CPU, GPU groups 21ms
total."*

peregrine closes that gap: a completion-driven scheduler where the **GPU lane**,
the **CPU lane**, and the **io_uring SSD lane** all drain the same MoE layer at
once. Target: Linux + NVIDIA CUDA.

## Status

**62 tests passing, 0 warnings, `cargo clippy` clean** (debug + release). Every
numeric kernel is ported from colibrì's `c/glm.c` and validated; the scalar
integer-dot kernels are the token-exactness reference and the SIMD variants are
checked bit-for-bit against them.

| Area | Crate(s) | Status | Validated by |
|---|---|---|---|
| Model loaders | `peregrine-core` | ✅ | config / safetensors index / QT format / dtype round-trips |
| CPU int4 forward | `peregrine-kernels`, `peregrine-model` | ✅ **runs end-to-end** | int8/int4 dots bit-exact on AVX-VNNI; MoE vs f32 ref; attention causality / decode==prefill; full `Model` load→forward→generate |
| io_uring streaming | `peregrine-io` | ✅ | io_uring reads validated byte-for-byte vs `pread` on real hardware; LRU cache; LFRU tiering; **registered files** (`IOSQE_FIXED_FILE`) + `SINGLE_ISSUER`/`COOP_TASKRUN` |
| CUDA GPU lane | `peregrine-cuda` | ⚙️ scaffold | FFI to the vendored `cuda/backend_cuda.cu` + `nvcc` build.rs behind the `cuda` feature; default build is a stub. **GPU path validates on an NVIDIA box** |
| Concurrent scheduler | `peregrine-sched` | ✅ core | `moe_streamed` overlaps io_uring streaming ∥ CPU expert compute (reusable ring); output == sequential |
| MLA absorption / MTP | `peregrine-model` | ✅ absorption / core | `mla_attention_absorb` ≈ dense + causal; `speculative_sample` rejection sampling statistically lossless |
| Serve (stdio drop-in) | `peregrine-engine` | ✅ | `READY`/`END` handshake — a drop-in for colibrì's `c/glm` behind `openai_server.py` |

### Not yet done (gated on an NVIDIA box or the `transformers` oracle)
GPU-lane integration into the scheduler + async stream variants; the token-exact
gate vs colibrì's `ref_glm.json`; DSA sparse selection; MTP head wiring; int2 /
grouped-int4 kernels; full OpenAI request framing.

## Architecture

```
crates/
  peregrine-core     formats: Cfg, safetensors index, QT quant detect, dtype, pack
  peregrine-kernels  std::arch int8/int4 dots + matmuls (scalar ref + AVX2/AVX-VNNI)
  peregrine-model    MLA attention, router, MoE, sampler, MTP, top-level Model
  peregrine-io       io_uring Reactor (registered files), LRU cache, LFRU tiering
  peregrine-cuda     FFI to cuda/backend_cuda.cu (feature = "cuda")
  peregrine-sched    concurrent MoE scheduler: io_uring streaming ∥ CPU compute
  peregrine-engine   binary `peregrine`: stdio serve protocol + demo mode
cuda/                vendored CUDA kernels from colibrì (backend_cuda.cu / .h)
```

## Build & test

```bash
cargo test --workspace          # 62 tests, CPU-only, no GPU needed
cargo build --release           # optimized (fat LTO)

# GPU lane (on an NVIDIA host with CUDA installed):
cargo build -p peregrine-cuda --features cuda
```

## Run

```bash
# self-contained end-to-end demo (builds a tiny synthetic model, loads, generates):
cargo run -p peregrine-engine --bin peregrine -- demo

# serve mode (drop-in for colibrì's c/glm behind openai_server.py):
cargo run --bin peregrine -- build /tmp/demo-model     # write a tiny model
COLI_MODEL=/tmp/demo-model cargo run --bin peregrine    # emits READY, then:
#   GEN <ngen> <tok0> <tok1> ...   → greedy-generates, replies, emits END
#   QUIT
```

`Model::load` accepts any real int4/int8 container model directory in the GLM-5.2
weight-naming scheme (`model.layers.N.self_attn.*`, `mlp.experts.M.*`, …). The
`COLI_MODEL` env var name is kept from colibrì for drop-in compatibility.

## Lineage & references

peregrine is a Rust spin-off of **colibrì** and ports its numerics and streaming
model faithfully. The design rationale (the phased-vs-concurrent gap, the
three-lane scheduler, milestones) is in [`DESIGN.md`](DESIGN.md).

- Upstream: [JustVugg/colibri](https://github.com/JustVugg/colibri) · fork:
  [s-b-repo/colibri](https://github.com/s-b-repo/colibri)
- Port sources / correctness anchors (in colibrì's `c/`): `glm.c` (MoE, MLA
  `attention_rows`, IDOT kernels, router, `spec_decode`), `st.h`, `uring.h`,
  `tier.h`, `backend_cuda.h/.cu` (vendored here under `cuda/`), `openai_server.py`,
  and `ref_glm.json` + `tools/make_glm_oracle.py` (the token-exact oracle gate).

## License

MIT, inherited from colibrì.
