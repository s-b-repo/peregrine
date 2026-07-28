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

**142 tests passing, 0 warnings, `cargo clippy` clean** (debug + release). Every
numeric kernel is ported from colibrì's `c/glm.c` and validated; the scalar
integer-dot kernels are the token-exactness reference and the SIMD variants are
checked bit-for-bit against them.

| Area | Crate(s) | Status | Validated by |
|---|---|---|---|
| Model loaders | `peregrine-core` | ✅ | config / safetensors index / QT format / dtype round-trips |
| CPU int4 forward | `peregrine-kernels`, `peregrine-model` | ✅ **runs end-to-end** | int8/int4 dots bit-exact on AVX-VNNI; MoE vs f32 ref; attention causality / decode==prefill; full `Model` load→forward→generate |
| io_uring streaming | `peregrine-io` | ✅ | io_uring reads validated byte-for-byte vs `pread` on real hardware; LRU cache; LFRU tiering; **registered files** (`IOSQE_FIXED_FILE`) + `SINGLE_ISSUER`/`COOP_TASKRUN`; O_DIRECT zero-copy lane |
| CUDA GPU lane | `peregrine-cuda` | ⚙️ FFI complete, host-gated | FFI to the vendored `cuda/backend_cuda.cu` (fused quant matmul, WMMA W4A16, SwiGLU, attention+RoPE) + `nvcc` build.rs behind the `cuda` feature; pinned staging, persistent stream, graph capture/replay (incl. multi-kernel). Default build is a stub — **GPU tests run on an NVIDIA box** |
| Concurrent scheduler | `peregrine-sched`, `peregrine-model` | ✅ core | `moe_streamed` overlaps io_uring streaming ∥ CPU expert compute; `concurrent.rs` runs N rings with lock-free work-stealing ∥ CPU pool ∥ GPU lane, fixed-order reduce; output == sequential |
| Data-parallel compute | `peregrine-par` | ✅ | persistent scoped pool for rmsnorm / resident MoE / per-row attention / every matmul; **bit-identical to serial** (`f32::to_bits`-exact), work-gated, nesting-safe |
| Prefetch & prediction | `peregrine-model` (`predict.rs`) | ✅ | K-deep `RouteHistory` + momentum / offline transition automaton, per-layer look-ahead emission, EWMA distance tuner, predictive eviction — all correctness-neutral (a wrong guess just re-streams identical bytes) |
| Continuous batching | `peregrine-serve` (`batch.rs`) | ✅ | one engine thread batches all in-flight requests; chunked prefill (64-token chunks) interleaved with decode, bit-identical to whole-prompt prefill (`engine_chunked_prefill_matches_reference`) |
| MLA absorption / MTP | `peregrine-model` | ✅ absorption / core | `mla_attention_absorb` ≈ dense + causal; `speculative_sample` rejection sampling statistically lossless |
| Serve (stdio drop-in) | `peregrine-engine` | ✅ | `READY`/`END` handshake — a drop-in for colibrì's `c/glm` behind `openai_server.py` |
| Serve (native HTTP) | `peregrine-serve` | ✅ | OpenAI-compatible `POST /v1/chat/completions` (SSE + non-streaming), `/v1/models`, `/health`; bearer auth, token caps, graceful shutdown, `#![forbid(unsafe_code)]` |

### Not yet done (gated on an NVIDIA box or the `transformers` oracle)
The token-exact gate vs colibrì's `ref_glm.json`; CUDA graphs wired into the
decode loop; persistent CUDA kernels; DSA sparse selection; MTP head wiring;
int2 kernels; adaptive CPU/GPU work balancing; GPUDirect Storage; multi-GPU
expert ownership. [`todo.md`](todo.md) is the audited roadmap (93 items, per-section
completion, ratings) — the prefetching/speculation section is 9/9 done.

## Architecture

```
crates/
  peregrine-core     formats: Cfg, safetensors index, QT quant detect, dtype, pack
  peregrine-kernels  std::arch int8/int4 dots + matmuls (scalar ref + AVX2/AVX-VNNI)
  peregrine-model    MLA attention, router, MoE, sampler, MTP, prefetch prediction,
                     the N-ring concurrent lane (concurrent.rs), top-level Model
  peregrine-io       io_uring Reactor (registered files, O_DIRECT), LRU cache, LFRU tiering
  peregrine-cuda     FFI to cuda/backend_cuda.cu (feature = "cuda")
  peregrine-sched    concurrent MoE scheduler: io_uring streaming ∥ CPU compute
  peregrine-par      persistent scoped worker pool, bit-identical to serial (std-only)
  peregrine-engine   binary `peregrine`: stdio serve protocol, demo, bench, automaton
  peregrine-serve    binary `peregrine-serve`: OpenAI HTTP server + continuous batching
cuda/                vendored CUDA kernels from colibrì (backend_cuda.cu / .h)
```

Five independent layers of concurrency (all I/O on io_uring; N work-stealing rings;
a data-parallel compute pool; an async GPU stream; prefill/decode interleaving in the
server) are mapped in [`DESIGN.md`](DESIGN.md#concurrency--parallelism-map-where-the-threads-are).

## Build & test

```bash
cargo test --workspace          # 142 tests, CPU-only, no GPU needed
cargo build --release           # optimized (fat LTO)
cargo clippy --workspace --all-targets    # clean
scripts/audit-bad-patterns.sh --strict   # quality gate: no panic-vectors/UB (see docs/BAD_PATTERNS.md)

# GPU lane (on an NVIDIA host with CUDA installed):
cargo build -p peregrine-cuda --features cuda
cargo test -p peregrine-cuda --features cuda    # GPU-gated tests, incl. graph capture
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

```bash
# native OpenAI-compatible HTTP server (continuous batching, SSE streaming):
cargo run --release -p peregrine-serve -- --model /path/to/model --port 8080
curl -s localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"glm-5.2","messages":[{"role":"user","content":"hi"}],"stream":true}'

# aggregate decode-throughput sweep over batch sizes (the batching amortization):
COLI_MODEL=/path/to/model cargo run --release --bin peregrine -- bench 1 4 16

# offline prefetch automaton: writes <model-dir>/automaton.json (auto-loaded next load)
cargo run --release --bin peregrine -- build-automaton /path/to/model 256
```

`Model::load` accepts any real int4/int8 container model directory in the GLM-5.2
weight-naming scheme (`model.layers.N.self_attn.*`, `mlp.experts.M.*`, …). The
`COLI_MODEL` env var name is kept from colibrì for drop-in compatibility.

### Tuning knobs (env)

All default to sensible values; every one of them affects performance only — the
token stream is unchanged.

| Var | Effect |
|---|---|
| `COLI_IO_RINGS` | io_uring rings for the streaming lane (default 4), each on its own thread |
| `COLI_IO_BATCH` | reads in flight per submit (default ~96) |
| `COLI_DIRECT` | O_DIRECT lane: DMA straight into aligned buffers, bypassing the page cache |
| `COLI_PAR_THREADS` | `peregrine-par` pool size; `1` = fully serial (the A/B baseline) |
| `COLI_ECACHE_GB` | warm expert RAM cache budget |
| `COLI_GPU`, `COLI_GPU_INT4` | GPU lane / VRAM expert tier |
| `COLI_PREFETCH_*` | lanes, look-ahead depth, distance tuner, protected set, verification |
| `COLI_ROUTE_HIST_DEPTH` | K-deep routing history feeding the predictor |

## Benchmarks & comparison

[**docs/peregrine-vs-colibri.md**](docs/peregrine-vs-colibri.md) is a same-hardware
study of peregrine (Rust) vs colibrì (C) running the real **GLM-5.2 744B** int4
model, with architecture comparison, the full catalogue of improvements, and
measured token specs. Headline (single RTX 3060 / Ryzen 5 5500 / 46 GB box,
CPU-streaming decode):

| | peregrine (Rust) | colibrì (C) |
|---|---|---|
| Decode, single sequence (steady state) | 0.054 tok/s | **0.077 tok/s** |
| **Batched decode, B=16 (aggregate)** | **0.280 tok/s** (4.4× over B=1) | — |
| Warm cache on a repeated forward | **3.58×** (100 % hit, 0 disk) | learned pin |
| Cross-token expert locality | 0.6 % (measured) | — |

Both are **disk-bandwidth-bound** (600 experts ≈ 11 GB/token); colibrì is currently
~1.4× faster at raw *single-sequence* streaming (deeper io_uring queue), while
peregrine adds a verified warm-cache/scheduler stack and memory safety.

Continuous batching is where the concurrent design starts paying on this hardware:
decoding B sequences together reads each routed expert **once per step and shares it
across the batch**, so step time grows only 3.6× for 16× the tokens — a measured
**4.4× aggregate gain at B=16** (0.064 → 0.280 tok/s on the real 744B model). The win
is amortization of the byte budget, not a faster drive; the absolute ceiling stays
disk-bound. On the resident (no-disk) path the `peregrine-par` compute pool lifts
B=256 aggregate to **79.6k vs 66.3k tok/s serial (1.2×)** with no small-batch
regression, and that lever scales with hidden size.

The scheduler's full advantage is still latent without expert residency — colibrì
reaches **6.84 tok/s on 6× RTX 5090** (full residency), the regime that motivates
peregrine's concurrent design. See the document for methodology, all numbers, and
limitations.

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
