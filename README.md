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

**282 tests passing, 0 warnings, `cargo clippy` clean** (debug + release). Every
numeric kernel is ported from colibrì's `c/glm.c` and validated; the scalar
integer-dot kernels are the token-exactness reference and the SIMD variants are
checked bit-for-bit against them. The 2026-07-30 "adaptive-runtime wave" added
lane telemetry, a bubble-detection + CPU/GPU balancer, an adaptive io_uring
worker cap, transparent zstd compression, an offline layout tool
(`peregrine-layout-reorg`) with Louvain community-detection and spectral
(Fiedler) ordering passes, cross-session routing-history persistence, a
two-priority admission queue, NUMA thread pinning, a heat-threshold
cache-admission gate, a real `perf_event_open` LLC-miss counter, idle-tick
background recompression, and per-workload-class prefetch breadth — all
env-gated and bit-identical when off. A third pass (the **completion sweep**)
closed every remaining non-hardware roadmap item: sensor governors (thermal /
RAPL power / memory bandwidth), entropy-adaptive prefetch, NUMA-bound
allocations + hierarchical pool dispatch, co-activation expert fusion +
hypergraph scheduling, macro-state routing compression, the one-shot
`galactic` preprocessing pass, Hilbert/2-opt/tier layout methods, physical
checkpoint self-rewrite (`--apply`), online bandit + Q-learning schedulers,
per-shape dispatch specialization, kblock tensor-layout auto-conversion, and
the `compile-plan` profile-guided execution plan. The serve layer now runs a
**vendored [gigatoken](https://github.com/marcelroed/gigatoken) BPE tokenizer**
(stable-toolchain subset, MIT) as its **sole runtime tokenizer** — measured **34×
faster than HF `tokenizers` on this box** (204 vs 6 MB/s), id-for-id parity-gated
(the HF crate remains only as the test-suite oracle). See
[`todo.md`](todo.md) for the audited roadmap (**~87% strict / ~88% weighted of
95 tracked items — every open item is hardware-gated**).

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
| Serve (native HTTP) | `peregrine-serve` | ✅ | OpenAI-compatible `POST /v1/chat/completions` (SSE + non-streaming), `/v1/models`, `/health`; bearer auth, token caps, graceful shutdown, `#![forbid(unsafe_code)]`, two-tier priority queue via `X-Peregrine-Priority` |
| Adaptive runtime | `peregrine-model` (`lane.rs`, `iotune.rs`, `telemetry.rs`, `workload.rs`) | ✅ | per-lane wall-time accum + `BubbleTuner` EWMA → `LaneBalancer` (CPU/GPU bias-driven downgrade); `IoTuner` adjusts `iowq_max_workers` between forwards; `PhaseTracker` + `PredictSource::PhaseAware`; per-class prefetch breadth from the serving layer's prompt classifier; heat-threshold cache admission; NUMA worker pinning; real `perf_event_open` LLC-miss counter; cross-session `route_stats.json` at Drop / auto-load on load |
| Offline layout tool | `peregrine-tools` | ✅ | `peregrine-layout-reorg` consumes `dump-routes` JSON, emits `<dir>/schedule.json` via `--method greedy`, `louvain`, or `spectral` (Fiedler ordering); loader picks it up and pre-sorts disk reads |
| Compression | `peregrine-core` (`compress.rs`), `peregrine-io` (`warmcache.rs`) | ✅ | zstd end-to-end on disk (`Blob::with_compression`, header carries the tag + original size); optional transparent zstd on WarmCache admissions (`COLI_CACHE_COMPRESS`) |
| Tokenizer fast path | `peregrine-token`, `peregrine-serve` (`tok.rs`) | ✅ | vendored gigatoken BPE subset (stable toolchain, no libpython); id-for-id parity vs HF `tokenizers` on the committed GPT-2 fixture; cross-request memo cache; sole runtime tokenizer (HF crate is test-oracle only); `--bench-tokenizer` (34× locally) |

### Not yet done (all hardware-gated)
Every remaining roadmap item needs hardware this workspace lacks: CUDA Graphs
wired into the decode loop, persistent CUDA kernels, GPU-side fused reduce and
a `cudaMallocAsync` pool (`nvcc` + GPU); CPU/GPU split GEMM and idle-cycle GPU
compute (GPU); PCIe bandwidth scheduling and GPUDirect Storage (vendor stack);
multi-GPU expert ownership / NVLink placement / VRAM replication (≥ 2 GPUs);
distributed inference (multiple hosts). See [`todo.md`](todo.md).

## Architecture

```
crates/
  peregrine-core     formats: Cfg, safetensors index (with zstd), QT quant detect, dtype, pack, compress
  peregrine-kernels  std::arch int8/int4 dots + matmuls (scalar ref + AVX2/AVX-VNNI)
  peregrine-model    MLA attention, router, MoE, sampler, MTP, prefetch prediction, lane
                     telemetry + bubble tuner + lane balancer, IoTuner, PhaseTracker,
                     WmmaTuner, PlanOptimizer, the N-ring concurrent lane
                     (concurrent.rs), top-level Model
  peregrine-io       io_uring Reactor (registered files, O_DIRECT, fadvise, batched hint),
                     LRU cache, LFRU tiering, warm cache (Bloom + optional zstd compression),
                     mem hints (hugepages, NUMA pinning), topology probe, perf counters,
                     aligned slab pool
  peregrine-cuda     FFI to cuda/backend_cuda.cu (feature = "cuda")
  peregrine-sched    concurrent MoE scheduler: io_uring streaming ∥ CPU compute
  peregrine-par      persistent scoped worker pool, bit-identical to serial (std-only)
  peregrine-engine   binary `peregrine`: stdio serve protocol, demo, bench, automaton
  peregrine-serve    binary `peregrine-serve`: OpenAI HTTP server + continuous batching
                     (two-tier priority queue, adaptive batch cap, adaptive prefill window)
  peregrine-tools    lib + binary `peregrine-layout-reorg`: offline expert re-layout
                     (greedy / Louvain / spectral / Hilbert, --optimize 2-opt), tier
                     placement (tiers.json), physical checkpoint rewrite (--apply)
  peregrine-token    vendored gigatoken v0.10.0 BPE subset (MIT): SIMD pretokenizers,
                     memoizing BPE engine, HF tokenizer.json loader; GigaTokenizer facade
cuda/                vendored CUDA kernels from colibrì (backend_cuda.cu / .h)
```

Five independent layers of concurrency (all I/O on io_uring; N work-stealing rings;
a data-parallel compute pool; an async GPU stream; prefill/decode interleaving in the
server) are mapped in [`DESIGN.md`](DESIGN.md#concurrency--parallelism-map-where-the-threads-are).

## Build & test

```bash
cargo test --workspace          # 282 tests, CPU-only, no GPU needed
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

# offline routing trace + disk-layout reorg (--method greedy|louvain|spectral):
cargo run --release --bin peregrine -- dump-routes /path/to/model routes.json 512
cargo run --release --bin peregrine-layout-reorg -- \
    --routes routes.json --out /path/to/model --method louvain
#   → writes /path/to/model/schedule.json (loader picks it up automatically)

# tokenizer throughput bench (no weights loaded; needs <model>/tokenizer.json):
cargo run --release -p peregrine-serve -- --model /path/to/model \
    --bench-tokenizer big_text_file.txt
```

`Model::load` accepts any real int4/int8 container model directory in the GLM-5.2
weight-naming scheme (`model.layers.N.self_attn.*`, `mlp.experts.M.*`, …). The
`COLI_MODEL` env var name is kept from colibrì for drop-in compatibility.

### Tuning knobs (env)

All default to sensible values; every one of them affects performance only — the
token stream is unchanged.

#### I/O & streaming
| Var | Default | Effect |
|---|---|---|
| `COLI_IO_RINGS` | 4 | io_uring rings for the streaming lane, each on its own thread |
| `COLI_IO_BATCH` | 16 | expert reads in flight per submit (× 6 regions ≈ 96) |
| `COLI_DIRECT` | off | O_DIRECT lane: DMA straight into aligned buffers, bypassing the page cache |
| `COLI_FADVISE_MAIN` | on | `POSIX_FADV_WILLNEED` batched before every main-path read |
| `COLI_FADVISE_DROP` | off | `POSIX_FADV_DONTNEED` after each streamed read (RSS-bounded runs) |
| `COLI_IO_TUNE` | on | Adaptive `set_iowq_max_workers` from the `IoTuner` EWMA |
| `COLI_IO_RECOVERY` | on | Per-region retry ladder on batched-read failure (transient EIO / EAGAIN / EINTR) |
| `COLI_HUGEPAGE` | on | `MADV_HUGEPAGE` on every ≥ 2 MB allocation |

#### Compute & scheduling
| Var | Default | Effect |
|---|---|---|
| `COLI_PAR_THREADS` | ncpus (≤ 16) | `peregrine-par` pool size; `1` = fully serial (the A/B baseline) |
| `COLI_LANE_BALANCE` | off | `LaneBalancer` overrides static residency: downgrade cold GPU residents to CPU when GPU is bottlenecked |
| `COLI_REPLICATE_K` | 0 | Top-K hottest GPU-residents also warmed into the CPU warm cache each `reheat` |
| `COLI_NUMA_PIN` | off | Pin workers round-robin across NUMA nodes; hierarchical pool dispatch; NUMA-bind ≥ 2 MB buffers |
| `COLI_PERF_COUNTERS` | off | Open a `perf_event_open` LLC-miss counter (needs `perf_event_paranoid ≤ 2`) |
| `COLI_SHAPE_SPECIALIZE` | off | Per-shape probe-then-memoize serial-vs-parallel matmul dispatch |
| `COLI_GPU_F32_FRAC` | unset | Adaptive per-expert precision: hottest fraction of residents promoted to f32 (cuda) |

#### Governors & learning
| Var | Default | Effect |
|---|---|---|
| `COLI_THERMAL_LIMIT_C` | unset | Shrink CPU workers above this package temperature; regrow 8 °C below |
| `COLI_POWER_CAP_W` | unset | Shrink workers when RAPL watts exceed the cap; regrow below 80 % |
| `COLI_BW_GOVERNOR` | off | Shrink workers on a CPU-lane GB/s plateau; periodic regrow probe |
| `COLI_ENTROPY_ADAPT` | off | Routing-entropy-adaptive prefetch breadth (needs `COLI_PREFETCH_TUNE`) |
| `COLI_LEARN_SCHED` | off | ε-greedy bandit over knob configurations; policy persisted |
| `COLI_RL_SCHED` | off | Tabular Q-learning scheduler (bandit wins if both are set) |

#### Scheduling affinity & offline artifacts
| Var | Default | Effect |
|---|---|---|
| `COLI_FUSE_THRESHOLD` | 0.9 | Co-firing rate above which expert pairs stay adjacent in dispatch |
| `COLI_HYPER_SCHED` | off | Group co-activation components into one io-claim window |
| `COLI_TIER_VRAM_MB` / `_RAM_MB` | unset | `galactic`: tier byte budgets → emit `tiers.json` |
| `COLI_TIER_SEED` | on | Prefetch-warm the planned RAM tier at model load |

#### Batching & priority
| Var | Default | Effect |
|---|---|---|
| `COLI_BATCH_SLA_MS` | unset | Shrink working batch cap on p95-latency overrun; regrow on slack |
| `COLI_ADAPTIVE_WINDOW` | 1 | Run prefill every Nth engine tick (decode-heavy window) |
| `X-Peregrine-Priority` | (header) | `high` → drained ahead of normal-priority requests |

#### Cache
| Var | Default | Effect |
|---|---|---|
| `COLI_ECACHE_GB` | 10% avail (cap 2 GiB) | Warm expert RAM cache byte budget |
| `COLI_CACHE_COMPRESS` | off | Zstd-compress warm-cache slabs on admit, decode on hit |
| `COLI_CACHE_COMPRESS_IDLE` | off | Background-recompress cold slots while the engine is idle |
| `COLI_CACHE_NEGATIVE_TTL` | 0 | Evict unhit warm-cache slots older than N clock ticks |
| `COLI_CACHE_ADMIT_MIN_HEAT` | 0 | Admit an expert into the cache only at ≥ N routings (0 = admit all) |

#### Prediction & prefetch
| Var | Default | Effect |
|---|---|---|
| `COLI_PREFETCH_*` | on | Lanes, look-ahead depth, distance tuner, protected set, verification |
| `COLI_PREFETCH_WARM_PATHS_<CLASS>` | unset | Per-workload-class breadth override (CODE / JSON / MATH / PROSE / MIXED); `_HINT_PATHS_<CLASS>` likewise |
| `COLI_ROUTE_HIST_DEPTH` | 4 | K-deep routing history feeding the predictor |
| `COLI_PHASE_THRESHOLD` | 0.6 | Jaccard distance above which `PhaseTracker` flags a shift |

#### Persistence & artifacts
| Var | Default | Effect |
|---|---|---|
| `COLI_ROUTE_STATS_PERSIST` | on | Save `route_stats.json` at Drop; auto-load matching one on `Model::load` |
| `COLI_LAYOUT_SCHEDULE` | on | Consume `<dir>/schedule.json` (from `peregrine-layout-reorg`) to sort disk reads |

#### GPU (feature = "cuda")
| Var | Default | Effect |
|---|---|---|
| `COLI_GPU`, `COLI_GPU_INT4` | off | GPU lane / VRAM expert tier |
| `COLI_CUDA_*` | vary | Tensor Core int4 / W4A16 gates, min row thresholds, async H2D/D2H |

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
