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

## Documentation

The full docs wiki lives in [**docs/**](docs/README.md):

- **Using it** — [getting started](docs/getting-started.md) ·
  [`peregrine` CLI + stdio protocol](docs/cli-peregrine.md) ·
  [HTTP serving](docs/serving.md) · [layout tools](docs/layout-tools.md) ·
  [configuration (all env knobs)](docs/configuration.md) ·
  [model format & artifacts](docs/model-format.md) ·
  [benchmarks](docs/benchmarks.md)
- **How it works** — [architecture](docs/architecture.md) ·
  [the 3-lane scheduler](docs/concurrent-scheduler.md) ·
  [adaptive runtime](docs/adaptive-runtime.md) ·
  [prefetch & caching](docs/prefetch-and-caching.md) ·
  [I/O & storage](docs/io-and-storage.md) · [GPU/CUDA](docs/gpu-cuda.md) ·
  [tokenizer](docs/tokenizer.md)
- **Project** — [testing & quality gates](docs/testing-and-quality.md) ·
  [roadmap & status](docs/roadmap.md) ·
  [peregrine vs colibrì (full study)](docs/peregrine-vs-colibri.md)

[`DESIGN.md`](docs/DESIGN.md) is the original design document; [`todo.md`](docs/todo.md)
is the audited per-item roadmap.

## Status

**482 tests passing, 0 warnings, `cargo clippy` clean** (debug + release). Every
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
faster than HF `tokenizers` on this box** (204 vs 6 MB/s, the per-line serve
pattern; the whole-buffer and parallel-batch paths run 3–7× higher still — see
[docs/tokenizer.md](docs/tokenizer.md#throughput-anatomy)), id-for-id parity-gated
(the HF crate remains only as the test-suite oracle).

The 2026-08-03 wave came from cross-reading two parallel projects solving the same
problem — [WASTE](https://github.com/sqliteai/waste) (C) and
[deltafin](https://github.com/gavamedia/deltafin) (Rust) — both of which publish
their measurements, including the ones that failed. Its headline is the **router
look-ahead**: every predictor peregrine had was a statistic over the router's *past
answers*, and this one asks the router, running layer `L+1`'s own norm and router
against layer `L`'s output to start the next layer's reads during the boundary the
disk would otherwise spend idle. It needs no stored artifact and works on the first
token of a cold process. It cannot change a token — the authoritative router still
decides — so it is on by default. Alongside it: a **predictor scoreboard**
(`COLI_PREDICT_EVAL`) that scores every predictor against the routing that actually
happened, because a correctness-neutral predictor that has degraded to noise costs
throughput silently and no test can catch it; **trunk wiring** (`COLI_MLOCK`);
**cgroup-aware memory budgeting**, since `/proc/meminfo` is not namespaced and
describes the host from inside a container; and a **bounded exact response memo** in
the serve layer. The borrowed *negative* results are recorded too, in
[prefetch & caching](docs/prefetch-and-caching.md#borrowed-negative-results).

See [`todo.md`](docs/todo.md) for the audited roadmap (**~86% strict / ~89% weighted of
136 tracked items**).

| Area | Crate(s) | Status | Validated by |
|---|---|---|---|
| Model loaders | `peregrine-core` | ✅ | config / safetensors index / QT format / dtype round-trips |
| CPU int4 forward | `peregrine-kernels`, `peregrine-model` | ✅ **runs end-to-end** | int8/int4 dots bit-exact on AVX-VNNI; MoE vs f32 ref; attention causality / decode==prefill; full `Model` load→forward→generate |
| io_uring streaming | `peregrine-io` | ✅ | io_uring reads validated byte-for-byte vs `pread` on real hardware (see the skip caveat in [testing](docs/testing-and-quality.md)); priority-weighted LRU cache; **owned-completion lane** forwards each expert as its reads land (`COLI_IO_COMPLETION`, wave path kept as escape hatch); **registered files** (`IOSQE_FIXED_FILE`, wired at model load) + `COOP_TASKRUN`; opt-in **SQPOLL** (`COLI_SQPOLL`); O_DIRECT zero-copy lane |
| CUDA GPU lane | `peregrine-cuda` | ⚙️ FFI complete, host-gated | FFI to the vendored `cuda/backend_cuda.cu` (fused quant matmul, WMMA W4A16, SwiGLU, attention+RoPE) + `nvcc` build.rs behind the `cuda` feature; pinned staging, persistent stream, graph capture/replay (incl. multi-kernel). Default build is a stub — **GPU tests run on an NVIDIA box** |
| Concurrent scheduler | `peregrine-model` (`concurrent.rs`) | ✅ core | `moe_streamed` overlaps io_uring streaming ∥ CPU expert compute; `concurrent.rs` runs N rings with lock-free work-stealing ∥ CPU pool ∥ GPU lane, fixed-order reduce; output == sequential |
| Data-parallel compute | `peregrine-par` | ✅ | persistent scoped pool for rmsnorm / resident MoE / per-row attention / every matmul; **bit-identical to serial** (`f32::to_bits`-exact), work-gated, nesting-safe |
| Prefetch & prediction | `peregrine-model` (`predict.rs`) | ✅ | K-deep `RouteHistory` + momentum / offline transition automaton, per-layer look-ahead emission, EWMA distance tuner, predictive eviction — all correctness-neutral (a wrong guess just re-streams identical bytes) |
| Continuous batching | `peregrine-serve` (`batch.rs`) | ✅ | one engine thread batches all in-flight requests; chunked prefill (64-token chunks) interleaved with decode, bit-identical to whole-prompt prefill (`engine_chunked_prefill_matches_reference`) |
| MLA absorption / MTP / DSA | `peregrine-model` | 🟡 all three opt-in, none measured on a real checkpoint | `mla_attention_absorb` behind `COLI_MLA_ABSORB` (≈ dense within 10% on one call, unmeasured end to end). MTP speculative decode is wired behind `--draft N` / `COLI_DRAFT`, greedy-identical, and refuses loudly without an MTP head. DSA sparse attention behind `COLI_DSA`: bit-identical below `index_topk` and with no indexer in the checkpoint, and — since the batched cores got the `sel` selection — on the served batch path too, not just single-sequence |
| Serve (stdio drop-in) | `peregrine-engine` | ✅ | `READY`/`END` handshake — a drop-in for colibrì's `c/glm` behind `openai_server.py` |
| Serve (native HTTP) | `peregrine-serve` | ✅ | OpenAI-compatible `POST /v1/chat/completions` (SSE + non-streaming), `/v1/models`, `/health`; bearer auth, token caps, graceful shutdown, `#![forbid(unsafe_code)]`, two-tier priority queue via `X-Peregrine-Priority` |
| Adaptive runtime | `peregrine-model` (`lane.rs`, `iotune.rs`, `telemetry.rs`, `workload.rs`) | ✅ | per-lane wall-time accum + `BubbleTuner` EWMA → `LaneBalancer` (CPU/GPU bias-driven downgrade); `IoTuner` adjusts `iowq_max_workers` between forwards; `PredictSource::PhaseAware` (depth-derived boost; `PhaseTracker` is the unwired stateful alternative); per-class prefetch breadth from the serving layer's prompt classifier; heat-threshold cache admission; NUMA worker pinning; real `perf_event_open` LLC-miss counter; cross-session `route_stats.json` at Drop / auto-load on load |
| Offline layout tool | `peregrine-tools` | ✅ | `peregrine-layout-reorg` consumes `dump-routes` JSON, emits `<dir>/schedule.json` via `--method greedy`, `louvain`, or `spectral` (Fiedler ordering); loader picks it up and pre-sorts disk reads |
| Compression | `peregrine-core` (`compress.rs`), `peregrine-io` (`warmcache.rs`) | ✅ | zstd end-to-end on disk (`Blob::with_compression`, header carries the tag + original size); optional transparent zstd on WarmCache admissions (`COLI_CACHE_COMPRESS`) |
| Tokenizer fast path | `peregrine-token`, `peregrine-serve` (`tok.rs`) | ✅ | vendored gigatoken BPE subset (stable toolchain, no libpython); id-for-id parity vs HF `tokenizers` on the committed GPT-2 fixture; cross-request memo cache; sole runtime tokenizer (HF crate is test-oracle only); `--bench-tokenizer` (34× locally) |

### Not yet done
**Ten of the nineteen open items need hardware this workspace lacks**: CUDA Graphs
wired into the decode loop, persistent CUDA kernels, GPU-side fused reduce and a
`cudaMallocAsync` pool (`nvcc` + GPU); idle-cycle GPU compute (GPU); GPUDirect
Storage (vendor stack); multi-GPU expert ownership / NVLink placement / VRAM
replication (≥ 2 GPUs); distributed inference (multiple hosts).

**Eight need no hardware at all** — fusing prefill into the decode batch, KV
quantization, paged KV, int2 checkpoint conversion, and heat-tiered on-disk
precision. Each is a substantial change to a core invariant rather than a blocked
one, and that is where the remaining throughput is: caching and prefetch plateau
on this workload, so the wins left are in moving *fewer* bytes per token, not in
moving 11.3 GB faster. (The 0.6 % figure often cited for this is a **warm-cache
hit rate**, not a routing statistic — see the correction in
[the study](docs/peregrine-vs-colibri.md#52-cache--locality-analysis-peregrine-measured)
and measure the routing quantity with `peregrine route-stats`.)

One item is open **by choice rather than by hardware**: CPU/GPU split GEMM. The
plumbing is small, but the CPU half computes int4 and the GPU half f32, and a
split point derived from wall-clock timings would make low-order output bits
depend on machine timing — the same prompt giving different logits run to run.
See [`todo.md`](docs/todo.md).

## Architecture

```
crates/
  peregrine-core     formats: Cfg, safetensors index (with zstd), QT quant detect, dtype, pack, compress
  peregrine-kernels  std::arch int8/int4 dots + matmuls (scalar ref + AVX2/AVX-VNNI)
  peregrine-model    MLA attention, router, MoE, sampler, MTP, prefetch prediction,
                     the byte ledger, tolerance-keyed state fingerprints, lane
                     telemetry + bubble tuner + lane balancer, IoTuner, PhaseTracker,
                     WmmaTuner, PlanOptimizer, the N-ring concurrent lane
                     (concurrent.rs), top-level Model
  peregrine-io       io_uring Reactor (registered files, O_DIRECT, fadvise, batched hint),
                     per-read latency histogram + fault split, device geometry probe,
                     priority-weighted LRU cache, warm cache (Bloom + optional zstd),
                     mem hints (hugepages, NUMA pinning), topology probe, perf counters,
                     aligned slab pool
  peregrine-cuda     FFI to cuda/backend_cuda.cu (feature = "cuda")
  peregrine-sched    two-lane streaming ancestor — NOT linked by any crate, but
                     kept as the cross-engine correctness oracle: its
                     streamed_matches_the_production_concurrent_path runs
                     moe_streamed and the live moe_forward_concurrent over the
                     same container bytes. The live 3-lane path is
                     peregrine-model/concurrent.rs
  peregrine-par      persistent scoped worker pool, bit-identical to serial (std-only)
  peregrine-engine   binary `peregrine`: stdio serve protocol, demo, bench, automaton
  peregrine-serve    binary `peregrine-serve`: OpenAI HTTP server + continuous batching
                     (two-tier priority queue, adaptive batch cap, adaptive prefill window)
  peregrine-tools    lib + binaries. `peregrine-basisfit`: cross-expert factorization
                     priced as rate-distortion on activations, with a shuffled-grouping
                     control. `peregrine-layout-reorg`: offline expert re-layout
                     (greedy / Louvain / spectral / Hilbert, --optimize 2-opt), tier
                     placement (tiers.json), physical checkpoint rewrite (--apply)
  peregrine-token    vendored gigatoken v0.10.0 BPE subset (MIT): SIMD pretokenizers,
                     memoizing BPE engine, HF tokenizer.json loader; GigaTokenizer facade
cuda/                vendored CUDA kernels from colibrì (backend_cuda.cu / .h)
```

Five independent layers of concurrency (all I/O on io_uring; N work-stealing rings;
a data-parallel compute pool; an async GPU stream; prefill/decode interleaving in the
server) are mapped in [`DESIGN.md`](docs/DESIGN.md#concurrency--parallelism-map-where-the-threads-are).

## Build & test

```bash
cargo test --workspace          # 482 tests, CPU-only, no GPU needed
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
token stream is unchanged. (Annotated reference with deep-dive links:
[docs/configuration.md](docs/configuration.md).)

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
| `COLI_PERF_COUNTERS` | off | LLC-miss counter on the decode thread; `[perf] llc-misses=N` at shutdown (that thread only) |
| `COLI_IO_LATENCY` | off | Per-read submit→complete **histogram** (not an EWMA — a mean is what a GC tail survives) plus the thread's page-fault delta. Reports p99/p50 and max/p50, and names undersampled quantiles instead of printing them |
| `COLI_UNION_STATS` | off | Batch-union sharing, and the **byte ledger**: `requested` / `unique after union` / `cache-served` / `from disk` / `prefetch waste`, with the flip-rate gate named beside every saving |
| `COLI_DEBUG` | off | Surface advisory-operation failures (madvise/fadvise hints, NUMA pinning, route-stats persistence) on stderr |
| `COLI_SHAPE_SPECIALIZE` | off | Per-shape probe-then-memoize serial-vs-parallel matmul dispatch |
| `COLI_GPU_F32_FRAC` | unset | Adaptive per-expert precision: hottest fraction of residents promoted to f32 (cuda) |
| `COLI_PCIE_BUDGET_MB` | unlimited | Cap on per-`reheat` PCIe upload bytes (cuda) |
| `COLI_PREFILL_CHUNK_DIV` | 4 (`0` = fixed 64) | Adaptive prefill chunk `pos/d`; output-neutral, cuts quadratic KV reconstruction |
| `COLI_GATE_STATS` | off | Tally negligible-gate-share routed experts (`[gate]` line) |
| `COLI_PREFIX_CACHE_MB` | 2048 (`0` = off) | Cross-request KV prefix cache budget; caches prompts *and* generated tokens; a hit shares the prefix by refcount, it is not copied |
| `COLI_KV_BUDGET_MB` | 0 (off) | Resident-KV byte ceiling for admission, alongside the `--max-batch` count |
| `COLI_KV_DTYPE` | `f32` | KV latent element type; `f16` halves resident KV (**changes token values** — pair with `COLI_MLA_ABSORB`) |
| `COLI_GPU_DENSE` | off | Place a **dense** model's MLP weights in VRAM and compute them there (refused on a checkpoint with routed experts). **Changes token values** — the device path is *more* accurate (rms 1.1e-7 vs the CPU path's 3.0e-3), so which layers are resident changes the output. Gated on the value: `=0` means off |
| `COLI_GPU_DENSE_LAYERS` | fit to VRAM | Pin the resident layer count. **Required for any measurement arm** — fitting to free VRAM makes placement depend on whatever else holds the card, so two boots can differ |
| `COLI_GPU_DENSE_HEADROOM_MB` | 1024 | VRAM the dense tier leaves for activations and the context |
| `COLI_KV_STORE_DIR` / `_MB` / `_TRIM` | unset | Disk-persisted KV sessions: completed prefixes ≥256 tokens checkpoint here and a restarted server restores them instead of re-prefilling. The prefix cache's disk extension |
| `COLI_IO_DEVICE_SCHED` | off | Device-aware ring scheduling — device-pure claim groups with cross-device work stealing, instead of one device-blind cursor. Needs streaming, >1 ring, and shards on >1 device |
| `COLI_TOPIC_ROUTING` / `_HALFLIFE` | off / 512 | Per-`TokenClass` residency steering: cache tiebreaks prefer experts this topic has routed before, with profiles decayed at a rate scaled by routing entropy. Streaming only — routing is recorded on the streaming path alone |
| `COLI_PREFETCH_STALE_DROP` / `_SLACK` | off / 1 | Drop a queued speculative warm before its disk read once the sweep has moved past it. At B=16, 98.6 % of speculative reads arrived too late to use. Advisory lane only, so output is untouched |
| `COLI_QWEN_THINK` | off | Keep Qwen's `<think>` block in the response. Off pre-closes it — with it open, a run whose budget expired before the closing tag rendered as an **empty completion** |
| `COLI_ACT_F32` | off | Quantized matmuls against f32 activations instead of int8 (**changes token values** — more accurate, still different) |
| `COLI_DSA` | off | DSA lightning-indexer sparse attention, where the checkpoint carries an indexer (**changes token values**) |
| `COLI_SPEC_GDN` | off | Speculation on a **recurrent** arch (Qwen3.5-hybrid). A linear-attention layer keeps a delta-rule state, not rows, so a rejected draft is rolled back by snapshot/restore + a re-advance over the accepted rows instead of a KV truncate. Output-neutral (`accept_run` still decides by argmax identity); the cost is a ~151 MB/sequence snapshot per drafting tick, reported as `spec.gdn_snapshot_bytes` |
| `COLI_SPEC_GDN_MAX_B` | 0 (uncapped) | Batch width above which `COLI_SPEC_GDN` stops drafting — the snapshot is charged per sequence while a forward's weight read is shared, so it stops paying at some width |
| `COLI_SPEC_UNION_MAX` | 0 (off) | Ceiling on a tick's projected routed-expert union, in expert-read requests — the cost-side twin of `COLI_SPEC_CONF`'s acceptance-side floor. Depth-only, so a greedy stream is bit-identical at any setting; may stop speculation, never stops decoding. Inert on a resident model |
| `COLI_DRAFT_NGRAM` | 0 (off) | Prompt-lookup drafting: propose whatever followed the last occurrence of the current suffix (match length ≤ n, floor 2). No weights, no forward pass — a `memcmp` where an MTP draft step is a sparse-MoE layer. Takes priority over the head when it matches; greedy requests only; needs `COLI_DRAFT` for the depth but **not** an MTP head. Reported separately as `ngram` on `/metrics` |
| `COLI_RLM` | off | Recursive Language Model: refine contested decode positions by re-running the last `COLI_RLM_LAYERS` layers (default 4) up to `COLI_RLM_DEPTH` times (default 2). Greedy recurses when top-2 logit gap < `COLI_RLM_MARGIN` (0.1). Pass-2 experts come from the warm cache, not SSD. Composes with `COLI_DRAFT`: recurses only at the post-acceptance contested position (**changes token values**) |
| `COLI_ROUTE_MIN_SHARE` | 0 (off) | Drop negligible-gate-share routed experts (**changes token values** — measured 2026-08-13: flips 21–28 % of top-1 at τ=0.02–0.05, so the byte saving is priced out; see bench-data/2026-08-13-route-min-share) |

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
| Warm-cache hit rate, sustained decode (10 GB cache) | 0.6 % (measured) | — |

Both are **disk-bandwidth-bound**; colibrì is currently
~1.4× faster at raw *single-sequence* streaming (deeper io_uring queue), while
peregrine adds a verified warm-cache/scheduler stack and memory safety.

### Per-read latency has a tail p99 alone does not see — measured 2026-08-21

`COLI_IO_LATENCY=1`, 2400 expert reads on the real container, B=1 cold:

| | |
|---|---:|
| median | 41.0 ms |
| p90 | 114.7 ms |
| p99 | 262.1 ms |
| **p99.9 / max** | **3.38 s** |
| p99/p50 | 6.4× |
| **max/p50** | **82.6×** |
| minor / major page faults | **0 / 0** |

p99/p50 is under 10× — "flat" by that measure. The slowest read is **82.6× the
median**. The `max/p50` column is what catches it, and the report says "rare
stalls p99 cannot resolve at this sample count" rather than "no tail". Zero page
faults over 2400 reads, so none of this is host fault handling; it is queueing
plus device time, and separating those needs a fixed-queue-depth arm.

### 11.3 GB/token is not one number — measured 2026-08-21

`COLI_UNION_STATS=1` prints a `[ledger]` block decomposing the figure this README
has been quoting. On the real GLM-5.2 int4 container, 15 tokens over B=1 and B=4:

| column | GB/token | |
|---|---:|---|
| **requested** | **11.349** | the arithmetic figure — routed selections × expert size |
| unique (union) | 7.188 | 36.7 % removed by the batch union |
| cache-served | — | 3.9 % of unique |
| **from disk** | **6.905** | what the drive actually moved |
| prefetch waste | — | 12.8 % of disk traffic |

The arithmetic figure is confirmed. **The disk traffic is 39 % lower**, and this
README previously used the two interchangeably — quoting `600 experts ≈ 11
GB/token` in a sentence about disk bandwidth overstates it by 1.64×. The
`from disk` row is the one an operator feels. One column, bytes re-read after a
wrong eviction, prints `NOT MEASURED` rather than being folded in silently. See
[docs/measurement.md](docs/measurement.md#the-byte-ledger--113-gbtoken-is-not-one-number).

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
three-lane scheduler, milestones) is in [`DESIGN.md`](docs/DESIGN.md).

- Upstream: [JustVugg/colibri](https://github.com/JustVugg/colibri) · fork:
  [s-b-repo/colibri](https://github.com/s-b-repo/colibri)
- Port sources / correctness anchors (in colibrì's `c/`): `glm.c` (MoE, MLA
  `attention_rows`, IDOT kernels, router, `spec_decode`), `st.h`, `uring.h`,
  `tier.h`, `backend_cuda.h/.cu` (vendored here under `cuda/`), `openai_server.py`,
  and `ref_glm.json` + `tools/make_glm_oracle.py` (the token-exact oracle gate).

## License

MIT, inherited from colibrì.
