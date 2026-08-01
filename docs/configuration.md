[« Docs index](README.md)

# Configuration reference

Every knob is an environment variable. All default to sensible values, and —
the core invariant — **every one of them affects performance only: the token
stream is unchanged**. Boolean gates accept `1`/`0` (and `false` where noted);
`(on)` means enabled by default.

The `COLI_` prefix is kept from colibrì for drop-in compatibility.

## Core

| Var | Default | Effect |
|---|---|---|
| `COLI_MODEL` | unset | model directory (serve mode; wins over the positional arg; the only source for `bench`) |
| `COLI_STREAM` | auto | `1`/`0` force/disable expert streaming (auto-decided from available RAM vs model size) |
| `COLI_DEBUG` | off | surface advisory (non-fatal) failures on stderr |
| `COLI_NO_ARENA_CAP` | off | skip the automatic `M_ARENA_MAX=2` malloc-arena cap (also skipped if `MALLOC_ARENA_MAX` is set) |

## I/O & streaming

| Var | Default | Effect |
|---|---|---|
| `COLI_IO_RINGS` | 4 | io_uring rings for the streaming lane, each on its own thread |
| `COLI_IO_BATCH` | 16 | expert reads in flight per submit (× 6 regions ≈ 96) |
| `COLI_DIRECT` | off | O_DIRECT lane: DMA straight into aligned buffers, bypassing the page cache |
| `COLI_REGBUF` | — | **not wired.** `register_read_buffers()`/`IORING_OP_READ_FIXED` exist and are tested, but no code reads this variable and the streaming path always uses the plain read. Listed here because operators have set it expecting an effect |
| `COLI_FADVISE_MAIN` | on | `POSIX_FADV_WILLNEED` batched before every main-path read |
| `COLI_FADVISE_DROP` | off | `POSIX_FADV_DONTNEED` after each streamed read (RSS-bounded runs) |
| `COLI_IO_TUNE` | on | adaptive `set_iowq_max_workers` from the `IoTuner` EWMA |
| `COLI_IO_RECOVERY` | on | per-region retry ladder on batched-read failure (transient EIO/EAGAIN/EINTR) |
| `COLI_HUGEPAGE` | on | `MADV_HUGEPAGE` on every ≥ 2 MB allocation |

## Compute & scheduling

| Var | Default | Effect |
|---|---|---|
| `COLI_PAR_THREADS` | ncpus (≤ 16) | `peregrine-par` pool size; `1` = fully serial (the A/B baseline) |
| `COLI_LANE_BALANCE` | off | `LaneBalancer` overrides static residency: downgrade cold GPU residents to CPU when GPU is bottlenecked |
| `COLI_REPLICATE_K` | 0 | top-K hottest GPU-residents also warmed into the CPU warm cache each `reheat` |
| `COLI_NUMA_PIN` | off | pin workers round-robin across NUMA nodes; hierarchical pool dispatch; NUMA-bind ≥ 2 MB buffers |
| `COLI_PERF_COUNTERS` | — | **not wired.** The counter (`PerfCounter::open_cache_misses`) works, but `open_l3_miss_counter` has no caller, so setting this has no effect. Listed because the docs advertised it as live |
| `COLI_SHAPE_SPECIALIZE` | off | per-shape probe-then-memoize serial-vs-parallel matmul dispatch |

## Governors & learning

| Var | Default | Effect |
|---|---|---|
| `COLI_THERMAL_LIMIT_C` | unset | shrink CPU workers above this package temperature; regrow 8 °C below |
| `COLI_POWER_CAP_W` | unset | shrink workers when RAPL watts exceed the cap; regrow below 80 % |
| `COLI_BW_GOVERNOR` | off | shrink workers on a CPU-lane GB/s plateau; periodic regrow probe |
| `COLI_ENTROPY_ADAPT` | off | routing-entropy-adaptive prefetch breadth (needs `COLI_PREFETCH_TUNE`) |
| `COLI_LEARN_SCHED` | off | ε-greedy bandit over knob configurations; policy persisted in `route_stats.json` |
| `COLI_RL_SCHED` | off | tabular Q-learning scheduler (the bandit wins if both are set) |

## Scheduling affinity & offline artifacts

| Var | Default | Effect |
|---|---|---|
| `COLI_FUSE_THRESHOLD` | 0.9 | co-firing rate above which expert pairs stay adjacent in dispatch |
| `COLI_HYPER_SCHED` | off | group co-activation components into one io-claim window |
| `COLI_TIER_VRAM_MB` / `COLI_TIER_RAM_MB` | unset | `galactic`: tier byte budgets → emit `tiers.json` |
| `COLI_TIER_SEED` | on | prefetch-warm the planned RAM tier at model load |
| `COLI_LAYOUT_SCHEDULE` | on | consume `<dir>/schedule.json` (from `peregrine-layout-reorg`) to sort disk reads |
| `COLI_ROUTE_STATS_PERSIST` | on | save `route_stats.json` at Drop; auto-load a matching one at `Model::load` |

## Batching & priority (serve layer)

| Var | Default | Effect |
|---|---|---|
| `COLI_BATCH_SLA_MS` | unset | shrink the working batch cap on latency overrun; regrow on slack |
| `COLI_ADAPTIVE_WINDOW` | 1 | run prefill every Nth engine tick (decode-heavy window) |
| `X-Peregrine-Priority` | (HTTP header) | `high` / `1` / `true` → drained ahead of normal-priority requests |

## Cache

| Var | Default | Effect |
|---|---|---|
| `COLI_ECACHE_GB` | 10 % avail (cap 2 GiB) | warm expert RAM cache byte budget |
| `COLI_CACHE_COMPRESS` | off | zstd-compress warm-cache slabs on admit, decode on hit |
| `COLI_CACHE_COMPRESS_IDLE` | off | background-recompress cold slots while the engine is idle |
| `COLI_CACHE_NEGATIVE_TTL` | 0 | evict unhit warm-cache slots older than N clock ticks |
| `COLI_CACHE_ADMIT_MIN_HEAT` | 0 | admit an expert into the cache only at ≥ N routings (0 = admit all) |

## Prediction & prefetch

| Var | Default | Effect |
|---|---|---|
| `COLI_PREFETCH_LOOKAHEAD` | on | per-layer look-ahead emission depth |
| `COLI_PREFETCH_LANES` | on | parallel-async prefetch-lane pool size (batched serving) |
| `COLI_PREFETCH_WARM_PATHS` / `COLI_PREFETCH_HINT_PATHS` | on | warm-tier / fadvise-hint-tier breadth |
| `COLI_PREFETCH_WARM_PATHS_<CLASS>` / `COLI_PREFETCH_HINT_PATHS_<CLASS>` | unset | per-workload-class breadth override (`CODE` / `JSON` / `MATH` / `PROSE` / `MIXED`) |
| `COLI_PREFETCH_TUNE` / `COLI_PREFETCH_DIST` / `COLI_PREFETCH_DIST_MAX` | on | EWMA distance tuner over prefetch used/wasted |
| `COLI_PREFETCH_PROTECT` | on | predictor ∪ hot experts protected from eviction (opaque cache priority) |
| `COLI_PREFETCH_VERIFY` | off | re-read + byte-compare each speculative load; accuracy log at shutdown |
| `COLI_ROUTE_HIST_DEPTH` | 4 | K-deep routing history feeding the predictor |
| `COLI_PHASE_THRESHOLD` | 0.6 | Jaccard distance above which `PhaseTracker` flags a shift |

## GPU (feature = `cuda`)

| Var | Default | Effect |
|---|---|---|
| `COLI_GPU` | off | enable the GPU lane |
| `COLI_GPU_INT4` | off | int4 VRAM expert tier (needs per-row int4 experts) |
| `COLI_GPU_F32_FRAC` | unset | adaptive per-expert precision: hottest fraction of residents promoted to f32 |
| `COLI_PCIE_BUDGET_MB` | unlimited | cap on bytes one `reheat` generation may upload across PCIe; the coldest deferred to the next generation. Unset = unlimited = unchanged behaviour |
| `COLI_PREFILL_CHUNK_DIV` | 0 | prefill chunk becomes `max(64, pos/d)`. A fixed chunk makes prefill quadratic in prompt length, since attention re-derives every cached position per call. Chunk size cannot change output — only how long one prefill step blocks decode |
| `COLI_GATE_STATS` | off | tally, per routed expert, whether its gate share is below 0.5/1/2/5% of its position's mass; printed as `[gate]` at shutdown. Diagnostics only |
| `COLI_PREFIX_CACHE_MB` | 0 | byte budget for the cross-request KV prefix cache. A new request is seeded from the longest cached prefix of its prompt, so a shared system prompt is prefilled once rather than once per request. Entries are matched by comparing tokens, not hashing. `0` disables it |
| `COLI_ROUTE_MIN_SHARE` | 0 | drop trailing routed experts carrying less than this share of a position's gate mass, renormalizing the survivors. **Unlike every other knob here, this one changes token values** — it removes a real (if small) term from the MoE sum. Size it with `COLI_GATE_STATS` and gate it with `Model::prediction_flip_rate`. `0` disables it |
| `COLI_CUDA_PROFILE` | off | accumulate per-call H2D/kernel/D2H timings |
| `COLI_CUDA_TC_*` | vary | Tensor Core int4/W4A16 gates, min-row thresholds |

## Bench & misc

| Var | Default | Effect |
|---|---|---|
| `COLI_BENCH_STEPS` | 3 | decode steps per batch size in `peregrine bench` |
| `PEREGRINE_API_KEY` | unset | bearer-auth key for `peregrine-serve` (same as `--api-key`) |

Deep dives on what the knobs actually steer:
[Adaptive runtime](adaptive-runtime.md) ·
[Prefetch & caching](prefetch-and-caching.md) ·
[I/O & storage](io-and-storage.md) · [GPU](gpu-cuda.md) ·
[Serving](serving.md).
