# Rust rewrite of colibrì: true CPU∥GPU∥SSD∥RAM concurrency

> This is the design document for **peregrine** — the Rust spin-off of colibrì.
> It is preserved as originally written; references to `rust/` map to this repo's
> root and references to `c/` are to the upstream [colibrì](https://github.com/JustVugg/colibri)
> engine (the CUDA source is vendored here under `cuda/`).

## Context

the engine use RAM, CPU, GPU, and SSD in
parallel *at the same time* to improve throughput, and to build a Rust version that uses
io_uring to cut syscalls.

**What the research found:**

- **io_uring already exists in the C engine** (`URING=1`, Linux-only, `c/uring.h`). It is a
  hand-rolled raw-syscall ring (no liburing), batches up to 512 reads / 64 expert-loads per
  submit, forces `IOSQE_ASYNC`, and caps io-wq workers. It already replaces the blocking
  loader threads on the expert path. A Rust version *re-implements* this idea rather than
  inventing it.
- **CPU + GPU + SSD + RAM overlap already partly exists.** `PIPE=1` overlaps disk reads with
  matmul; `CUDA_DENSE`/`COLI_CUDA_ATTN` put dense/attention on the GPU while the CPU streams
  experts; a three-tier VRAM/RAM/disk hierarchy is implemented (`pin_load`, `resource_plan.py`,
  `tier.h`). The documented optimization stack already reaches **1.41 tok/s (4.3×)**.
- **The one real gap — the throughput lever.** Under CUDA, the MoE inner loop is *phased, not
  concurrent*: VRAM-resident experts are collected and deferred (`c/glm.c:3079-3081`), RAM/disk
  experts are computed on the CPU **inline** (`c/glm.c:3084-3101`), and the GPU expert group is
  dispatched **only after** that loop finishes (`c/glm.c:3124-3127`). The in-code comment
  measures the waste: *"9343 experts in VRAM sat unused during prefill — 81s of expert-matmul
  all on CPU, GPU groups 21ms total"* (`c/glm.c:2921-2923`). So on the same MoE block, CPU-expert
  compute and GPU-expert compute never run at once. Metal has a GPU∥disk overlap that CUDA lacks.

**Chosen direction** A **full from-scratch Rust engine**, targeting
**Linux + NVIDIA CUDA**, whose goal is to **beat current tok/s** by closing that phased gap —
a completion-driven scheduler where the GPU lane, CPU lane, and io_uring SSD lane all drain the
same layer's experts simultaneously. Byte-exact parity with the C engine is **not** required:
the project already documents that batched/GPU forwards round differently and that the bar is
"every emitted token is the argmax of a *valid* forward" (`README.md:60`). We hold that bar and
validate token-exactness on the single-token CPU path against the existing `transformers` oracle.

**Intended outcome:** a `rust/` cargo workspace producing a `peregrine-engine` binary that is a
drop-in for `c/glm` (same stdio serve protocol), matches the reference architecture semantics,
and delivers higher decode/prefill tok/s than the C engine on the same NVIDIA box by running all
four resources concurrently.

---

## Architecture — the concurrent 3-lane scheduler (the centerpiece)

Per MoE layer, after routing (Phase A) and batch-union dedup (Phase B), each **unique** expert
becomes exactly one `ExpertTask` (this structurally enforces the "compute each expert once, apply
to all its rows" invariant — stronger than the C `seen[]` scan). Tasks are classified O(1) by
residency into three lanes that run **as concurrent actors, not sequential phases**:

- **GPU lane** — one dispatcher thread per CUDA device. Consumes VRAM-resident tasks *and*
  disk-miss tasks the I/O lane promotes after a read completes. Coalesces ready tasks into
  `coli_cuda_expert_group`-style calls on the device's non-blocking stream, double-buffered
  pinned staging so H2D(n+1) ∥ kernel(n) ∥ D2H(n−1). Driven **continuously** via new
  non-syncing `_async` entry points + a CUDA event per batch (vs. the ABI's per-call
  `cudaStreamSynchronize`).
- **CPU lane** — a physical-core-pinned pool (`crossbeam` deque + `core_affinity`), outer
  parallelism over *experts* (inverting the C engine, which runs experts serially with each
  matmul internally OMP-parallel). Each worker runs one expert's SwiGLU: fused gate+up → silu·up
  → down → weighted scatter. Inner O-row tiling stays SIMD, no nested threading.
- **I/O lane** — one reactor thread owning the `io-uring` ring. Submits the coalesced ~19 MB
  gate/up/down read + 3 scale reads with `IOSQE_ASYNC` (mirroring `uring_load_add`,
  `c/glm.c:1863`). **On each CQE completion it never blocks** — it stamps the task's slab and
  routes the now-ready expert to whichever compute lane is shorter (GPU if VRAM staging free,
  else CPU). This is the CPU∥GPU∥SSD hand-off the C engine lacks.

**Accumulation (correct + non-serializing):** top-K means two experts can write the same output
row, so raw scatter races. Decode (small S): tiny lane-private accumulators summed once at layer
end (no locks). Prefill (large S): per-expert contiguous `partial` staging + one indexed reduce
sorted by `(row, expert-rank)` → deterministic run-to-run. A single atomic `remaining` counter
signals layer completion.

**PILOT prefetch, concurrent:** L+1's predicted-router experts are submitted into the *same*
ring as low-priority speculative reads tagged with a distinct `user_data` band; slab-arena
generation tags (the lock-free idea from `PipePool`'s gen-tagged cursor, `c/glm.c:2010`) replace
the C mutex + inflight barrier, so a straggler speculative load can never write a wrong-generation
slab.

**Why it beats the phased design:** for a decode block of 3 VRAM + 2 RAM + 3 disk experts, the C
wall-clock ≈ `max(disk_chain, cpu_5_experts) + gpu_3_experts` (GPU idle during the CPU phase); the
Rust wall-clock ≈ `max(gpu_lane, cpu_lane, disk_lane)` — the slowest single lane, not the sum.

### Concurrency & parallelism map (where the threads are)

Five independent layers of concurrency, so this isn't re-requested:

- **I/O is io_uring everywhere.** Config, safetensors headers, and *all* weight loading go through the
  reactor (`peregrine-core/src/safetensors.rs::read_at` → `Mutex<Reactor>`); per-token expert streaming
  is the hot path. Only `tokenizer.json` + `/proc/meminfo` are synchronous (one-time, negligible).
- **The streaming MoE lane is N parallel io_uring rings with lock-free work-stealing.**
  `peregrine-model/src/concurrent.rs` runs `COLI_IO_RINGS` rings (default 4), each on its own thread,
  atomically claiming expert batches off an `AtomicUsize` cursor (`io_work.fetch_add`) and issuing a deep
  batched submit (~96 in-flight reads). A CPU worker pool computes SwiGLU as bytes land; an optional GPU
  lane runs one batched `expert_group`; a single deterministic fixed-order reduce merges them
  (bit-identical to serial). This *already is* "multiple io_uring reading multiple experts in parallel."
- **Expert reads are zero-copy into the weight.** A landing region is a `peregrine_io::Bytes` that the
  streamed `QtWeight` *moves* in (no copy), and every kernel reads it as `&[u8]` via `Deref`. The buffered
  lane has the kernel fill the caller's `Vec` directly; the O_DIRECT lane (`COLI_DIRECT`) DMAs each
  region's 4096-aligned superset straight into an owned `AlignedBuf` and exposes the exact
  `[off, off+len)` sub-slice (`Reactor::read_direct_aligned`) — so bulk weight bytes are never memcpy'd in
  userspace on either path, and O_DIRECT bypasses the page cache with no realignment copy.
- **The resident compute path is data-parallel on a persistent pool** (`peregrine-par`): `rmsnorm_rows`,
  resident `moe_forward` (per-expert compute, serial scatter), per-row attention, and every matmul
  (`QtWeight::apply_vec`) run on a process-global scoped thread pool. It is **bit-identical** to serial
  (row/expert-independent + fixed-order reduces; `f32::to_bits`-exact tests), **work-gated** (tiny
  matrices stay serial — no small-batch regression), and **nesting-safe** (an expert's matmul inside a
  parallel MoE runs serial via a thread-local guard, so the fixed pool can't deadlock). `COLI_PAR_THREADS`
  overrides the size (`1` = fully serial).
- **The GPU backend is async**: `expert_group` uses pinned staging + a persistent non-blocking stream,
  with CUDA graph capture/replay available (`peregrine-cuda`).
- **The serving engine interleaves prefill with decode.** `peregrine-serve/src/batch.rs` queues a new
  request and advances its prompt one `PREFILL_CHUNK` (64 tokens) at a time, round-robin, *between*
  batched decode steps — so admitting a long prompt never stalls the in-flight batch for the whole
  prefill. Chunked prefill is bit-identical to whole-prompt prefill (same causal KV build), asserted by
  `engine_chunked_prefill_matches_reference`. A two-tier priority queue drains high-priority
  admissions ahead of normal, and a latency-SLA hook shrinks / grows the working batch cap between
  ticks (`COLI_BATCH_SLA_MS`).

---

## Adaptive runtime — the feedback loop over the 3-lane scheduler

The 3-lane scheduler is stateless per-forward; the adaptive-runtime wave (2026-07-30) layers a
telemetry → tuner → placement feedback loop on top of it. Every knob is env-gated and defaults to
the historical behavior so existing benchmarks stay bit-identical, and every override is
correctness-neutral (only latency and residency change, never the reduced values).

- **Lane telemetry** — `moe_forward_concurrent` brackets each of the three lanes and the reduce
  phase with `Instant::now()` and bumps atomic counters on a shared `LaneTimingsAccum` threaded
  through `ForwardCtx`. Between forwards, `Model::publish_lane_timings` swap-resets the accumulator,
  stores the sample on `Model::last_lane_timings`, and folds it into the tuners below.
  (`crates/peregrine-model/src/lane.rs`, `concurrent.rs`, `model.rs`.)
- **Bubble tuner + lane balancer** — `BubbleTuner` maintains an EWMA per lane (α = 0.3), declares a
  `Bias::Toward{Cpu,Gpu,Io}` when the top lane exceeds `1.5 × max(others)` for `k = 3` consecutive
  forwards (hysteresis defeats one-off spikes), and publishes it. `LaneBalancer::choose(gpu_resident,
  heat)` reads that bias inside the scheduler: on `Bias::TowardGpu` a cold GPU-resident expert
  downgrades to the CPU lane; on `Bias::TowardCpu` a hot streamed expert is a candidate to spill
  onto the GPU (spill-upgrade path is reserved — requires on-demand upload machinery). Gated on
  `COLI_LANE_BALANCE`.
- **Runtime expert replication** — `Model::enqueue_expert_replicas(K)` (called from `reheat`) takes
  the top-K hottest currently-VRAM-resident experts and enqueues prefetch reads for them so their
  bytes also land in the CPU warm cache. Composed with the balancer above, a `TowardGpu` downgrade
  serves the resident-but-now-CPU expert straight from RAM — no disk read. Gated on
  `COLI_REPLICATE_K`.
- **IoTuner** — mirrors the `PrefetchTuner` shape: EWMA over per-forward `io_us`, SQ-full-driven
  halving, target-driven slow grow. `Model::publish_lane_timings` applies the recommendation via
  `Reactor::set_iowq_max_workers` on every ring, deduped against the last-applied cap.
- **Adaptive batching & priority** — `EngineHandle` holds two `mpsc::UnboundedSender`s
  (`Priority::High`, `Priority::Normal`); the engine drains high first each tick. A `recv_priority`
  helper uses a small current-thread runtime for the biased-`select!` blocking wait. Per-forward
  decode wall time is EWMA-tracked and, when `COLI_BATCH_SLA_MS` is set, drives a working-cap
  shrink/grow. `COLI_ADAPTIVE_WINDOW=N` runs prefill every Nth tick so decode gets more consecutive
  time before yielding.
- **Phase-aware prefetch** — `PredictSource::PhaseAware` wraps any inner source and, when the
  Jaccard distance between the newest two frames of `RouteHistory` exceeds `threshold_bp / 10000`,
  folds a heavy vote on the newest frame's experts. `PhaseTracker` maintains the same EWMA
  standalone for consumers that want a raw signal (batching engine, etc.).
- **Workload classification** — `workload::classify_str` buckets a tokenizer-decoded tail into
  `TokenClass::{Prose, Code, Json, Math, Mixed}` from ratios of alnum / punct / digits / brace
  shapes. Wired end-to-end: the HTTP handler classifies the last user message's tail (UTF-8-safe
  512-char cap), stamps `EngineRequest.class`, admission calls `Model::set_workload_class`, and
  every prefetch-context builder resolves breadth through `PrefetchPolicy::for_class`
  (`COLI_PREFETCH_WARM_PATHS_<CLASS>` / `_HINT_PATHS_<CLASS>`, falling back to the base knobs).
  Latest-admission-wins for a mixed batch (breadth is batch-global today).
- **Cross-session persistence** — `RouteHistory::to_json`/`from_json` + `HeatTable::restore` +
  `Model::save_route_stats_here` on `Drop` and `Model::try_load_route_stats` at load: `<dir>/route_stats.json`
  survives a process restart so prefetch/residency start warm on the previous session's routing.
  Config-tag guarded to reject stale artifacts. Same shape as the offline transition automaton
  (`automaton.json`).
- **Warm cache extras** — 2048-bit Bloom over resident keys short-circuits `WarmCache::get`'s
  miss path; a negative-TTL pass (`COLI_CACHE_NEGATIVE_TTL`) evicts unhit slots early; transparent
  zstd (`COLI_CACHE_COMPRESS`, `SlotBytes::Compressed { six, orig_lens }`) shrinks resident bytes
  by ~2-3× at the cost of one decode per hit. Two admission/maintenance companions: a
  **heat-threshold admission gate** (`COLI_CACHE_ADMIT_MIN_HEAT` — an expert is cached only once
  its routing heat reaches N, filtering one-off experts; heat is bumped post-reduce so N=1 means
  "cache from the second routing on"), and **idle-tick background recompression**
  (`COLI_CACHE_COMPRESS_IDLE` — `WarmCache::recompress_one_cold` densifies the coldest raw slot
  per call; the batch engine sweeps while no requests are pending, interrupting the sweep the
  moment one arrives).
- **Fault-tolerant I/O** — on a batched-read failure the buffered path re-issues each region via
  `Reactor::read_exact_retry` (linear backoff, transient `EIO`/`EAGAIN`/`EINTR`). Gated on
  `COLI_IO_RECOVERY`.
- **Huge pages** — `advise_hugepages` (`MADV_HUGEPAGE`) applied at every ≥ 2 MB allocation choke
  point: `AlignedBuf::with_capacity`, `Reactor::register_read_buffers`, safetensors read landing
  buffers. Range is narrowed to whole pages inside the caller's allocation so the destructive
  `MADV_DONTNEED` companion never touches neighboring mappings.
- **Topology probe + NUMA pinning** — `peregrine_io::topo` reads
  `/sys/devices/system/node/nodeN/cpulist` for the NUMA layout and
  `/sys/bus/pci/devices/<bdf>/current_link_{speed,width}` for PCIe links; single-node fallback on
  non-Linux keeps every caller safe against the "always at least one node" invariant. Under
  `COLI_NUMA_PIN=1`, worker threads pin round-robin across node-grouped CPUs: the `peregrine-par`
  pool via a std-only worker-startup hook (`set_worker_start_hook(fn(usize))`, installed by
  `Model::load` before the pool's lazy build), and the prefetch pool at its spawn site. CPUs are
  enumerated node-grouped, so consecutive workers fill a node before spilling to the next socket.
- **Offline layout tool** — `peregrine-tools::peregrine-layout-reorg` consumes the `dump-routes`
  JSON, builds per-layer co-occurrence graphs, and emits `<dir>/schedule.json` via `--method greedy`
  (greedy nearest-neighbor), `--method louvain` (single-phase modularity maximization +
  intra-community greedy walk), or `--method spectral` (Fiedler vector via deflated power iteration
  on the graph Laplacian, sorted by embedding value — the classical min-cut 1-D ordering).
  `Model::load` picks it up and, at `moe_forward_concurrent` entry, sorts each layer's streamed
  `EPlan`s by the schedule's rank so the batched io_uring submit issues contiguous-offset reads
  first. Bit-identical to natural-id order (the reduce uses `pos`, not submission order).
- **WMMA autotuner** — `WmmaTuner` records per-shape `(D, I, count, max_rows) → TileConfig` EWMAs
  and persists across sessions in `kernel_tuning.json`. The CUDA-side dispatch selector that would
  consume this is a CUDA-only follow-up.
- **PlanOptimizer** — folds `LaneTimings`, `BubbleTuner`, and `IoTuner` snapshots into a
  `RuntimeTelemetry` value per forward; the /metrics endpoint (planned) scrapes it. Ticking the
  IoTuner from here keeps the io_uring cap adjustment on a stable cadence.
- **Hardware perf counters** — `peregrine_io::PerfCounter` is a real `perf_event_open(2)` LLC-miss
  counter (thread-following, user-space-only to lower the paranoid bar), with the 64-byte
  `PERF_ATTR_SIZE_VER0` attr layout hand-declared because the pinned libc doesn't export it for gnu
  targets. `telemetry::open_l3_miss_counter` gates it on `COLI_PERF_COUNTERS=1`; every constructor
  degrades to `None` when the kernel refuses (CI containers, paranoid ≥ 3, no PMU), so the counter
  is an optimization input, never a dependency.

**Composition.** Each forward: (1) `moe_forward_concurrent` brackets its lanes, consults
`LaneBalancer` for per-expert placement using a live heat snapshot, and applies the co-activation
affinity order (fusion pairs adjacent, hyperedge components grouped); (2) after the forward,
`publish_lane_timings` swap-resets the accumulator, updates the tuners, steps the sensor governors
(thermal / RAPL power / bandwidth — all writing one governor-adjustable worker count with
shrink-wins arbitration), folds the routing-entropy EWMA, rewards + re-chooses the learned
scheduler (bandit or Q), feeds the co-activation tracker, and applies any changed io_uring cap;
(3) on the next forward `build_balancer` reads the fresh bias and `forward_hidden` applies the
staged learned/entropy prefetch-distance nudges. Feedback loop closed. All new state lives on
`Model` and is safe to expose to a `&self` scrape (atomics + `parking_lot::Mutex`).

**The completion sweep** (same day) finished every non-hardware roadmap item on top of this loop:
NUMA-bound landing buffers (`bind_local_if_enabled`, first-touch-correct `mbind`), hierarchical
two-level pool dispatch (`plan_assignments` over a worker→node map), per-expert adaptive mixed
precision (`plan_precision`, applied in the cuda tier's `reheat`), SQ-full-delta-driven io-wq
tuning, macro-state routing compression (`MacroTable` + `PredictSource::WithMacro`), the
`galactic` one-shot preprocessing pass (all artifacts from one corpus run), Hilbert / spectral /
2-opt layout methods + hypergraph tier placement (`tiers.json`, RAM tier prefetch-warmed at load),
the physical checkpoint self-rewrite (`apply_layout`, teacher-forcing-equality-gated), online
bandit and tabular-Q schedulers over the knob envelope (policies persisted in `route_stats.json`),
per-shape dispatch specialization (`shape_dispatch` probe-then-memoize), kblock tensor-layout
auto-conversion (header-tagged, loader-normalized), and the `compile-plan` profile-guided
execution plan (`plan.json`, consumed atomically at load).

**The tokenizer fast path** (gigatoken integration): `peregrine-serve` selects its tokenizer at
boot via `tok::TokenBackend` — the vendored gigatoken BPE engine when the model's
`tokenizer.json` is a supported BPE flavor (logged, overridable with `COLI_TOKENIZER=giga|hf`),
else the HF `tokenizers` crate. The gigatoken instance is process-persistent behind a mutex, so
its pretoken memo cache warms across requests — repeated chat-template prefixes encode from
cache. Correctness bar: the parity suite asserts id-for-id equality with the HF oracle over an
edge-case corpus (unicode, CJK/RTL, chat markup, contractions, empty inputs) plus decode round
trips; `--bench-tokenizer` measured 204 MB/s vs 6 MB/s (34×) on this box. The vendored subset
links no libpython and builds on stable (verified: `ldd` clean, `cargo +stable`).

---

## Crate & toolchain choices (Linux + CUDA + io_uring)

- **io_uring:** the `io-uring` crate (tokio-rs) with a **custom single-owner reactor thread** —
  *not* `tokio-uring` (its per-op Future model fights the batched-submit / `IOSQE_ASYNC` /
  io-wq-worker-cap ownership model the C ring uses). Keep O_DIRECT twin-fd + 4 KB base/len
  alignment and the 16 KB-aligned slab arena.
- **CUDA:** raw `extern "C"` FFI to the existing, validated `backend_cuda.o` first (flat ~40-fn
  ABI over opaque `ColiCudaTensor*`, `c/backend_cuda.h`). `build.rs` runs `nvcc` on
  `../c/backend_cuda.cu`, links `-lcudart -lstdc++` (mirrors `c/Makefile:191-193`). Add a few
  `_async` non-syncing stream variants for the scheduler. Defer `cudarc` (re-wrapping WMMA
  kernels = re-validation tax).
- **CPU parallelism:** custom physical-core-pinned pool, **not** rayon's logical-core global
  pool (README warns quantized kernels regress when SMT siblings contend for memory channels).
- **SIMD:** `std::arch` intrinsics with `is_x86_feature_detected!` runtime dispatch, **not**
  `portable_simd` — token-exactness needs exact AVX2 `maddubs`+`madd` / VNNI `dpbusd` / i8mm
  `smmla` accumulation order. This is the most correctness-sensitive module.
- **safetensors:** hand-rolled pread-based index (mirror `c/st.h` with `fadvise(DONTNEED)` +
  O_DIRECT to keep RSS flat), header via `serde_json` — **not** the `safetensors` crate (mmaps,
  no DONTNEED/O_DIRECT control). `memmap2` behind a `COLI_MMAP` flag. `half` for bf16/f16→f32.
- **tokenizer:** originally the `tokenizers` crate to bootstrap; the serve layer now runs a
  **vendored gigatoken BPE subset** (`peregrine-token`, from marcelroed/gigatoken v0.10.0, MIT)
  as the default fast path — SIMD (`std::arch`) pretokenizers with runtime dispatch, a memoizing
  BPE engine, and the HF `tokenizer.json` loader, all stable-toolchain (upstream is nightly-only
  via `portable_simd`, which lives solely in its SentencePiece engine — dropped here). The HF
  `tokenizers` crate stays as the automatic fallback for SentencePiece/non-BPE models and as the
  id-for-id parity oracle (`crates/peregrine-serve/tests/tokenizer_parity.rs`).
- **Support:** `crossbeam`, `core_affinity`, `bytemuck`, `parking_lot`, `serde_json`, `clap`.

---

## Repo layout & build

Standalone cargo workspace (the C runtime stays intact upstream in colibrì's `c/`; its CUDA
kernels are vendored here under `cuda/`):

```
crates/
  peregrine-core/     # QT formats (fmt 0..4), Cfg, safetensors index (zstd-aware),
                      #   compress (zstd codec)                       (↔ c/st.h)
  peregrine-kernels/  # std::arch int4/int8/int2 + f32 matmul, token-exact  (↔ matmul_qt_ex, glm.c:978)
  peregrine-io/       # io-uring reactor (I/O lane, O_DIRECT, fadvise batched, retry),
                      #   slab arena (generation-tagged), LRU/pin, warm cache
                      #   (Bloom + optional zstd), mem hints (hugepages, NUMA), topology probe
                      #                                                (↔ c/uring.h, tier.h)
  peregrine-cuda/     # -sys FFI to backend_cuda.h + build.rs(nvcc) + wrapper
  peregrine-model/    # MLA, router, MoE, DSA, MTP, prefetch prediction (predict.rs w/ PhaseAware),
                      #   lane telemetry + bubble tuner + lane balancer (lane.rs),
                      #   IoTuner (iotune.rs), PhaseTracker + workload classifier (workload.rs),
                      #   WmmaTuner (wmma_tune.rs), PlanOptimizer + telemetry (telemetry.rs),
                      #   + the N-ring 3-lane scheduler (concurrent.rs) (↔ glm.c forward)
  peregrine-sched/    # CPU∥SSD streaming core (moe_streamed) + reconstruct
  peregrine-par/      # persistent scoped worker pool, bit-identical to serial (std-only)
  peregrine-engine/   # binary `peregrine`: CLI (demo/build/bench/build-automaton/dump-routes)
                      #   + stdio serve protocol (drop-in for c/glm)
  peregrine-serve/    # binary `peregrine-serve`: OpenAI HTTP (axum/tokio) + continuous batching
                      #   (two-tier priority queue, adaptive batch cap, adaptive prefill window)
  peregrine-tools/    # lib + binary `peregrine-layout-reorg`: offline expert re-layout
                      #   (greedy / Louvain / spectral / Hilbert, --optimize 2-opt), tier
                      #   placement (tiers.json), physical checkpoint rewrite (--apply)
  peregrine-token/    # vendored gigatoken v0.10.0 BPE subset (MIT): SIMD pretokenizers,
                      #   memoizing BPE engine, HF tokenizer.json loader; GigaTokenizer facade
```

- Build: `cargo build --release --features cuda`; CPU-only drops the `cuda` feature (pure-CPU,
  like `make` without `CUDA=1`).
- Integration: `c/coli` gains a `--engine rust` / `COLI_ENGINE=rust` branch; the Rust binary
  speaks the existing `openai_server.py` stdio protocol (`Popen([exe, cap])`, `READY`/`END`/
  `CANCEL` sentinels, `c/openai_server.py:457-476`) so `serve`/`web`/desktop need zero changes.
- Feature flags mirror C env knobs (`URING`, `DIRECT`, `PIPE_WORKERS`, `PIN_GB`, `CUDA_EXPERT_GB`,
  `RAM_GB`, `TOPP`, `DRAFT`, `DSA`), preserving C precedence (explicit flag > env > auto).

---

## Milestones (each independently verifiable; start on the 2.4 MB tiny-random model)

| M | Goal | Verify |
|---|---|---|
| **M0** | Workspace; parse `config.json`/tokenizer/safetensors header; load tiny-random model | tensor inventory matches `st.h`; tokenizer round-trips id-for-id vs `tok.h` |
| **M1** | **CPU-only int4 forward, token-exact**: MLA (q/kv-LoRA, partial RoPE, absorption), sigmoid router, dense+shared+routed MoE, SIMD int4/int8/int2 kernels | `TF=1` 32/32 + greedy 20/20 vs oracle `ref_glm.json` (the README:57 bar) |
| **M2** | io_uring streaming + LRU/pin tiers on the **real 744B model**, CPU-only | coherent decode; hit-rate/disk-wait counters same order as `./glm`; warm-cache A/B within "valid forward" tolerance |
| **M3** | CUDA expert lane via FFI (still phased): link `backend_cuda.o`, upload VRAM tier, route VRAM experts through `coli_cuda_expert_group` | `tools/benchmark_cuda_fixture.py` on 313M fixture; CPU vs CUDA same tokens |
| **M4** | **The concurrent 3-lane scheduler** (centerpiece): completion-driven dispatch, CPU∥GPU∥SSD on the same layer, sharded/indexed accumulation, PILOT via ring | **tok/s beats C engine** on a matched box; argmax stream unchanged vs M3; profiler shows GPU busy during CPU/disk |
| **M5** | MLA weight-absorption + DSA lightning indexer (top-2048, auto-detected from `out-idx-*`) | DSA-off reproduces dense attention token-for-token (README:67); absorption TF 32/32 |
| **M6** | MTP speculative decode (int8 head draft + batch-union verify) | 39–59% acceptance / 2.2–2.8 tok/fw on int8-head model (README:60); rejection-sampling correct under sampling |
| **M7** | serve / OpenAI drop-in: stdio `READY`/`END`/`CANCEL` child | `openai_server.py` spawns Rust binary unchanged; `curl` chat streams; web UI works |

De-risking order: CPU-only tiny model first (M0/M1) → reuse `.cu` via FFI before the scheduler
(M3 before M4) → keep the phased path as a correctness oracle for M4.

---

## Reuse decisions (keep via FFI / process boundary)

- **`c/backend_cuda.cu` kernels via FFI** — validated WMMA/quant/attention over the flat C ABI;
  compiled by `peregrine-cuda/build.rs`. Rewriting them early buys nothing and costs re-validation.
- **`openai_server.py` gateway + `coli` CLI** — the Rust binary is a drop-in for `c/glm`; gateway,
  web UI, desktop unchanged.
- **Oracle/eval tooling** (`tools/make_glm_oracle.py`, `eval_glm.py`, `ref_glm.json`, `ref.json`)
  — the correctness gate at every milestone, reused verbatim.
- **`resource_plan.py`** (header-only planner) and the **FP8→int4 converter** + int4/int8 container
  format (incl. int8 MTP heads) — unchanged; the Rust engine consumes the same files.

---

## Risks & mitigations

- **Token-exact hand-written SIMD (highest risk):** `std::arch` (controlled accumulation order);
  port `qrow_i8` / `matmul_*_idot` / `matmul_i4_pair` exactly; validate each kernel bit-identical
  to a NOPACK-style f32 reference before integration; chase byte-parity only on the S=1 CPU path.
- **Re-validation enormity:** reuse the exact oracle harness every milestone; tiny model keeps the
  loop at seconds; gate on TF 32/32 + greedy 20/20.
- **io_uring under O_DIRECT:** keep 4 KB base/len alignment + 16 KB slab arena; buffered twin-fd
  fallback when unaligned; property-test the alignment arithmetic.
- **Memory/OOM:** port `cap_for_ram` auto-sizing from `MemAvailable`; bound the slab arena; reserve
  ~2 GB/device VRAM headroom before placing the expert tier.
- **Scheduler correctness:** the one-`ExpertTask`-per-unique-expert queue enforces the batch-union
  invariant; generation-tagged slabs prevent stale speculative writes; keep the phased path
  available as a differential oracle for M4.

---

## Critical files (C references to port/bind)

- `c/glm.c` — `moe()` phased loop **2658–3191** (the exact serialization to replace: CPU inline
  3084–3101, GPU group after 3124–3147, the measured-waste comment 2921–2923); `matmul_qt_ex`
  978; `expert_load` 1641; `layer_forward_rows` 3629; `spec_decode` 4146; `pin_load` 5432.
- `c/uring.h` — io_uring ownership / `IOSQE_ASYNC` / io-wq model for the Rust I/O lane.
- `c/backend_cuda.h` / `c/backend_cuda.cu` — flat C ABI to FFI; add `_async` stream variants.
- `c/st.h` — safetensors index + fadvise/O_DIRECT streaming behavior.
- `c/tier.h` — LFRU eviction/promotion math for the tier manager.
- `c/openai_server.py` — `READY`/`END`/`CANCEL` stdio protocol the Rust binary must implement.

## Verification (end-to-end)

1. Per-milestone oracle gate: `SNAP=./glm_tiny TF=1 <rust-engine> ...` reproduces `ref_glm.json`
   (TF 32/32) and greedy 20/20 — the same bar the C engine passes (`README.md:57`).
2. M4 throughput proof: run the documented optimized stack config on the same NVIDIA box for both
   engines; the Rust `peregrine-engine` must exceed the C `tok/s`, with a profiler trace showing GPU,
   CPU, and io_uring lanes busy simultaneously within a layer (the phased C trace shows them
   sequential).
3. Drop-in proof: point `c/openai_server.py` at the Rust binary and run a `curl` chat completion +
   the web dashboard — unchanged behavior confirms the stdio protocol match.
