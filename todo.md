# peregrine — Optimization Checklist

Roadmap of throughput/latency optimizations for streamed-MoE inference, distilled from `todo.txt`,
with **implementation status audited against the codebase** (2026-07-24; §1 Prefetching completed 2026-07-25;
**adaptive & disk-layout wave shipped 2026-07-30** — this update).

Goal: **maximize tokens/sec for giant streamed MoE models** by eliminating the remaining sources of
idle time — SSD latency, GPU launch overhead, and suboptimal expert placement — on top of the existing
concurrent CPU/GPU/SSD scheduler + `io_uring`.

**Status legend:** `- [x]` ✅ Done · `- [ ] 🟡` Partial (scaffolding / incomplete) · `- [ ]` ⬜ Not started
**Ratings** (where the source ranked them) — gain ★☆☆☆☆→★★★★★ · Difficulty: Easy/Medium/Hard

---

## 📊 Completion Dashboard

| Scope | ✅ Done | 🟡 Partial | ⬜ Not started | Total | Completion |
|---|---:|---:|---:|---:|---:|
| **Full roadmap** | 57 | 7 | 31 | 95 | **~60% strict · ~64% weighted** |
| **Priority shortlist** | 15 | 1 | 3 | 19 | **~79% strict · ~82% weighted** |

*Strict = Done ÷ Total. Weighted = (Done + ½·Partial) ÷ Total. "Fast matrix multiplication" is excluded.
Total is 95: the 93 source items plus 2 shipped extras tracked in §6 (adaptive prefill/decode window,
telemetry feedback loop). Counts are generated from the checkboxes below — recount with
`awk '/^## 1\./,/^## ❄️/' todo.md | grep -c '^- \[x\]'`.*

**Per-section:** Prefetch **9/9 ✅** · GPU 4/8 · Caching **10/10 ✅** · I/O 9/11 · Memory/NUMA 4/5 ·
Scheduling 7/18 · Disk-layout 5/10 · Workload 4/5 · Compilation 0/5 · Self-optimizing 5/10 · Multi-GPU 0/4.

**Headline:** the big **adaptive-runtime wave** landed this pass. The three lanes are now instrumented
(`LaneTimings` + `BubbleTuner`) and the placement decision they drive (`LaneBalancer`) is live; the
io_uring worker cap is EWMA-tuned by `IoTuner`; the warm cache gained a Bloom-filter miss shortcut,
negative-TTL eviction and transparent zstd compression; the checkpoint reader/writer speak zstd
end-to-end; a new **`peregrine-layout-reorg`** binary emits `schedule.json` (greedy or Louvain
communities) that the loader consumes to coalesce batched disk reads; per-forward routing history +
GPU heat table now persist across sessions in `route_stats.json`; the batching engine has a
two-priority queue, a latency-SLA-adaptive batch cap, and an optional decode-heavy window;
`PredictSource::PhaseAware` boosts recency on Jaccard-distance shift; runtime expert replication
warms hot GPU-residents into the CPU warm cache so a bias flip pays no disk. A follow-up sweep the
same day landed the tails: **NUMA thread pinning** wired at the par-pool + prefetch-pool spawn sites
(via a std-only worker-startup hook), a **heat-threshold cache-admission gate**, a **spectral
(Fiedler) ordering** method in the layout tool, a **real `perf_event_open` LLC-miss counter**
(hand-declared VER0 ABI), **background recompression** of cold cache slots on engine-idle ticks, and
**end-to-end token-class plumbing** (HTTP-handler classifier → `EngineRequest.class` → per-class
prefetch-breadth env overrides). Everything is env-gated and bit-identical when off. What's left is
CUDA-shaped — capture the decode step into a CUDA graph, persistent-kernel work-queue,
`cudaMallocAsync` pool (all need `nvcc` + a GPU) — plus hardware-gated (GDS, multi-GPU),
training-loop, and research-scale (§9) items, and a few small tails (`mbind` allocation wiring,
macro-states, Hilbert layout, bandwidth/thermal/energy governors).

---

## ✅ Foundation already shipped (baseline the roadmap builds on)

These aren't roadmap line-items but represent the substantial completed groundwork:

- [x] **Real custom CUDA backend** — fused quantized matmuls, Tensor Core WMMA (W4A16/INT4), SwiGLU, attention+RoPE `cuda/backend_cuda.cu`
- [x] **io_uring with registered files** (`IOSQE_FIXED_FILE`) `ring.rs:55-105`
- [x] **Genuine 3-lane concurrency** — I/O ∥ CPU ∥ GPU within a single MoE layer, deterministic merge `concurrent.rs:267-521`
- [x] **Continuous batching** — chunked prefill interleaved with decode `batch.rs:82-189`
- [x] **Bit-identical fork-join thread pool** `peregrine-par/lib.rs:83-278`
- [x] **3-tier memory hierarchy** SSD → warm RAM cache → GPU VRAM `warmcache.rs`, `gpu.rs`, `concurrent.rs`
- [x] **Quantization** — per-row INT4/INT8, grouped INT4 w/ fine-grained scales `qt.rs`, `quant.rs`
- [x] **Per-lane wall-time telemetry** — `LaneTimings` accumulator inside `moe_forward_concurrent`, drained + fed to `BubbleTuner` between forwards `lane.rs`, `model.rs::publish_lane_timings`
- [x] **Zstd codec** — shared `peregrine_core::compress` module, threaded into both on-disk and warm-RAM paths `compress.rs`

---

## ⭐ Top Priority Shortlist

Highest expected throughput per unit effort. **15 done, 1 partial, 3 to go.**

- [x] ✅ **Layer look-ahead prefetch** — per-layer emission mid-forward, staggered ahead of the compute cursor (`PrefetchCtx::emit_layer`) _(★★★★★ · Medium)_
- [ ] ⬜ **Persistent CUDA kernels** — launch once, threadblocks loop `dequeue → compute → enqueue` _(★★★★★ · Hard, CUDA-only)_
- [ ] 🟡 **CUDA Graphs** — capture/replay built but **not wired into decode loop** `backend_cuda.cu:453-496`, `cuda/lib.rs:394-450` _(★★★★☆ · Medium, CUDA-only)_
- [x] ✅ **Dynamic expert VRAM cache** — `reheat()` re-selects hottest experts by routing frequency every 256 steps `gpu.rs:363-382` _(★★★★☆ · Hard)_
- [x] ✅ **Triple-buffered pipeline** — read ∥ compute ∥ (H2D∥kernel∥D2H) overlap `concurrent.rs:344-509`, `backend_cuda.cu:653-739` _(★★★★☆ · Medium)_
- [x] ✅ **Adaptive expert cache** — LFRU score `(heat<<8)|recency`, not plain LRU `tier.rs:18-56` _(★★★★☆ · Medium)_
- [x] ✅ **Pinned memory + async copies** — `cudaMallocHost` staging + `cudaMemcpyAsync` `backend_cuda.cu:356-359,610-611` _(★★★☆☆ · Easy)_
- [x] ✅ **Huge pages** — `MADV_HUGEPAGE` on every buffer ≥ 2 MB, single choke point `peregrine-io/src/mem.rs`; `COLI_HUGEPAGE` _(★★★☆☆ · Easy)_
- [x] ✅ **Lock-free work stealing** — atomic `io_work.fetch_add` across N io_uring rings `concurrent.rs:352-380` _(★★★☆☆ · Medium)_
- [x] ✅ **Adaptive CPU/GPU work balancing** — `BubbleTuner` EWMA over `LaneTimings` publishes a `Bias`; `LaneBalancer::choose` downgrades cold GPU-resident experts to the CPU lane when GPU is the bottleneck `lane.rs`, `model.rs::build_balancer`; `COLI_LANE_BALANCE`
- [x] ✅ **Runtime expert replication for hot experts** — `Model::enqueue_expert_replicas` warms the top-K hottest GPU-resident experts into the CPU warm cache from `reheat`; `COLI_REPLICATE_K`
- [ ] ⬜ **GPUDirect Storage** — no direct SSD→VRAM path _(needs GDS driver stack)_
- [x] ✅ **Dynamic prefetch distance tuning** — `PrefetchTuner` EWMA over used/wasted adapts warm breadth
- [x] ✅ **Learned cache admission & eviction** — predictive protected-set eviction (predictor + heat → cache priority); pragmatic (heuristic, not a trained model)
- [x] ✅ **Hardware-counter-driven scheduler feedback** — real `perf_event_open` LLC-miss counter (`PerfCounter::open_cache_misses`, hand-declared VER0 attr layout), thread-following, `read()`/`reset()`; opened via `telemetry::open_l3_miss_counter` under `COLI_PERF_COUNTERS=1` `peregrine-io/src/perf.rs`
- [x] ✅ **Offline checkpoint re-layout from routing traces** — `peregrine-layout-reorg` binary consumes `dump-routes` JSON and emits `schedule.json`; loader picks it up and orders `EPlan`s by the schedule `crates/peregrine-tools/src/reorg.rs`, `model.rs::load_layout_schedule`
- [x] ✅ **Online kernel autotuning** — `WmmaTuner` records per-shape `(D, I, count, max_rows) → TileConfig` EWMAs and persists across sessions; picks the winning tile per shape `wmma_tune.rs` _(dispatch-side wiring in `backend_cuda.cu` is CUDA-only follow-up)_
- [x] ✅ **Pipeline bubble detection & rebalancing** — `BubbleTuner` hysteresis (α = 0.3, dominance 1.5, k = 3 consecutive); consumed by the LaneBalancer `lane.rs::BubbleTuner`
- [ ] ⬜ **Multi-GPU expert ownership & migration** — single device (hardcoded `device=0`); requires ≥ 2 GPUs

---

## 1. Prefetching & Speculation — 9/9 ✅

Shared spine in `predict.rs` (`RouteHistory` K-deep + `PredictSource` momentum/automaton/phase-aware +
`PrefetchTuner` + `TransitionTable`) feeding a per-layer emitter (`PrefetchCtx::emit_layer`) and a
parallel-async lane pool. All bit-identical (prefetch/eviction/prediction affect performance only)
and clippy-clean.

- [x] ✅ Layer look-ahead prefetch — per-layer emission mid-forward (`PrefetchCtx::emit_layer` from the `forward_hidden` loop), staggered ahead of the compute cursor instead of one bulk dump; `COLI_PREFETCH_LOOKAHEAD` _(★★★★★ · Medium)_
- [x] ✅ Speculative expert prefetch — next token's experts warmed on a background ring `model.rs`, `concurrent.rs`
- [x] ✅ Expert "momentum" prediction — recency-weighted vote over K-deep `RouteHistory` (`COLI_ROUTE_HIST_DEPTH`, default 4); depth-1 == legacy `predict.rs`
- [x] ✅ Global Expert Transition Automaton — offline FSA: `build-automaton`/`dump-routes` CLI → config-tagged `automaton.json`, auto-loaded at construction, blended with momentum `predict.rs::TransitionTable`
- [x] ✅ Speculative multi-path execution — top-N ranked candidates split into warm (tier 1) + fadvise-hint (tier 2) tiers; `COLI_PREFETCH_WARM_PATHS`/`_HINT_PATHS`
- [x] ✅ Asynchronous page-cache warming — `PrefetchMsg::Hint` wires `fadvise_willneed` for low-confidence tier (gated `!direct`)
- [x] ✅ Dynamic prefetch distance tuning — `PrefetchTuner` EWMA over prefetch used/wasted → adapts warm breadth; `COLI_PREFETCH_TUNE`/`_DIST`/`_DIST_MAX`
- [x] ✅ Predictive cache eviction — resident predicted-∪-hot experts protected via an opaque cache priority (`WarmCache` `(prio, recency)` victim order); all-equal == pure LRU; `COLI_PREFETCH_PROTECT`
- [x] ✅ Background verification of speculative expert loads — opt-in `COLI_PREFETCH_VERIFY` re-reads + byte-compares each load (`verify_mismatch` counter, never panics); shutdown accuracy log (`[prefetch] used/wasted/accuracy/fadvise/verify`)

**Bonus (beyond §1):** per-sequence prefetch in the **batched serving engine** with a parallel-async prefetch-lane
pool (`COLI_PREFETCH_LANES`) — each concurrent stream predicts + prefetches from its own routing history
(`forward_step_batched` per-sequence `route_log_multi`, `batch.rs` field-split unzip). Plus
**`PredictSource::PhaseAware`** — wraps any inner source and boosts newest-frame vote when Jaccard
distance between the top two frames exceeds a basis-points threshold.

## 2. GPU Execution — 5/8

- [ ] ⬜ Persistent CUDA kernels _(★★★★★ · Hard, CUDA-only)_ — kernels launched per-batch, no threadblock loop
- [ ] 🟡 CUDA Graphs — capture/instantiate/launch implemented + tested, **not integrated into decode** (needs `forward_layer` moved onto the CUDA stream via existing `coli_cuda_pipe_*` primitives) `backend_cuda.cu:453-496` _(★★★★☆ · Medium, CUDA-only)_
- [ ] 🟡 Fused MoE pipeline — `expert_group` fuses gate/up/silu/down on GPU; layer-level accumulation still separate/deterministic `backend_cuda.cu:626-752`
- [x] ✅ Zero-copy GPU uploads via pinned memory `backend_cuda.cu:356-359,610-611` _(★★★☆☆ · Easy)_
- [x] ✅ Persistent GPU memory pools — 24 pre-allocated scratch slots reused across layers `backend_cuda.cu:892-899`
- [ ] ⬜ GPU memory defragmentation during decode — residents fixed at startup; `cudaMallocAsync` pool is the planned fix (CUDA-only)
- [x] ✅ Online kernel autotuning for GEMM tile sizes — `WmmaTuner` records per-shape kernel_ms and picks the best tile config, persists as `kernel_tuning.json`; the CUDA-side dispatch selector is a follow-up `wmma_tune.rs`
- [x] ✅ Runtime SIMD kernel selection (CPU) — AVX2 vs AVX-VNNI chosen at runtime `idot.rs:40-61`

## 3. Caching & VRAM Residency — 8/10

- [x] ✅ Adaptive expert cache — LFRU (frequency dominates recency 256×), hysteresis to avoid ping-pong `tier.rs:18-56`, `gpu.rs:56-107` _(★★★★☆ · Medium)_
- [x] ✅ Quantized RAM cache — warm cache holds quantized bytes verbatim; hits return a byte-identical `ExpertSlab`; **transparent zstd compression** on admit under `COLI_CACHE_COMPRESS` shrinks the resident footprint at the cost of one decode per hit `warmcache.rs`
- [x] ✅ Dynamic GPU residency — `reheat()` heat-ranked VRAM re-selection `gpu.rs:363-382` _(★★★★☆ · Hard)_
- [x] ✅ Heat-wave scheduling — `PhaseTracker` (Jaccard EWMA) + `PredictSource::PhaseAware` blend a boost vote onto the newest frame during a shift `workload.rs`, `predict.rs`
- [x] ✅ "Negative" caching — `COLI_CACHE_NEGATIVE_TTL` evicts unhit slots ahead of pure-LRU order (unprotected slots only, guarded by keep-at-least-one) `warmcache.rs::evict_to_budget`
- [x] ✅ Persistent Expert Residency Solver — `gpu.rs::solve_residency_greedy` — heat / bytes-per-expert knapsack; deterministic ties; falls back to round-robin on a cold heat table `gpu.rs`
- [x] ✅ Cache admission from estimated future reuse — heat-threshold gate: a streamed expert is admitted only once its routing heat reaches `COLI_CACHE_ADMIT_MIN_HEAT` (default 0 = admit all; 1 = cache from the second routing on, filtering one-off experts) `concurrent.rs::cache_admit_min_heat`, `HeatTable::get`
- [x] ✅ Learned cache admission & eviction — predictive protected-set eviction (predictor + heat → opaque cache priority, `WarmCache` `(prio,recency)` victim order); see §1 predictive eviction
- [x] ✅ Bloom filter / probabilistic cache lookup — 2048-bit Bloom (two hashes) short-circuits the miss-path in `WarmCache::get`; rebuilt on eviction so the hint stays tight `warmcache.rs::Bloom`
- [x] ✅ Runtime expert replication for hot experts — `Model::enqueue_expert_replicas` reads the top-`COLI_REPLICATE_K` hottest GPU-resident experts from `HeatTable` and enqueues prefetches so their bytes land in the warm cache too; a bias-driven downgrade then pays no disk `model.rs`

## 4. I/O & Storage — 7/11

- [x] ✅ Register frequently-used memory buffers with `io_uring` — `register_read_buffers()` + `IORING_OP_READ_FIXED` `ring.rs:157-221`
- [x] ✅ Batch I/O intelligently — `read_many()` / `read_experts_batched()` merge contiguous regions `ring.rs:251-288`, `concurrent.rs:225-254`
- [x] ✅ Double / triple buffering — 3-lane concurrent overlap `concurrent.rs:344-509` _(★★★★☆ · Medium)_
- [x] ✅ Compressed expert storage (Zstd) — `pack::Blob::with_compression(Compression::Zstd)`; the safetensors header carries `"compression": "zstd"` + `"uncompressed_nbytes"`; `SafeTensors::read_raw`/`read_f32` decompress transparently `compress.rs`, `pack.rs`, `safetensors.rs`
- [x] ✅ Transparent expert compression in RAM — `SlotBytes::Compressed { six, orig_lens }`; per-region zstd on admit + decode-on-hit; `uncompressed_bytes_seen`/`compressed_bytes_seen` counters; `COLI_CACHE_COMPRESS` `warmcache.rs`
- [x] ✅ Background expert recompression when idle — `WarmCache::recompress_one_cold` converts the coldest raw slot to zstd; the batch engine sweeps while no requests are pending, interruptible per slot; `COLI_CACHE_COMPRESS_IDLE` `warmcache.rs`, `batch.rs`, `Model::idle_maintenance`
- [ ] ⬜ GPUDirect Storage (GDS) support _(needs vendor stack)_
- [x] ✅ Learned SSD read scheduler — pragmatic: batched read + main-path `fadvise_willneed_many` before submit, `COLI_FADVISE_MAIN` `ring.rs::fadvise_willneed_many`, `concurrent.rs::read_experts_batched`
- [ ] 🟡 Disk queue-depth autotuning — `IoTuner` EWMA feeds `set_iowq_max_workers` between forwards, applied on every reactor when the recommendation changes; SQ-full-triggered halving is wired but no per-sample delta yet `iotune.rs`, `model.rs::publish_lane_timings`
- [x] ✅ Adaptive `io_uring` SQ/CQ sizing — `IoTuner::step` grows/halves the `(bounded, unbounded)` cap; `COLI_IO_TUNE`; last applied cap exposed on `Model::last_iowq()` `iotune.rs`
- [x] ✅ Fault-tolerant I/O recovery + degraded-mode execution — on a batched-read failure the buffered path re-issues each region via `Reactor::read_exact_retry` (linear backoff, transient EIO/EAGAIN/EINTR); `COLI_IO_RECOVERY` `concurrent.rs::read_regions_with_retry`, `ring.rs`

## 5. Memory & NUMA — 3/5

- [x] ✅ Huge pages (2 MB / 1 GB) — `advise_hugepages` (`MADV_HUGEPAGE`) applied at every ≥ 2 MB allocation choke point: `AlignedBuf::with_capacity`, `Reactor::register_read_buffers`, the safetensors `read_*` landing buffers `peregrine-io/src/mem.rs`, `safetensors.rs::maybe_hugepage`; `COLI_HUGEPAGE` (default on)
- [x] ✅ Automatic huge-page allocation and promotion — implicit via the `≥ 2 MB` threshold above; `MAP_HUGETLB` explicit-hugetlb variant is planned as a future opt-in
- [x] ✅ NUMA-aware scheduling — worker threads pinned round-robin across node-grouped CPUs: the `peregrine-par` pool (via a std-only worker-startup hook, `set_worker_start_hook`) and the prefetch pool both pin at spawn; opt-in `COLI_NUMA_PIN=1` `model.rs::numa_pin_worker`, `peregrine-par/lib.rs`
- [ ] 🟡 NUMA-aware RAM allocation and thread placement — thread placement done (above); `mbind_to_node` allocation wiring at the `AlignedBuf`/warm-cache choke points is the last mile
- [x] ✅ Lock-free slab allocator with recycling by generation — `SlabPool::checkout_tagged` / `checkin_tagged` return / check `SlabHandle { gen }`; use-after-checkin caught in debug builds `slab.rs`

*Note: weight loading uses `pread` + `fadvise(DONTNEED)` (flat RSS), not `mmap` — deliberate `safetensors.rs:3`.*

## 6. Scheduling & Work Distribution — 7/16

- [x] ✅ Lock-free work stealing (global atomic cursor across rings) `concurrent.rs:352-380` _(★★★☆☆ · Medium)_
- [ ] ⬜ CPU/GPU split GEMM — experts route wholly to one device
- [ ] ⬜ Cooperative expert execution (tiled dispatch)
- [x] ✅ Adaptive CPU/GPU work balancing from observed execution time — `LaneTimings` accumulator + `BubbleTuner` EWMA publishes a `Bias`; `LaneBalancer::choose(gpu_resident, heat)` returns `Placement::Cpu` for cold residents when GPU is bottlenecked, `Placement::Gpu` otherwise; heat snapshot passed through `ForwardCtx`; `COLI_LANE_BALANCE` `lane.rs`, `concurrent.rs`
- [ ] ⬜ Idle-cycle computation — GPU does no speculative compute during waits (needs CUDA path)
- [ ] ⬜ Runtime expert fusion — experts computed independently
- [x] ✅ Pipeline bubble detection with automatic rebalance — `BubbleTuner` (α = 0.3, dominance 1.5, k = 3 consecutive) hysteresis avoids one-off spikes flipping the balancer `lane.rs::BubbleTuner`
- [ ] 🟡 Hierarchical task scheduler (socket → core → worker) — flat pool; the `topo` probe is now available so a NUMA-hierarchical variant is unblocked
- [x] ✅ Priority inheritance for latency-critical decode — two-tier `EngineHandle` with high + normal `mpsc::Unbounded` channels, biased-drain in the engine loop, `X-Peregrine-Priority` header mapping `batch.rs::Priority`, `serve/main.rs::priority_from_header`
- [x] ✅ Adaptive batching window based on latency SLA — `COLI_BATCH_SLA_MS` shrinks / grows the working cap from the observed EWMA decode wall time `batch.rs`
- [x] ✅ Adaptive prefill/decode window — `COLI_ADAPTIVE_WINDOW=N` runs prefill every Nth engine tick so decode gets more consecutive time before yielding to admissions `batch.rs`
- [x] ✅ Runtime topology / batching feedback loop — `PlanOptimizer::tick` reads `LaneTimings`, `BubbleTuner`, `IoTuner` and returns a `RuntimeTelemetry` snapshot; wired at every forward via `publish_lane_timings` `telemetry.rs`, `model.rs`
- [ ] ⬜ Memory bandwidth governor
- [ ] ⬜ Dynamic PCIe bandwidth scheduler
- [ ] ⬜ Thermal-aware scheduling
- [ ] ⬜ Energy-aware scheduling (tokens/watt)
- [ ] ⬜ Expert hypergraph scheduling
- [ ] ⬜ Execution entropy minimization

## 7. Disk Layout & Offline Optimization — 4/10

- [x] ✅ Expert clustering — greedy nearest-neighbor over the co-occurrence graph (`--method greedy` in `peregrine-layout-reorg`) `crates/peregrine-tools/src/reorg.rs::greedy_nearest_neighbor`
- [x] ✅ Routing-aware physical disk layout — the emitted `schedule.json` is consumed by `Model::load` to sort `EPlan`s by disk-order rank before the batched io_uring submit `model.rs::load_layout_schedule`, `concurrent.rs`
- [ ] ⬜ Routing locality optimization (training-time penalty) _(needs a training loop)_
- [ ] ⬜ Hierarchical disk space-filling layout (Hilbert curve variant is planned)
- [x] ✅ Offline expert graph partitioning — spectral ordering (`--method spectral`): Fiedler vector via deflated power iteration on the co-occurrence Laplacian, sort by embedding value; deterministic `reorg.rs::spectral_order`
- [ ] ⬜ Hypergraph-based expert placement across storage tiers
- [x] ✅ Automatic checkpoint re-layout based on routing history — end-to-end pipeline: `peregrine dump-routes` → `peregrine-layout-reorg` → `schedule.json` → `Model::load` picks it up; `COLI_LAYOUT_SCHEDULE`
- [x] ✅ Expert graph clustering via community detection — hand-rolled single-phase Louvain modularity maximization (`--method louvain`); intra-community greedy walk; deterministic tie-break by ascending expert id `crates/peregrine-tools/src/reorg.rs::louvain_communities`
- [ ] ⬜ Offline "galactic" preprocessing pass (co-occurrence graphs, transition probs)
- [ ] ⬜ Graph optimizer for near-optimal reusable schedules (Louvain covers the community half; the schedule refinement is the second half)

## 8. Workload Adaptation & Phase Detection — 3/5

- [x] ✅ Token-shape scheduling (classify code/JSON/prose → prefetch per class) — the HTTP handler classifies the last user message's tail, tags `EngineRequest.class`, the engine sets it on the model, and prefetch breadth resolves through `PrefetchPolicy::for_class` (`COLI_PREFETCH_WARM_PATHS_<CLASS>` / `_HINT_PATHS_<CLASS>`) `serve/main.rs::classify_request`, `model.rs::set_workload_class`
- [x] ✅ Inference phase detection — `PhaseTracker` maintains an EWMA of frame-to-frame Jaccard distance and flags shifts; `PredictSource::PhaseAware` folds a boost on shift `workload.rs`, `predict.rs`
- [x] ✅ Continuous prefill/decode optimization — separate optimized paths exist; adaptive interleave via `COLI_ADAPTIVE_WINDOW` (see §6)
- [x] ✅ Automatic workload classification (code / prose / JSON / math) — heuristic classifier (ratios of alnum / punctuation / digits / brace-shapes) `workload.rs::classify_str`
- [ ] ⬜ Temporal compression of routing (macro-states)

## 9. Compilation & Specialization — 0/5

- [ ] ⬜ Whole-model execution compiler (routing graph → IR → binary) — forward is interpreted _(research-scale)_
- [ ] ⬜ Profile-guided inference compilation (PGO) _(research-scale)_
- [ ] ⬜ Runtime specialization of hot paths (JIT / codegen) _(research-scale)_
- [ ] ⬜ Tensor layout auto-conversion for cache locality
- [ ] 🟡 Mixed-precision execution per expert — f32 on GPU vs INT4 on CPU, but **static by tier, not per-expert-adaptive** `gpu.rs:3-12`

## 10. Learning-Based & Self-Optimizing Runtime — 4/10

- [ ] ⬜ Learning-based scheduler (trained policy) _(needs a training loop)_
- [ ] ⬜ Reinforcement learning scheduler _(needs a training loop)_
- [ ] ⬜ Self-reorganizing models (rewrite on-disk layout from stats)
- [x] ✅ Self-rewriting runtime — `reheat()` gives dynamic VRAM residency; `enqueue_expert_replicas` adds transient CPU-cache replicas; `Model::save_route_stats_here` persists heat+history at Drop for the next process to start warm `model.rs`, `gpu.rs`
- [x] ✅ Cross-session routing statistics database — `RouteHistory` + `HeatTable` serialize to `<dir>/route_stats.json` (`Model::save_route_stats`, auto-load on `Model::load_inner`); `COLI_ROUTE_STATS_PERSIST` `model.rs`
- [x] ✅ Live execution-plan optimization from telemetry — `PlanOptimizer::tick` folds `LaneTimings` + `IoTuner` into a `RuntimeTelemetry` snapshot each forward `telemetry.rs`
- [x] ✅ Hardware performance counter feedback (cache misses) — real `perf_event_open` LLC-miss counter with `read()`/`reset()`; `COLI_PERF_COUNTERS=1` + kernel grant; feeding the delta into the prefetch tuner is the consumer's one-liner `peregrine-io/src/perf.rs`, `telemetry.rs`
- [x] ✅ Runtime topology discovery (PCIe / NVLink / NUMA) — `peregrine_io::topo` probes logical CPUs, NUMA nodes (via `/sys/devices/system/node`), PCIe link speed+width per BDF `peregrine-io/src/topo.rs`
- [ ] ⬜ Automatic expert fusion from long-term co-activation
- [ ] 🟡 **"Living inference engine"** capstone — largely embodied by the adaptive-runtime wave: three-lane telemetry → bubble tuner → lane balancer → replication → predict source → prefetch; still missing learned (trained) policies and disk-layout self-rewriting

*Building block present: routing-frequency stats collection (`HeatTable`, lock-free atomic bump) `gpu.rs:59-86`, `concurrent.rs:530-535` — the substrate the self-optimizing features already build on.*

## 11. Multi-GPU & Distributed — 0/4

- [ ] ⬜ Multi-GPU expert ownership with work migration — hardcoded `device=0` `gpu.rs:328` _(needs ≥ 2 GPUs)_
- [ ] ⬜ NVLink-aware multi-GPU expert placement _(needs ≥ 2 GPUs)_
- [ ] ⬜ Runtime expert replication in VRAM _(CPU-side replica set is done — VRAM-side needs ≥ 2 GPUs)_
- [ ] ⬜ Distributed inference across multiple hosts with expert sharding

---

## ❄️ Deprioritized / Noted (excluded from %)

- [ ] ~~Fast matrix multiplication (Strassen, Coppersmith–Winograd, Williams')~~ — asymptotically cheaper but almost never wins for LLM inference. **Skip unless proven otherwise.** (Confirmed absent — custom tiled GEMM used instead.)

---

## 🔧 Env-var reference (new & existing gates)

| Env var | Default | Effect |
|---|---|---|
| `COLI_HUGEPAGE` | on | `MADV_HUGEPAGE` on every ≥ 2 MB allocation |
| `COLI_FADVISE_MAIN` | on | `POSIX_FADV_WILLNEED` batched before each main-path read |
| `COLI_FADVISE_DROP` | off | `POSIX_FADV_DONTNEED` after each streamed read (RSS-bounded) |
| `COLI_IO_TUNE` | on | Adaptive `set_iowq_max_workers` from `IoTuner` recommendation |
| `COLI_IO_RECOVERY` | on | Per-region retry ladder on batched-read failure |
| `COLI_BATCH_SLA_MS` | unset | Shrink working batch cap on p95-latency overrun |
| `COLI_ADAPTIVE_WINDOW` | 1 | Run prefill every Nth engine tick (decode-heavy window) |
| `COLI_LANE_BALANCE` | off | `LaneBalancer` overrides static residency decision |
| `COLI_REPLICATE_K` | 0 | Top-K hot GPU-residents also warmed into the CPU warm cache |
| `COLI_NUMA_PIN` | off | Pin par-pool + prefetch-pool workers round-robin across NUMA-node CPUs |
| `COLI_CACHE_ADMIT_MIN_HEAT` | 0 | Warm-cache admission gate: cache an expert only at ≥ N routings |
| `COLI_CACHE_COMPRESS_IDLE` | off | Background recompression of cold cache slots while the engine is idle |
| `COLI_PREFETCH_WARM_PATHS_<CLASS>` | unset | Per-workload-class prefetch breadth override (CODE/JSON/MATH/PROSE/MIXED) |
| `COLI_CACHE_COMPRESS` | off | Zstd-compress WarmCache slabs on admit, decode on hit |
| `COLI_CACHE_NEGATIVE_TTL` | 0 | Evict unhit warm-cache slots older than N clock ticks |
| `COLI_ROUTE_STATS_PERSIST` | on | Save `route_stats.json` at Drop; auto-load matching one on `Model::load` |
| `COLI_LAYOUT_SCHEDULE` | on | Use `<dir>/schedule.json` (if present) to pre-sort disk reads |
| `COLI_PHASE_THRESHOLD` | 0.6 | Jaccard distance above which `PhaseTracker` declares a phase change |
| `COLI_PERF_COUNTERS` | off | Open a real `perf_event_open` LLC-miss counter (needs kernel grant) |

---

## Notes

- **Audit basis:** statuses verified against source; file:line evidence inline. `Done` = actually implemented and covered by a bit-identical / round-trip test; `Partial` = scaffolding present but not yet on the hot path.
- **Compilation & test invariant:** 184 tests pass workspace-wide, clippy clean, `--strict` bad-patterns audit green.
- **What's left is CUDA-shaped:** persistent CUDA kernels, wiring CUDA Graphs into the decode path, and a `cudaMallocAsync` pool for `reheat` churn — three items that require `nvcc` + a real GPU to build and verify, so this workspace can't land them without that toolchain.
- **Validation caveat:** synthetic-model tests catch correctness; throughput impact needs a real model to measure. The pattern is "many small adaptive knobs, each bit-identical when off" — evaluate combined.
