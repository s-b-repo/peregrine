# peregrine — Optimization Checklist

Roadmap of throughput/latency optimizations for streamed-MoE inference, distilled from `todo.txt`,
with **implementation status audited against the codebase** (2026-07-24; §1 Prefetching completed 2026-07-25).

Goal: **maximize tokens/sec for giant streamed MoE models** by eliminating the remaining sources of
idle time — SSD latency, GPU launch overhead, and suboptimal expert placement — on top of the existing
concurrent CPU/GPU/SSD scheduler + `io_uring`.

**Status legend:** `- [x]` ✅ Done · `- [ ] 🟡` Partial (scaffolding / incomplete) · `- [ ]` ⬜ Not started
**Ratings** (where the source ranked them) — gain ★☆☆☆☆→★★★★★ · Difficulty: Easy/Medium/Hard

---

## 📊 Completion Dashboard

| Scope | ✅ Done | 🟡 Partial | ⬜ Not started | Total | Completion |
|---|---:|---:|---:|---:|---:|
| **Full roadmap** | 19 | 13 | 61 | 93 | **~20% strict · ~27% weighted** |
| **Priority shortlist** | 8 | 2 | 9 | 19 | **~42% strict · ~47% weighted** |

*Strict = Done ÷ Total. Weighted = (Done + ½·Partial) ÷ Total. "Fast matrix multiplication" is excluded (deliberately out of scope).*

**Per-section:** Prefetch **9/9 ✅** · GPU 3/8 · Caching 3/10 · I/O 3/11 · Memory/NUMA 0/5 · Scheduling 1/16 ·
Disk-layout 0/10 · Workload 0/5 · Compilation 0/5 · Self-optimizing 0/10 · Multi-GPU 0/4.

**Headline:** the *foundation* is largely built (see below) and the entire **Prefetching & Speculation section is now
complete** (all 9 items — momentum + layer look-ahead + multi-path + fadvise + distance tuner + predictive eviction +
offline transition automaton + verification, plus per-sequence parallel-async prefetch in the batched engine). The
long tail of research/exotic items (scheduling governors, disk-layout optimization, self-optimizing runtime, multi-GPU)
remains untouched. All new work is bit-identical (prefetch/eviction/prediction affect only performance) and clippy-clean.

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

---

## ⭐ Top Priority Shortlist

Highest expected throughput per unit effort. **5 done, 2 partial, 12 to go.**

- [x] ✅ **Layer look-ahead prefetch** — per-layer emission mid-forward, staggered ahead of the compute cursor (`PrefetchCtx::emit_layer`) _(★★★★★ · Medium)_
- [ ] ⬜ **Persistent CUDA kernels** — launch once, threadblocks loop `dequeue → compute → enqueue` _(★★★★★ · Hard)_
- [ ] 🟡 **CUDA Graphs** — capture/replay built but **not wired into decode loop** `backend_cuda.cu:453-496`, `cuda/lib.rs:394-450` _(★★★★☆ · Medium)_
- [x] ✅ **Dynamic expert VRAM cache** — `reheat()` re-selects hottest experts by routing frequency every 256 steps `gpu.rs:363-382` _(★★★★☆ · Hard)_
- [x] ✅ **Triple-buffered pipeline** — read ∥ compute ∥ (H2D∥kernel∥D2H) overlap `concurrent.rs:344-509`, `backend_cuda.cu:653-739` _(★★★★☆ · Medium)_
- [x] ✅ **Adaptive expert cache** — LFRU score `(heat<<8)|recency`, not plain LRU `tier.rs:18-56` _(★★★★☆ · Medium)_
- [x] ✅ **Pinned memory + async copies** — `cudaMallocHost` staging + `cudaMemcpyAsync` `backend_cuda.cu:356-359,610-611` _(★★★☆☆ · Easy)_
- [ ] ⬜ **Huge pages** — no `MAP_HUGETLB` / `MADV_HUGEPAGE` anywhere _(★★★☆☆ · Easy)_
- [x] ✅ **Lock-free work stealing** — atomic `io_work.fetch_add` across N io_uring rings `concurrent.rs:352-380` _(★★★☆☆ · Medium)_
- [ ] ⬜ **Adaptive CPU/GPU work balancing** — placement is static, no runtime rebalance from observed latency
- [ ] ⬜ **Runtime expert replication for hot experts** — each expert lives in at most one tier
- [ ] ⬜ **GPUDirect Storage** — no direct SSD→VRAM path
- [x] ✅ **Dynamic prefetch distance tuning** — `PrefetchTuner` EWMA over used/wasted adapts warm breadth
- [x] ✅ **Learned cache admission & eviction** — predictive protected-set eviction (predictor + heat → cache priority); pragmatic (heuristic, not a trained model)
- [ ] ⬜ **Hardware-counter-driven scheduler feedback** — no perf-counter integration
- [ ] 🟡 **Offline checkpoint re-layout from routing traces** — routing history *recorded* `concurrent.rs:44-48`, but no re-layout applied
- [ ] ⬜ **Online kernel autotuning** — WMMA tile sizes hardcoded
- [ ] ⬜ **Pipeline bubble detection & rebalancing** — no stall detection
- [ ] ⬜ **Multi-GPU expert ownership & migration** — single device (hardcoded `device=0`)

---

## 1. Prefetching & Speculation — 9/9 ✅

Shared spine in `predict.rs` (`RouteHistory` K-deep + `PredictSource` momentum/automaton + `PrefetchTuner` +
`TransitionTable`) feeding a per-layer emitter (`PrefetchCtx::emit_layer`) and a parallel-async lane pool. All
bit-identical (prefetch/eviction/prediction affect performance only) and clippy-clean.

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
(`forward_step_batched` per-sequence `route_log_multi`, `batch.rs` field-split unzip).

## 2. GPU Execution — 3/8

- [ ] ⬜ Persistent CUDA kernels _(★★★★★ · Hard)_ — kernels launched per-batch, no threadblock loop
- [ ] 🟡 CUDA Graphs — capture/instantiate/launch implemented + tested, **not integrated into decode** `backend_cuda.cu:453-496` _(★★★★☆ · Medium)_
- [ ] 🟡 Fused MoE pipeline — `expert_group` fuses gate/up/silu/down on GPU; layer-level accumulation still separate/deterministic `backend_cuda.cu:626-752`
- [x] ✅ Zero-copy GPU uploads via pinned memory `backend_cuda.cu:356-359,610-611` _(★★★☆☆ · Easy)_
- [x] ✅ Persistent GPU memory pools — 24 pre-allocated scratch slots reused across layers `backend_cuda.cu:892-899`
- [ ] ⬜ GPU memory defragmentation during decode — residents fixed at startup
- [ ] ⬜ Online kernel autotuning for GEMM tile sizes — WMMA tiles hardcoded (16×16×16 / 8×8×32)
- [x] ✅ Runtime SIMD kernel selection (CPU) — AVX2 vs AVX-VNNI chosen at runtime `idot.rs:40-61`

## 3. Caching & VRAM Residency — 3/10

- [x] ✅ Adaptive expert cache — LFRU (frequency dominates recency 256×), hysteresis to avoid ping-pong `tier.rs:18-56`, `gpu.rs:56-107` _(★★★★☆ · Medium)_
- [ ] 🟡 Quantized RAM cache — warm cache holds quantized bytes verbatim; CPU path computes on INT4 without dequant, GPU dequants on upload `warmcache.rs:34-51`
- [x] ✅ Dynamic GPU residency — `reheat()` heat-ranked VRAM re-selection `gpu.rs:363-382` _(★★★★☆ · Hard)_
- [ ] 🟡 Heat-wave scheduling — speculative prefetch is naive history replay; no topic/phase-shift detection `model.rs:245-270`
- [ ] ⬜ "Negative" caching — no long-unused tracking
- [ ] ⬜ Persistent Expert Residency Solver — greedy single-step, no IP/constraint solver
- [ ] 🟡 Cache admission from estimated future reuse — implicit via prefetch-warm; no reuse estimator `model.rs:694-728`
- [x] ✅ Learned cache admission & eviction — predictive protected-set eviction (predictor + heat → opaque cache priority, `WarmCache` `(prio,recency)` victim order); see §1 predictive eviction. Pragmatic (heuristic, not a trained model)
- [ ] ⬜ Bloom filter / probabilistic cache lookup — HashMap lookups
- [ ] ⬜ Runtime expert replication/duplication for hot experts

## 4. I/O & Storage — 3/11

- [x] ✅ Register frequently-used memory buffers with `io_uring` — `register_read_buffers()` + `IORING_OP_READ_FIXED` `ring.rs:157-221`
- [x] ✅ Batch I/O intelligently — `read_many()` / `read_experts_batched()` merge contiguous regions `ring.rs:251-288`, `concurrent.rs:225-254`
- [x] ✅ Double / triple buffering — 3-lane concurrent overlap `concurrent.rs:344-509` _(★★★★☆ · Medium)_
- [ ] ⬜ Compressed expert storage (Zstd) with parallel decompress — none (stored quantized)
- [ ] ⬜ Transparent expert compression in RAM
- [ ] ⬜ Background expert recompression when idle
- [ ] ⬜ GPUDirect Storage (GDS) support
- [ ] ⬜ Learned SSD read scheduler
- [ ] ⬜ Disk queue-depth autotuning — ring depth fixed at init
- [ ] 🟡 Adaptive `io_uring` SQ/CQ sizing — depth + `set_iowq_max_workers()` tunable, but no dynamic tuning `ring.rs:67,140-143`
- [ ] ⬜ Fault-tolerant I/O recovery + degraded-mode execution — I/O errors abort the forward

## 5. Memory & NUMA — 0/5

- [ ] ⬜ Huge pages (2 MB / 1 GB) _(★★★☆☆ · Easy)_
- [ ] ⬜ Automatic huge-page allocation and promotion
- [ ] ⬜ NUMA-aware scheduling — avoid cross-domain expert-weight moves
- [ ] ⬜ NUMA-aware RAM allocation and thread placement
- [ ] 🟡 Lock-free slab allocator with recycling by generation — `SlabPool` recycles via free-list, but **not lock-free and no generation tags** `slab.rs:189-245`

*Note: weight loading uses `pread` + `fadvise(DONTNEED)` (flat RSS), not `mmap` — deliberate `safetensors.rs:3`.*

## 6. Scheduling & Work Distribution — 1/16

- [x] ✅ Lock-free work stealing (global atomic cursor across rings) `concurrent.rs:352-380` _(★★★☆☆ · Medium)_
- [ ] ⬜ CPU/GPU split GEMM — experts route wholly to one device
- [ ] ⬜ Cooperative expert execution (tiled dispatch)
- [ ] ⬜ Adaptive CPU/GPU work balancing from observed execution time
- [ ] ⬜ Idle-cycle computation — GPU does no speculative compute during waits
- [ ] ⬜ Runtime expert fusion — experts computed independently
- [ ] ⬜ Pipeline bubble detection with automatic rebalance
- [ ] ⬜ Hierarchical task scheduler (socket → core → worker) — flat pool
- [ ] ⬜ Priority inheritance for latency-critical decode — single unbounded FIFO queue
- [ ] ⬜ Adaptive batching window based on latency SLA — `max_batch` static
- [ ] ⬜ Memory bandwidth governor
- [ ] ⬜ Dynamic PCIe bandwidth scheduler
- [ ] ⬜ Thermal-aware scheduling
- [ ] ⬜ Energy-aware scheduling (tokens/watt)
- [ ] ⬜ Expert hypergraph scheduling
- [ ] ⬜ Execution entropy minimization

## 7. Disk Layout & Offline Optimization — 0/10

- [ ] ⬜ Expert clustering — experts stored in checkpoint (numerical) order `pack.rs`
- [ ] ⬜ Routing-aware physical disk layout
- [ ] ⬜ Routing locality optimization (training-time penalty)
- [ ] ⬜ Hierarchical disk space-filling layout
- [ ] ⬜ Offline expert graph partitioning
- [ ] ⬜ Hypergraph-based expert placement across storage tiers
- [ ] 🟡 Automatic checkpoint re-layout based on routing history — history *recorded* (`route_log`) but never applied to persist a new layout `concurrent.rs:44-48`
- [ ] ⬜ Expert graph clustering via community detection
- [ ] ⬜ Offline "galactic" preprocessing pass (co-occurrence graphs, transition probs)
- [ ] ⬜ Graph optimizer for near-optimal reusable schedules

## 8. Workload Adaptation & Phase Detection — 0/5

- [ ] ⬜ Token-shape scheduling (classify code/JSON/prose → prefetch per class)
- [ ] ⬜ Inference phase detection (swap caches on English→Python→JSON transitions)
- [ ] 🟡 Continuous prefill/decode optimization — separate optimized paths exist, but no *adaptive* detection/switching `model.rs:865-891`
- [ ] ⬜ Automatic workload classification (code / prose / JSON / math)
- [ ] ⬜ Temporal compression of routing (macro-states)

## 9. Compilation & Specialization — 0/5

- [ ] ⬜ Whole-model execution compiler (routing graph → IR → binary) — forward is interpreted
- [ ] ⬜ Profile-guided inference compilation (PGO)
- [ ] ⬜ Runtime specialization of hot paths (JIT / codegen)
- [ ] ⬜ Tensor layout auto-conversion for cache locality
- [ ] 🟡 Mixed-precision execution per expert — f32 on GPU vs INT4 on CPU, but **static by tier, not per-expert-adaptive** `gpu.rs:3-12`

## 10. Learning-Based & Self-Optimizing Runtime — 0/10

- [ ] ⬜ Learning-based scheduler (trained policy)
- [ ] ⬜ Reinforcement learning scheduler
- [ ] ⬜ Self-reorganizing models (rewrite on-disk layout from stats)
- [ ] 🟡 Self-rewriting runtime — `reheat()` gives dynamic VRAM residency, but no graph/layout/policy rewrite `gpu.rs:363-383`
- [ ] 🟡 Cross-session routing statistics database — `HeatTable` accumulates per-session, **reset on model load**, not persisted `gpu.rs:64-86`
- [ ] ⬜ Live execution-plan optimization from telemetry
- [ ] ⬜ Hardware performance counter feedback (cache misses, bandwidth, occupancy)
- [ ] ⬜ Runtime topology discovery (PCIe / NVLink / NUMA) — single device only
- [ ] ⬜ Automatic expert fusion from long-term co-activation
- [ ] 🟡 **"Living inference engine"** capstone — partially embodied by heat-driven residency; no learned policies / cross-session memory / model rewriting

*Building block present: routing-frequency stats collection (`HeatTable`, lock-free atomic bump) `gpu.rs:59-86`, `concurrent.rs:530-535` — the substrate future self-optimizing features would build on.*

## 11. Multi-GPU & Distributed — 0/4

- [ ] ⬜ Multi-GPU expert ownership with work migration — hardcoded `device=0` `gpu.rs:328`
- [ ] ⬜ NVLink-aware multi-GPU expert placement
- [ ] ⬜ Runtime expert replication in VRAM
- [ ] ⬜ Distributed inference across multiple hosts with expert sharding

---

## ❄️ Deprioritized / Noted (excluded from %)

- [ ] ~~Fast matrix multiplication (Strassen, Coppersmith–Winograd, Williams')~~ — asymptotically cheaper but almost never wins for LLM inference. **Skip unless proven otherwise.** (Confirmed absent — custom tiled GEMM used instead.)

---

## Notes

- **Audit basis:** statuses verified against source on 2026-07-24 via subsystem sweeps; file:line evidence inline. `Done` = actually implemented; `Partial` = scaffolding/related mechanism but not the full item.
- **Biggest near-term win still open:** layer look-ahead prefetch — the current next-token prefetch (`enqueue_prefetch`) is a solid base to extend into intra-forward N+1/N+2 look-ahead.
- **Cheap wins available:** huge pages (Easy, not started), wiring the already-built CUDA Graphs into decode (Partial → Done), and activating the existing `fadvise_willneed()` API on the main path.
- **Validation caveat:** all of these need empirical benchmarking — biggest wins usually come from combining several modest improvements into a coherent runtime, not one dramatic optimization. (Per verify-env: no real model → judge by correctness/counters/tests, not throughput.)
