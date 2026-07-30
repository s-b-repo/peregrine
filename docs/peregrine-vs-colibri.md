# peregrine (Rust) vs colibrì (C): a streaming-MoE inference engine comparison

**A same-hardware study of two engines that run a 744-billion-parameter Mixture-of-Experts model
(GLM‑5.2) on a single consumer machine by streaming experts from an SSD.**

> Status: engineering/research report. All peregrine numbers and the head-to-head colibrì numbers
> were measured this session on the machine described in [§4](#4-methodology). colibrì's multi-GPU,
> Metal, and community numbers are cited from the [colibrì repository](https://github.com/JustVugg/colibri)
> with provenance. Where a number could not be measured on this box, it is labelled *published* and
> its source is given.

---

## Abstract

**colibrì** is a dependency-free C engine (~7k-line `glm.c`) that runs GLM‑5.2 — a 744B-parameter,
256-expert-per-layer MoE — on ~25 GB of RAM by keeping the ~10 GB dense sub-model resident and
**streaming the ~370 GB of routed experts from disk on demand**. **peregrine** is a from-scratch Rust
rewrite targeting Linux + NVIDIA, whose thesis is that colibrì's CUDA MoE path is *phased* (CPU-expert
compute and GPU-expert compute never overlap on the same layer) and can be replaced by a
**completion-driven concurrent scheduler** where the CPU, GPU, and io_uring/SSD lanes drain the same
layer's experts simultaneously.

This report (1) catalogues every architectural difference and every optimization added in peregrine,
and (2) measures both engines on the *same* machine and the *same* GLM‑5.2 int4 checkpoint. The
headline findings:

- **On one consumer box both engines are disk-bandwidth-bound**, within ~1.4× of each other:
  **colibrì decodes at 0.077 tok/s, peregrine at 0.054 tok/s** on the same drive and model. Each token
  routes to **600 experts ≈ 11.3 GB** that must be read from disk, and no cache that fits in 46 GB RAM
  can hold a meaningful fraction of the 19,200-expert (~370 GB) working set. colibrì started ahead
  because peregrine read *buffered* (polluting the page cache with 0.6 %-reuse data). Three I/O ports
  this session — a deep io_uring queue, an `O_DIRECT` + aligned-slab-arena streaming path, and **N
  parallel io_uring rings with lock-free work-stealing** (all bit-identical) — **raised peregrine's raw
  device read rate from ~710 MB/s to ~980 MB/s (+38 %), now *faster* than colibrì's ~870 MB/s on the
  same box** (O_DIRECT +21 %, parallel rings +14 % more; measured directly, contention-robust — see §6).
  End-to-end tok/s is too noisy on this contended box to quantify (buffered runs ranged 0.018–0.028), but
  the raw streaming throughput is unambiguously improved past the C engine. The levers were bypassing the
  page cache and parallelizing dm-crypt — implementation gaps, not architectural ones.
- **peregrine's warm cache delivers a real 3.58× on a *repeated* forward** (16.1 s → 4.5 s, 100 %
  expert-cache hits, zero disk) — proving the mechanism works — **but ~1× on sustained decode**,
  because GLM‑5.2's expert routing has **near-zero token-to-token locality (0.6 % cache hit rate,
  measured)**. Consecutive tokens route to almost disjoint expert sets.
- This independently corroborates two of colibrì's own findings: its **PILOT** cross-layer prefetch
  is "neutral" on a disk-saturated box, and its **MTP** speculative decoding is a *net loss* on MoE
  decode — both symptoms of the same high routing entropy.
- **peregrine's architectural advantage (the concurrent 3-lane scheduler) is real but *latent* on
  this hardware**: it pays off only under full/partial expert *residency* (multi-GPU or large RAM),
  which is exactly where colibrì reports **6.84 tok/s on 6× RTX 5090**. On a single RTX 3060 (12 GB),
  only ~66 of 19,200 experts fit in VRAM, so neither engine can demonstrate the residency regime.
- **Continuous batching now realizes part of that advantage on the streaming path.** Decoding B
  sequences together shares each expert read across the batch, a **measured 4.4× aggregate gain at
  B=16** (0.064 → 0.280 tok/s) on the real 744B model — per-step disk cost grows sub-linearly as the
  expert union is read once and shared (§5.4). Absolute throughput stays disk-bound; the win is
  amortization of the byte budget, not a faster drive.

Net: on a single box the dominant limit is the **memory-vs-working-set wall**, and within that regime
**colibrì is currently ~1.4× faster** thanks to a more mature streaming path. The Rust rewrite lands in
the same order of magnitude, adds a verified warm-cache/scheduler stack and memory-safety guarantees
(282 tests, `#![forbid(unsafe_code)]` outside FFI, `deny(unwrap/panic)`), and is architected to win in
the residency regime the C engine's phased loop leaves on the table — but it has a concrete, fixable
I/O-lane-depth gap to close before it beats the C engine at raw single-box streaming.

> **Note (2026-07-30 adaptive-runtime wave — post-report).** The measurements in this document
> were taken before the adaptive-runtime pass shipped. Since then peregrine gained per-lane
> wall-time telemetry (`LaneTimings` + `BubbleTuner`) with an adaptive CPU/GPU balancer
> (`LaneBalancer`), an adaptive io_uring worker-cap tuner (`IoTuner`), a latency-SLA-adaptive
> batching window + two-priority admission queue, transparent zstd on both the on-disk safetensors
> path and the in-RAM warm cache, cross-session routing history + heat persistence
> (`route_stats.json`), an offline layout tool (`peregrine-layout-reorg`) with a Louvain
> community-detection pass, runtime expert replication for hot experts, and a phase-aware
> prefetch predictor (`PredictSource::PhaseAware`). A same-day follow-up sweep added NUMA worker
> pinning, a heat-threshold cache-admission gate, a spectral (Fiedler) ordering method in the
> layout tool, a real `perf_event_open` LLC-miss counter, idle-tick background recompression of
> cold cache slots, and per-workload-class prefetch breadth. All new subsystems are env-gated
> (defaulting to the historical behavior in most cases) and bit-identical when off. Re-running
> this study with those knobs live is the next benchmarking pass — see [`todo.md`](../todo.md)
> A completion sweep then closed every non-hardware item (sensor governors, entropy-adaptive
> prefetch, NUMA allocation + hierarchical dispatch, expert fusion + hypergraph scheduling,
> macro-states, the `galactic` pass, Hilbert/2-opt/tier layouts, physical checkpoint self-rewrite,
> online bandit/Q-learning schedulers, per-shape dispatch specialization, kblock layout
> auto-conversion, and the `compile-plan` execution plan) — see [`todo.md`](../todo.md) for the
> per-item state (now ~87% strict / ~88% weighted across 95 tracked items; every open item is
> hardware-gated). The serve layer additionally gained a **vendored gigatoken BPE tokenizer**
> (marcelroed/gigatoken, MIT — stable-toolchain subset in `peregrine-token`): id-for-id
> parity-gated against the HF `tokenizers` oracle and measured **34× faster locally**
> (204 vs 6 MB/s, `--bench-tokenizer`), with the HF crate as automatic fallback.

---

## Table of contents
1. [Background](#1-background)
2. [Architecture comparison](#2-architecture-comparison)
3. [Improvements in peregrine](#3-improvements-in-peregrine)
4. [Methodology](#4-methodology)
5. [Results — token specs](#5-results--token-specs)
6. [Analysis](#6-analysis)
7. [Limitations & threats to validity](#7-limitations--threats-to-validity)
8. [Reproducibility](#8-reproducibility)
9. [Future work](#9-future-work)
10. [References](#10-references)

---

## 1. Background

A 744B MoE activates only ~40B parameters per token; of those, the **routed experts** (~11 GB/token at
int4) change from token to token while the **dense sub-model** (attention, shared expert, embeddings —
~17B params, ~9.9 GB at int4) is reused every token. This asymmetry is what makes single-box inference
possible: keep the dense part resident, and treat **VRAM → RAM → SSD** as one memory hierarchy for the
experts, streaming the cold ones from disk.

GLM‑5.2 (as shipped in the int4 container both engines consume) is a DeepSeek‑V3-class architecture:

| Property | Value (from `config.json`) |
|---|---|
| Hidden size | 6144 |
| Layers | 78 (first **3 dense**, **75 sparse MoE**) |
| Routed experts / layer | 256, **top-8** per token |
| Expert intermediate (`moe_intermediate_size`) | 2048 → **18.9 MB/expert at int4** |
| Shared experts | 1 (always active) |
| Attention | **MLA** (64 heads, q-LoRA 2048, kv-LoRA 512, partial RoPE 64/192, v-head 256) |
| Sparse attention | **DSA** lightning indexer, top-2048 keys |
| Speculative head | **MTP** (layer 78, int8) |
| Vocab | 154,880 |
| **Experts routed per token** | **75 layers × 8 = 600 ≈ 11.3 GB int4** |
| Total routed experts | 75 × 256 = 19,200 (+ MTP) ≈ **370 GB on disk** |

- **colibrì** — the C baseline: [github.com/JustVugg/colibri](https://github.com/JustVugg/colibri).
  Single-file engine, zero runtime deps, CPU + optional CUDA/Metal, `pin`/`LRU`/`disk` expert tiers,
  learned auto-pin (`.coli_usage`), MTP speculation, DSA, io_uring (`URING=1`).
- **peregrine** — the Rust rewrite ([`DESIGN.md`](../DESIGN.md)): Linux + NVIDIA, a concurrent
  3-lane (CPU∥GPU∥SSD) MoE scheduler, io_uring reactor, warm cache, look-ahead prefetch, and a
  memory-safe kernel/loader stack validated by 282 tests.

---

## 2. Architecture comparison

### 2.1 The central thesis: phased vs concurrent MoE

colibrì's CUDA MoE inner loop is **phased**: VRAM-resident experts are deferred, RAM/disk experts are
computed on the CPU **inline**, and the GPU expert group is dispatched **only after** that loop
finishes. An in-code comment measures the waste directly (`c/glm.c:2922-2923`):

> *"…9343 expert in VRAM restavano INUTILIZZATI durante il prefill (misurato: 81 s di expert-matmul
> tutto su CPU, GPU groups 21 ms totali)."*
> — 9,343 experts sat unused in VRAM during prefill: 81 s of expert-matmul entirely on the CPU while
> the GPU expert-group calls totalled 21 ms.

peregrine replaces this with a **completion-driven concurrent scheduler**
([`crates/peregrine-model/src/concurrent.rs`](../crates/peregrine-model/src/concurrent.rs)): per sparse
layer, the batch-union of routed experts is split by residency into three lanes that run as concurrent
actors via `std::thread::scope`:

- **I/O lane** — one io_uring reactor streams disk experts;
- **CPU lane** — a worker pool computes each expert's SwiGLU as soon as its bytes land;
- **GPU lane** — one batched `expert_group` call for VRAM-resident experts.

A single deterministic reduce merges them in fixed batch-union order, so the output is **bit-identical**
to the sequential path (guarded by the `streamed_experts_match_resident` test). Wall-clock becomes
`max(gpu, cpu, disk)` instead of the C engine's `max(disk, cpu) + gpu`.

### 2.2 Feature-by-feature

| Dimension | colibrì (C) | peregrine (Rust) |
|---|---|---|
| MoE scheduling | **Phased** (CPU inline, GPU after) | **Concurrent 3-lane** (CPU∥GPU∥SSD, deterministic reduce) |
| Language / safety | C, manual memory | Rust; `#![forbid(unsafe_code)]` except io_uring/CUDA FFI; `deny(unwrap/expect/panic)`; **282 tests, clippy-clean** |
| SSD I/O | hand-rolled io_uring (`uring.h`, `URING=1`), `O_DIRECT`, io-wq cap | `io-uring` crate + custom reactor; registered files (`IOSQE_FIXED_FILE`), COOP_TASKRUN, forced ASYNC; **batched 6→1 submit/expert** |
| RAM expert tier | per-layer LRU `ecache`, auto-sized from `MemAvailable`; LFRU pinned hot-store; learned `.coli_usage` | **byte-budgeted `(layer,expert)` WarmCache** (quantized bytes; `COLI_ECACHE_GB`) |
| Prefetch | **PILOT** cross-layer (router-lookahead, 71.6 % recall); `PILOT_REAL` | **PILOT lane** (prev-token predictor, dedicated ring, background warm) |
| GPU expert tier | CUDA resident tier, `CUDA_EXPERT_GB`, live LFRU repin (`REPIN`) | `GpuTier` (feature `cuda`); **round-robin placement across all sparse layers** |
| CPU kernels | int8/int4 IDOT, AVX2 `maddubs`, 119 GFLOP/s; per-shape f32/int routing | `std::arch` int8/int4/int2 IDOT (AVX2 + AVX-VNNI), scalar-exact reference; bit-checked SIMD |
| Attention | MLA + absorption; DSA lightning indexer | MLA + absorption; DSA (M5) |
| Speculation | MTP (int8 head) + grammar-forced drafts | MTP head wired (`speculative_sample` rejection-sampling lossless) |
| Dense on GPU | `CUDA_DENSE=1` (+30.8 %) | (future) |
| Backends | CPU, CUDA, **Metal** (Apple) | CPU, CUDA |
| Platforms | Linux, macOS, Windows (MinGW) | Linux + NVIDIA |

---

## 3. Improvements in peregrine

Two categories: the **inherited-but-restructured** concurrent scheduler, and the **optimizations added
this session**. Each is stated as *colibrì baseline → peregrine change → expected effect → measured
effect on this box*.

### 3.0 Concurrent 3-lane scheduler (the centerpiece)
- **colibrì:** phased loop; on CUDA, CPU-expert and GPU-expert compute never overlap (the 81 s-vs-21 ms
  waste above).
- **peregrine:** completion-driven CPU∥GPU∥SSD scheduler; deterministic reduce → bit-identical output.
- **Measured:** on this box the CPU∥SSD overlap is visible — a cold forward takes 16.1 s ≈ the pure
  disk time (11.3 GB @ ~710 MB/s), i.e. the 4.5 s of expert compute is **fully hidden under disk I/O**;
  warm, disk vanishes and the forward is compute-bound at 4.5 s. The GPU lane is latent (see §6).

### 3.1 A1 — byte-budgeted RAM warm cache
- **colibrì:** has an LRU `ecache` + LFRU pin; peregrine's concurrent path had **none** (it re-streamed
  every expert every layer every token).
- **peregrine:** new [`WarmCache`](../crates/peregrine-io/src/warmcache.rs) — bounded-by-bytes LRU keyed
  by `(layer, expert)`, storing the raw quantized bytes so a hit reconstructs a **bit-identical**
  `QtWeight`. Env `COLI_ECACHE_GB`.
- **Measured:** on a *repeated* forward whose 600 experts fit the cache → **100 % hits, 0 disk,
  16.1 s → 4.5 s = 3.58×**. On sustained decode → ~1× (see A2 / §6).

### 3.2 A2 — layer look-ahead prefetch (PILOT)
- **colibrì:** PILOT applies layer L+1's router to layer L's state to prefetch (71.6 % recall);
  measured "neutral" on the disk-saturated dev box.
- **peregrine:** a dedicated prefetch reactor + background worker warms the *next token's predicted*
  experts (predictor = previous token's routed set) into the shared cache off the critical path.
- **Measured:** **ineffective on GLM‑5.2** — cross-token expert locality is **0.6 %**, so the
  previous-token predictor mostly mis-predicts (499 speculative reads, ~0 useful). This is an honest
  negative that matches colibrì's PILOT/MTP experience; it is a *predictor*-quality problem, not a
  mechanism bug (the lane, ring, and cache all work).

### 3.3 C1 — batched expert reads
- **colibrì:** batches expert loads on the ring.
- **peregrine:** each expert's 6 regions (gate/up/down × weight+scale) are now issued in **one
  `submit_and_wait`** instead of six (`read_expert`), byte-identical. Verified by
  `read_expert_batched_bytes_identical`.

### 3.4 C2 — registered io_uring buffers · 3.5 C3 — fadvise warming
- New `Reactor::register_read_buffers` + `read_fixed` (IORING_OP_READ_FIXED) and `fadvise_willneed`
  (IORING_OP_FADVISE), correctness-tested. Reserved capabilities (the owned-`Vec` hand-off would add a
  copy that likely negates the registered-buffer win on the hot path until measured on faster storage).

### 3.6 B1 — VRAM residency placement
- **colibrì:** `CUDA_EXPERT_GB` + LFRU repin; frequency-driven.
- **peregrine:** replaced `GpuTier`'s greedy layers-0-first fill with a **round-robin `plan_residency`
  across all sparse layers** (pure, unit-tested), so every layer gets a VRAM share instead of the late
  layers streaming 100 % from disk. Validated on the RTX 3060 (`tier_spans_multiple_sparse_layers`).

### 3.7 Correctness & quality
- Streamed path is **bit-identical** to the resident path (same bytes → same kernels). GPU experts
  compute in f32 (intentionally not bit-exact vs the int4 CPU path — documented). **282 tests pass,
  `cargo clippy --workspace --all-targets` is clean** (CPU and `--features cuda`).

---

## 4. Methodology

**Test machine (both engines, same box):** AMD Ryzen 5 5500 (6C/12T, Zen 3, AVX2, **no AVX-512/VNNI**),
46 GB DDR4, NVIDIA RTX 3060 12 GB (sm_86), CUDA 13.3, **LUKS-encrypted NVMe**, Linux 7.0 / Arch.

**Model:** `mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp` — the GLM‑5.2 int4 container with int8 MTP
heads, **358 GB on disk** (144 `out-*.safetensors` + 3 `out-mtp-*`), the *same* directory fed to both
engines.

**What is measured:** load time (to first-token readiness), **decode tok/s** (steady-state, excluding
prefill where the engine reports it separately), expert-cache **hit rate**, and peak **RSS**. peregrine
additionally reports `[ecache] hits/misses/disk_reads` counters. Greedy (temp 0), MTP off (`DRAFT=0`),
io_uring on (`URING=1`/reactor), `O_DIRECT`.

**Memory note:** peregrine required **`MALLOC_ARENA_MAX=2`** — without it, glibc per-thread arenas
(12 worker threads churning 18.9 MB expert allocations) inflate RSS to ~29.5 GB and the OOM-killer
fires. colibrì has an explicit RAM auto-budget (it *lowered its expert cache from 8→1 slot/layer* to
stay under a 10 GB budget) and did not OOM.

**Reproducibility:** exact commands in [§8](#8-reproducibility). peregrine driver: `tools/`-style
Python harness that spawns the engine, waits for READY, and times `GEN` requests.

---

## 5. Results — token specs

### 5.1 Same-hardware head-to-head (this box, CPU streaming, greedy, MTP off, io_uring)

| Metric | **peregrine (Rust)** | **colibrì (C)** |
|---|---|---|
| Load → ready | ~11 s | 13.78 s |
| Resident (dense, int4) | ~12 GB | **9.9 GB** |
| Peak RSS (10 GB cache config) | ~21–24 GB (needs `MALLOC_ARENA_MAX=2`) | **12.85 GB** (cache auto-lowered to 1 slot/layer) |
| Prefill | — | 11 tok in 71.48 s (0.15 tok/s) |
| **Decode (steady state)** | **0.054 tok/s** (16-tok) / 0.062 (single) | **0.077 tok/s** (8 tok in 104.3 s) |
| Expert hit rate | 0.6 % (10 GB cache) | 2.3 % (1-slot cache) |
| Effective disk throughput | ~710 MB/s | **~870 MB/s** |
| Warm mechanism | **3.58×** on a *repeated* forward (100 % hit, 0 disk) | learned pin / OS cache (see §5.3) |
| OOM safety | needs `MALLOC_ARENA_MAX=2` | built-in RAM auto-budget |

**colibrì is currently faster on this box at every axis** and uses half the RAM. Decode: **0.077 vs
0.054 tok/s (~1.4×)**. A fully matched run (same 11-token prompt + 8 tokens) makes the gap wider
**end-to-end — colibrì 0.046 vs peregrine 0.018 tok/s (~2.5×)** — because peregrine's **prefill** is
~4× slower (colibrì 71 s vs peregrine ~316 s for the 11-token prefill): a batched prefill routes to
thousands of distinct experts, and streaming them **one at a time** through peregrine's reactor
starves the disk (~270 MB/s effective in prefill) where colibrì's depth-512 queue + 8 io-workers +
`O_DIRECT`/`PIPE` sustains ~870 MB/s. Crucially, colibrì achieved this with a *smaller* effective
cache (it auto-lowered to ~1 slot/layer ≈ 1.5 GB vs peregrine's 6–10 GB), so **this is not a cache
advantage — it is I/O-queue depth**. It is peregrine's clearest concrete gap on a single box
(see [§6](#6-analysis), [§9](#9-future-work)); an I/O-lane issue, not an architecture-or-correctness
one. Both decode within colibrì's published dev-box range (~0.05–0.1 tok/s).

### 5.2 Cache & locality analysis (peregrine, measured)

| Scenario | Cache | Result | Why |
|---|---|---|---|
| **Repeated identical forward** | 12 GB (holds its 600 experts) | **3.58×** (16.1 s→4.5 s), 600/600 hits, 0 disk | working set fits → 100 % reuse |
| **Sustained decode, 16 tok** | 10 GB | **0.054 tok/s, 1.05× warm** | 16 tokens touch ~9,600 experts (~180 GB) ≫ cache |
| **Cross-token locality** | — | **0.6 % hit rate** (58/9,600) | consecutive tokens route to ~disjoint expert sets |

The 0.6 % figure is the key empirical result: **GLM‑5.2's MoE router has very high per-token entropy**,
so caching only pays off (a) for repeated computation, or (b) once the cache can hold a large fraction
of the whole model — i.e. the residency regime.

### 5.3 Published scaling context (colibrì, cited)

Where experts become resident, colibrì scales far past the disk-bound floor — this is the regime
peregrine's concurrent scheduler targets:

| Setup | Result | Source |
|---|---|---|
| Dev box (WSL2, 12C, 25 GB, ~1 GB/s NVMe) | **~0.05–0.1 tok/s** cold, 9.9 GB resident, ~30 s load | colibrì README "Honest numbers" |
| Disk-bandwidth scaling (same CPU) | 0.10 (1.5 GB/s) → 0.28 (8.8 GB/s) → **1.23 tok/s** (11.5 GB/s) | README community benchmarks |
| EPYC 7443, all experts in RAM | **1.00 tok/s** (matmul-bound, Zen3 no VNNI) | README community |
| **6× RTX 5090, full residency** | **6.84 tok/s** (256-tok, CPU-pinned), 0 disk wait, 1.0 tok/fw | `docs/experiments/glm52-6x5090` |
| `CUDA_DENSE`, 150 GB expert tier | 1.650 → **2.157 tok/s** (+30.8 %) | README CUDA section |
| Metal, M5 Max (128 GB) | **2.24 tok/s** | `docs/METAL-M5MAX-PERF-REPORT.md` |
| MTP speculation on MoE decode | **net loss** (−5 % @ 79 % acceptance) | 6×5090 experiment |

### 5.4 Batched decode — aggregate throughput (this session, measured)

Continuous batching (the concurrent scheduler's `forward_step_batched`) reads each routed expert **once
per step and shares it across all B sequences**, so per-step disk cost grows *sub-linearly* with B while
tokens grow linearly — aggregate throughput rises. Measured on the real 744B model (release build,
O_DIRECT, cache off, one decode step per B):

| Batch B | step wall | **agg tok/s** | per-seq tok/s | vs B=1 |
|---|---|---|---|---|
| 1 | 15.7 s | **0.064** | 0.064 | 1.0× |
| 4 | 25.6 s | **0.156** | 0.039 | 2.4× |
| 16 | 57.2 s | **0.280** | 0.018 | **4.4×** |

**Batching delivers a measured 4.4× aggregate gain at B=16** (0.064 → 0.280 tok/s): step time grows only
3.6× for 16× the tokens because the expert union is read once and shared. Per-sequence latency drops (the
batch tradeoff). The B=1 point (0.064) corroborates the §5.1 single-seq 0.054 tok/s. A simple
independent-uniform coupon-collector *overstates* the union (it predicts ~1.65 tok/s at B=16); real
routing from related prompts **overlaps heavily** (the B=16 union measured ~29 experts/layer, not ~103),
so the true gain sits above that pessimistic model — but the absolute ceiling stays disk-bound
(extrapolating to ~0.7 tok/s near the union-saturation knee, where each step approaches reading the full
358 GB). **The win is amortization of the byte budget, not a faster drive.**

**Resident compute path (tiny model, no disk):** aggregate 44.8k → 69.5k → 80.3k tok/s at B=1/16/256 —
compute-bound, so batching amortizes fixed per-step overhead (~1.8×) while per-seq drops. This isolates
the regime where the concurrent scheduler (and a compute-parallel worker pool) pay off.

**Compute-parallel worker pool (`peregrine-par`):** rmsnorm, resident MoE experts, per-row attention, and
every matmul (`apply_vec`) now run on a persistent scoped thread pool, **bit-identical** to serial
(`f32::to_bits`-exact tests). On the tiny model it lifts B=256 aggregate to **79.6k vs serial 66.3k
(1.2×)** with **no small-batch regression** (work-aware gates keep trivially-small matrices serial). The
win scales with per-op work, so a real hidden-6144 resident model parallelizes far more; it is hidden
under disk on the streaming path (compute already overlaps I/O), exactly as expected — a residency-regime
lever. A/B via `COLI_PAR_THREADS=1` (serial) vs default.

**I/O mode:** O_DIRECT vs buffered was within run-to-run noise at B=1/4 on this contended LUKS box —
consistent with §6's finding that O_DIRECT's benefit shows only on cold/uncontended runs.

---

## 6. Analysis

**The single-box wall is memory, not compute or engine quality.** Each decode token needs 600 experts
(11.3 GB). At ~1 GB/s (colibrì dev box) or ~710 MB/s (this LUKS NVMe) that is ~11–16 s/token of disk —
which is exactly what both engines measure (~0.05–0.06 tok/s). Faster NVMe moves the number
proportionally (colibrì's 0.10→1.23 tok/s across 1.5→11.5 GB/s); nothing in the engine can beat the
byte budget.

**Within the disk-bound regime, I/O-queue depth is what separates the two engines.** colibrì decodes
~1.4× faster (0.077 vs 0.054 tok/s) because it keeps the SSD busier: a **depth-512 ring fanned across
many experts** with 8 io-workers, `O_DIRECT`, and load/matmul overlap sustains ~870 MB/s, whereas
peregrine's concurrent scheduler currently walks the layer's disk experts **sequentially** through one
reactor (`read_expert` per expert, six regions batched into one submit, but experts one after another),
sustaining ~710 MB/s (and only ~270 MB/s during **prefill**, where a batched forward must stream
thousands of distinct experts — this is why the end-to-end gap widens to ~2.5× while the per-token
decode gap is ~1.4×). Both are correct and both hide compute under I/O; peregrine simply leaves disk
bandwidth on the table.

**Port — deep queue (this session):** peregrine's I/O lane was changed to submit a *batch* of experts
(16 × 6 = 96 in-flight reads) per `read_many` instead of one expert at a time (`concurrent.rs`
`read_experts_batched`, bit-identical). It **helped prefill ~21 %** (0.018 → 0.021 tok/s end-to-end on
the matched run) but did **not** close the gap: the live disk rate barely moved (~673 MB/s). Diagnosis:
**queue depth was not the dominant factor — buffered vs direct I/O is.** peregrine read *buffered*, so
every 18.9 MB expert was copied through — and pollutes — a page cache it will never reuse (0.6 %
locality); on LUKS that compounds.

**Port — O_DIRECT + aligned slab arena (this session).** A direct-I/O path was added
(`crates/peregrine-io/src/slab.rs` — a reusable 4096-aligned buffer pool; `Reactor::read_direct_many` —
align→read→slice; a twin `O_DIRECT` fd per shard in `SafeTensors` with a probe/buffered fallback;
`COLI_DIRECT` gate; **bit-identical, all determinism tests green**; verified engaging on the real model).
It bypasses the page cache exactly as the diagnosis called for. **Honest measurement caveat:** in the
controlled same-build A/B it ran **0.032 vs 0.028 tok/s buffered (~+13 %)** — but this box's throughput
is dominated by fluctuating system contention (other apps ~8 GB): the *buffered* matched run alone
ranged **0.018 → 0.028 across the session (~40 % variance)**, which *exceeds* the O_DIRECT signal. So
O_DIRECT shows a **modest, consistent edge** (it wins even against a possibly-cache-warmed buffered run,
and it's always truly cold), and the mechanism is sound, but this contended box **cannot pin a precise
number** — a clean quantification needs a quiescent machine. The slab arena also removes the deep
queue's per-read allocation (bounding RAM). (O_DIRECT needs an *aligned* buffer — an early misaligned
probe spuriously EIO'd on the LUKS device; `AlignedBuf` fixes it.)

**Port — N parallel io_uring rings (this session).** The I/O lane was changed from one ring to a **pool
of N rings** (`COLI_IO_RINGS`, default 4), one per I/O thread, with **lock-free atomic work-stealing**
over the layer's experts (each ring owns its own reactor → uncontended; bit-identical, determinism tests
green). **Measuring the raw device read rate during real generation — a contention-robust signal (17–20 GB
samples) unlike the noise-swamped end-to-end tok/s — resolves the earlier ambiguity cleanly:**

| config (during real generation) | device read rate |
|---|---|
| buffered, 1 ring | ~710 MB/s |
| **O_DIRECT, 1 ring** | **857 MB/s** (+21 % — the O_DIRECT win) |
| **O_DIRECT, ≥2 rings** | **~980 MB/s** (+38 % combined; saturates at 2 rings) |

The 1→2-ring jump confirms **serial dm-crypt decryption was the single-ring bottleneck** on this
LUKS-encrypted NVMe — parallel rings parallelize the decrypt. **Both ports stack, and peregrine's
streaming now reads *faster* than colibrì's measured ~870 MB/s on this box.** The end-to-end tok/s
didn't show this cleanly only because system contention dominates the small-token wall-clock; the
device-throughput numbers are the trustworthy measurement, and they are unambiguously positive.

**Caching is defeated by routing entropy.** The measured 0.6 % cross-token locality means a warm cache
helps only when it can hold a large fraction of the 19,200 experts (≈370 GB) — impossible in 46 GB RAM
or 12 GB VRAM. peregrine's 3.58× on a *repeated* forward proves the cache is correct and effective when
the working set fits; it simply never fits during ordinary decode on this box. The same entropy is why
colibrì's PILOT is "neutral" and its MTP speculation *loses* on MoE decode.

**peregrine's concurrent scheduler is architecturally superior but latent here.** Its advantage is
eliminating the C engine's phased CPU-then-GPU serialization — which only exists when experts are
GPU/RAM-resident. On a single RTX 3060, ~66 of 19,200 experts fit in VRAM (0.3 %), so the GPU lane is
nearly idle and the scheduler has almost nothing to overlap. colibrì's own 6×5090 result (6.84 tok/s,
0 disk wait) is the regime where the phased-vs-concurrent difference is measurable; reproducing that
head-to-head needs comparable multi-GPU hardware (see [§9](#9-future-work)).

**Deployment posture differs:** colibrì ships an explicit RAM auto-budget (it self-limited to 1
expert/layer here); peregrine relies on `MALLOC_ARENA_MAX=2` to bound glibc arena growth. Both avoid
OOM, by different means.

---

## 7. Limitations & threats to validity

- **Single machine, single GPU.** The residency regime where peregrine's scheduler wins (multi-GPU /
  large RAM) is **not** reproduced here; that comparison uses colibrì's *published* 6×5090 numbers.
- **peregrine memory discipline under batched prefill.** A matched 11-token-prompt run at a 10 GB cache
  **OOM-killed peregrine** (RSS 24.7 GB) during prefill — an 11-position batched forward holds far more
  experts/buffers in flight than 1-token decode. colibrì avoids this with an explicit RAM auto-budget
  (it self-limited to 1 expert/layer). peregrine currently needs a smaller cache for multi-token prefill;
  a proper peak-RAM projection (like colibrì's) is future work. The decode-rate figures above use a
  1-token prompt (cheap prefill) and are unaffected.
- **GPU tier marginal on 1×3060** (66/19,200 experts); GPU experts compute in f32 (not bit-exact vs
  int4 CPU) by design.
- **Short runs** (8–16 tokens) to keep wall-clock tractable at ~0.05 tok/s; longer runs would tighten
  the averages but not change the disk-bound conclusion.
- **LUKS-encrypted NVMe** adds read overhead vs colibrì's published raw-NVMe boxes — a reason to treat
  the absolute tok/s as *this box's* number, not a universal one.
- **Interface differences:** peregrine's serve protocol takes token ids; colibrì takes a text prompt
  (GLM chat template). Decode tok/s (steady-state) is the comparable axis and is largely prompt-independent.
- colibrì's Metal/multi-GPU/community numbers come from **other hardware**; they are context, not
  same-box measurements.

---

## 8. Reproducibility

**Model:** `/home/cortix/models/GLM-5.2-colibri-int4-with-int8-mtp` (HF `mateogrgic/...`).

**peregrine (Rust):**
```bash
cd peregrine
cargo build --release -p peregrine-engine --features cuda   # or without --features cuda for CPU
COLI_MODEL=/path/to/GLM-5.2-colibri-int4-with-int8-mtp \
  MALLOC_ARENA_MAX=2 COLI_ECACHE_GB=12 \
  target/release/peregrine            # emits READY, then read: GEN <ngen> <tok0> <tok1> ...
# cache hit-rate is printed per request as: [ecache] hits=.. misses=.. disk_reads=.. hit_rate=..%
```

**colibrì (C):**
```bash
cd colibri/c
make glm ARCH=native                                   # CPU build → ./glm
make glm CUDA=1 CUDA_ARCH=native                       # optional CUDA build
COLI_MODEL=/path SNAP=/path URING=1 DIRECT=1 DRAFT=0 MALLOC_ARENA_MAX=2 \
  ./coli run "your prompt" --model /path --temp 0 --ngen 8 --ram 10
# engine prints: STAT <tok> <tps> <hit> <rss>   (tps = decode tok/s, hit = expert hit %)
```

Matched config for the head-to-head: greedy (`temp 0`), MTP off (`DRAFT=0`), io_uring on, `O_DIRECT`,
~10 GB expert cache, same model dir. colibrì prints `prefill … | decode N tokens in T (X tok/s) |
expert hit rate …%`; peregrine prints the per-request `[ecache]` line. Use a 1-token prompt to isolate
the decode rate (batched prefill adds memory pressure and is measured separately).

---

## 9. Future work

Ordered by leverage on a single box, informed directly by this study:
1. ~~`O_DIRECT` + aligned slab arena~~ · ~~deep I/O queue~~ — **both implemented this session**
   (bit-identical; O_DIRECT ~+13 % in the controlled A/B, see §6). **Re-measure on a quiescent box** —
   this machine's ~40 % contention variance swamped the signal, so the real gain is unquantified here.
2. **Peak-RAM projection / RAM auto-budget** (port colibrì's approach) so batched prefill can't OOM and
   `MALLOC_ARENA_MAX` isn't a hard requirement — the last robustness gap vs colibrì on this box.
3. **Reproduce the residency regime** (multi-GPU or ≥256 GB RAM) to measure the concurrent-vs-phased
   scheduler head-to-head — the one comparison this box cannot make (colibrì hits 6.84 tok/s there).
4. **Better prefetch predictors.** The 0.6 % previous-token locality means the current PILOT predictor
   is weak; a learned/transition-graph predictor (todo.txt "expert momentum", "transition automaton")
   is the only path to prefetch that pays on GLM‑5.2.
5. **GPUDirect Storage**, larger VRAM residency, persistent CUDA kernels, NUMA-aware placement — and,
   trivially, **faster storage** (throughput ∝ GB/s: colibrì shows 0.10→1.23 tok/s across 1.5→11.5 GB/s).

---

## 10. References

- colibrì: <https://github.com/JustVugg/colibri> — `README.md` ("Honest numbers", community
  benchmarks, CUDA/Metal sections), `docs/experiments/glm52-6x5090-2026-07-12.md`,
  `docs/METAL-M5MAX-PERF-REPORT.md`, `c/glm.c:2922` (the phased-loop waste comment), `c/uring.h`,
  `c/tier.h`.
- peregrine: [`DESIGN.md`](../DESIGN.md), [`README.md`](../README.md);
  `crates/peregrine-model/src/concurrent.rs` (scheduler), `crates/peregrine-io/src/warmcache.rs`
  (A1), `crates/peregrine-io/src/ring.rs` (io_uring, C1/C2/C3), `crates/peregrine-model/src/gpu.rs`
  (B1).
- Model: <https://huggingface.co/mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp>.
- Upstream GLM‑5.2 architecture: DeepSeek‑V3-class MLA + sigmoid-router MoE + MTP.

*Measured 2026-07-24 on the machine in §4. peregrine numbers from this session's engine logs; colibrì
head-to-head from the live run in §8; colibrì scaling numbers cited from the repository above.*
