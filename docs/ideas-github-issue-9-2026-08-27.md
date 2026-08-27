[« Docs index](README.md)

# Issue #9 — "Path to 2 tok/s": triage against the tree (2026-08-27)

Issue #9 proposes a phased program toward ≥2 tok/s aggregate on this box, plus
three comment threads of research directions. This page maps every item to
what exists in the tree after the GLM-5.3-Flash landing
([deployment-glm53-flash.md](deployment-glm53-flash.md)), in the same
local-annotations format as
[issue #7](ideas-from-colibri.md)/[#8](ideas-github-issue-8-2026-08-23.md)'s
pages: **SHIPPED** (where), **PARTIAL** (what's missing), **OPEN** (not
started), **CLOSED** (measured against, or declined with a reason).

## The numbered program

| # | Item | Status |
|---|---|---|
| 1 | Byte ledger | **SHIPPED** — `ledger.rs` + `COLI_UNION_STATS=1`; the re-read-after-eviction column is now wired (was counted by `WarmCache::rereads_after_eviction` and printed "NOT MEASURED" — an `[R]`-class defect, closed 2026-08-27). Remaining gaps: columns are `slabs × bytes_per_expert`, not true bytes; no `late_prefetch` counter (`prefetch_stale_dropped` is drop-before-issue); layer granularity only for disk/prefetch reads. |
| 2 | Native GLM-5.3-Flash | **SHIPPED** — `Arch::Glm5Next`: KDA, NoPE MLA, k-pool DSA indexer, mHC hyper-connections, clamped SwiGLU, GLM-dialect MTP, FP8 import with stem rename. The issue's real question — bytes/token — is ~4.2 GB routed at int4 vs GLM-5.2's 11.3 GB. Measure with the ledger once the checkpoint lands (download ETA days at the link ceiling). |
| 3 | Four-SSD routing-aware placement | **SHIPPED** (pre-existing) — `peregrine-reshard` (bandwidth-proportional groups), `COLI_IO_DEVICE_SCHED` (default on), `COLI_SSD_AWARE_SCHED`, and the Louvain/spectral layout machinery in `peregrine-layout-reorg`. Works on Glm5Next unchanged because the importer renames to the `model.layers.` stem. Physical placement stays a manual post-step by design. |
| 4 | Selective VRAM residency | **PARTIAL** — the whole-expert machinery exists (`gpu.rs` heat knapsack, `reheat`, LFRU); **on Glm5Next the GPU expert tier is refused** until the CUDA kernels grow the SwiGLU clamp. Partial-hot / tile-level residency: OPEN (below). |
| 5 | Confidence-gated prefetch | **PARTIAL** (pre-existing) — entropy-adaptive breadth (`COLI_ENTROPY_ADAPT`, needs `COLI_PREFETCH_TUNE`), yield auto-disable, staleness gate, Warm-vs-Hint two-tier emission. Router look-ahead is **off under mHC streams** (the predictor wants the collapsed hidden, which does not exist mid-stack) — wiring a per-layer collapse for prediction is the open piece. Byte-denominated waste counters: open. |
| 6 | Activation-calibrated quantization | **PARTIAL** (pre-existing) — `COLI_CALIB_CAPTURE` + `quant_i3_g64_weighted` + the flip-rate gate. The Glm5Next contract already spends precision non-uniformly (int8 on gate-critical matrices). AWQ-class methods: open. |
| 7 | Aggressive RAM staging | **SHIPPED** (pre-existing) — warm cache, pinned staging, O_DIRECT slab pool, hugepages, NUMA probe. |
| 8 | Batch decoding | **SHIPPED** (pre-existing) — the batch engine + `forward_layer_batched_hc` extends it to Glm5Next (`glm5next_batched_decode_matches_per_sequence`). B=16 aggregate is the route to the 2 tok/s target. |
| 9 | What not to prioritize | **CLOSED** — matches the tree's own recorded negatives (roadmap.md, BAD_PATTERNS `[R]`). |

## The comment-thread directions (ranked ideas)

| Idea | Status |
|---|---|
| Expert surrogate + speculative verification (SPICE/S²-MoE) | **OPEN** — the enabling pieces exist (MTP verify batching, `COLI_SPEC_UNION_MAX` pricing drafts in expert-reads, `COLI_SPEC_CONF`); a resident low-rank surrogate lane does not. Note the tree already refutes the "MTP is a net loss" premise for peregrine (that was colibrì; `COLI_SPEC_CONF` inverted the regression to +37%). |
| Storage-aware routing | **CLOSED for production** — `COLI_ROUTE_MIN_SHARE` is the scar (12.5% reads for 27.9% top-1 flips); anything that biases routing is quality-gated research, not a default. |
| Cross-expert shared basis / residual storage | **PARTIAL** — `peregrine-basisfit` measures the rate–distortion offline; no runtime `B + Δ_e` reconstruction. The honest next step stays the offline verdict on real GLM-5.3 experts once the checkpoint lands. |
| Partial-hot expert tiles / progressive GEMM | **OPEN** — the issue's most interesting new mechanism and the largest lift: the read path is whole-expert-as-six-regions by design, and `reshard` deliberately preserves that. |
| Lossless entropy coding | **PARTIAL** (pre-existing) — zstd on-disk and in-RAM (`COLI_CACHE_COMPRESS`); ANS/tile coding open; the tree records why VRAM-side decompression does not pay here. |
| Expert XOR/delta coding, weight dictionary (AQLM-class) | **OPEN** — same offline-first discipline as the basis work. |
| ML cache replacement, cross-layer prediction | **PARTIAL** (pre-existing) — LFRU + co-activation + topic routing + the predictor ensemble; learned policies beyond that open. |
| CPU-vs-GPU per-expert cost model | **PARTIAL** — `LaneBalancer` heuristic; the full cost model is declined-by-choice in roadmap.md (timing-derived splits change logits). Moot on Glm5Next until the CUDA clamp lands. |
| Near-storage compute (CXL/NDP) | **CLOSED** — hardware this box does not have; scale-out-design.md territory. |

## What to measure first when the checkpoint arrives

1. `peregrine bench 1 16` with `COLI_UNION_STATS=1` — the real bytes/token
   ledger for this model (§1+§2 in one run).
2. `peregrine flip-rate` against an HF teacher-forcing dump — the int4/int8
   contract's quality gate.
3. Consecutive-token expert overlap (`dump-routes`) — decides how much of the
   GLM-5.2 cache economics transfers.
