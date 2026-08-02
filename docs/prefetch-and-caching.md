[« Docs index](README.md)

# Prefetch, prediction & caching

Everything on this page is **correctness-neutral**: a wrong prediction just
re-streams identical bytes, and cache/eviction choices can never change the
token stream. That property is what lets the whole subsystem be aggressive.

## The prediction spine (`predict.rs`)

- **`RouteHistory`** — a K-deep window of recent routing frames
  (`COLI_ROUTE_HIST_DEPTH`, default 4). Depth 1 reproduces the legacy
  single-frame behavior.
- **`PredictSource`** — pluggable predictors over that history:
  - **Momentum** — recency-weighted vote across the K frames.
  - **Automaton** — an offline global expert-transition FSA
    (`automaton.json`, built by `peregrine build-automaton` or the `galactic`
    pass; config-tagged, auto-loaded at model construction) blended with
    momentum.
  - **PhaseAware** — wraps any inner source; when the Jaccard distance between
    the two newest frames exceeds `COLI_PHASE_THRESHOLD`, folds a heavy vote
    onto the newest frame (routing phase shift → trust recency).
  - **WithMacro** — blends `MacroTable` macro-states: consecutive identical
    top-k sets collapse into dwell-counted states with state→state
    transitions (`macrostates.json`, built by `galactic`).
- **`PrefetchTuner`** — EWMA over prefetch used/wasted adapts the warm breadth
  (`COLI_PREFETCH_TUNE` / `_DIST` / `_DIST_MAX`).
- **`CoActivation`** — long-term pair co-firing tracker, persisted in
  `route_stats.json`; feeds [dispatch-order fusion](concurrent-scheduler.md#dispatch-order-shaping).

## Prefetch emission

- **Layer look-ahead** — `PrefetchCtx::emit_layer` runs from the
  `forward_hidden` loop, staggered ahead of the compute cursor
  (`COLI_PREFETCH_LOOKAHEAD`) instead of one bulk dump.
- **Two-tier speculation** — top-N ranked candidates split into a **warm tier**
  (actually read into the warm cache; breadth `COLI_PREFETCH_WARM_PATHS`) and
  a **hint tier** (`fadvise WILLNEED` page-cache warming only; breadth
  `COLI_PREFETCH_HINT_PATHS`, skipped under O_DIRECT). Per-workload-class
  overrides: `COLI_PREFETCH_WARM_PATHS_<CODE|JSON|MATH|PROSE|MIXED>` and the
  `_HINT_PATHS_` twins.
- **Per-sequence prefetch in batched serving** — each concurrent stream
  predicts from its own routing history on a parallel prefetch-lane pool
  (`COLI_PREFETCH_LANES`).
- **Verification** — opt-in `COLI_PREFETCH_VERIFY` re-reads and byte-compares
  every speculative load (a `verify_mismatch` counter, never a panic) and logs
  used/wasted/accuracy at shutdown.

## The warm RAM cache (`peregrine-io/src/warmcache.rs`)

Budgeted by `COLI_ECACHE_GB` (default: 10 % of available RAM, capped at
2 GiB). Holds quantized expert bytes verbatim — a hit returns a byte-identical
slab.

| Feature | Gate | What it does |
|---|---|---|
| Bloom filter | always on | 2048-bit, two hashes, short-circuits the miss path in `WarmCache::get`; rebuilt on eviction so the hint stays tight |
| Transparent zstd | `COLI_CACHE_COMPRESS=1` | compress slabs on admit (~1.2× smaller resident footprint — measured), decode on hit |
| Idle recompression | `COLI_CACHE_COMPRESS_IDLE=1` | the serve engine converts the coldest raw slot to zstd per idle tick, interruptible the moment a request arrives |
| Negative TTL | `COLI_CACHE_NEGATIVE_TTL=<N>` | evict never-hit slots older than N clock ticks ahead of LRU order (unprotected slots only; always keeps at least one) |
| Admission gate | `COLI_CACHE_ADMIT_MIN_HEAT=<N>` | admit an expert only once its routing heat reaches N (heat bumps post-reduce, so `1` = "cache from the second routing on", filtering one-off experts; `0` = admit all) |
| Predictive protection | `COLI_PREFETCH_PROTECT` | predictor ∪ hot experts get an opaque cache priority; the victim order is `(priority, recency)` — all-equal degenerates to pure LRU |

## Tiering & GPU residency

- **Warm-cache eviction** (`peregrine-io/src/warmcache.rs::evict_to_budget`):
  the victim is the lowest `(priority, recency)` — a **priority-weighted LRU**.
  Heat does not enter the victim score.
- **LFRU tier scoring** (`peregrine-io/src/tier.rs`) computes
  `(heat << 8) | recency` with hysteresis, and **nothing calls it.** It is
  written, tested, and unreachable — the `[R]` defect class in
  [BAD_PATTERNS](BAD_PATTERNS.md). This page previously described it, in the
  present tense, as the policy in force; it never has been. Wire it into
  `evict_to_budget` or delete the module — but do not read it as live.
- **Heat table** (`gpu.rs::HeatTable`): lock-free atomic routing-frequency
  counters, the substrate every residency/admission decision reads.
- **Dynamic VRAM residency**: `reheat()` re-selects the hottest experts every
  256 steps; initial placement is a greedy heat/bytes knapsack
  (`solve_residency_sized`) with deterministic ties, seeded from the previous
  session's `route_stats.json` heat when one is present.
- **Replication**: `COLI_REPLICATE_K` warms the top-K hottest GPU residents
  into the RAM cache as well, so a lane-balancer downgrade costs no disk read.

## Offline artifacts that feed prediction

Built once, consumed automatically at load (all config-tagged):

| File | Producer | Consumer effect |
|---|---|---|
| `automaton.json` | `peregrine build-automaton` / `galactic` | transition-FSA predictor blended with momentum |
| `macrostates.json` | `galactic` | macro-state predictor (`PredictSource::WithMacro`) |
| `route_stats.json` | saved at `Drop` (gate `COLI_ROUTE_STATS_PERSIST`) | warm-start history, heat, co-activation, learned policy |
| `schedule.json` / `tiers.json` | [`peregrine-layout-reorg`](layout-tools.md) / `galactic` | disk-order read sorting; RAM-tier prefetch seeding (`COLI_TIER_SEED`) |

See [Model format & artifacts](model-format.md) for file shapes and
[Configuration](configuration.md) for the full knob table.
