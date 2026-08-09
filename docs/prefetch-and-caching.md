[« Docs index](README.md)

# Prefetch, prediction & caching

Everything on this page is **correctness-neutral**: a wrong prediction just
re-streams identical bytes, and cache/eviction choices can never change the
token stream. That property is what lets the whole subsystem be aggressive.

## The router look-ahead

Everything in the next section predicts from *routing history* — what the router
answered on previous tokens. The router look-ahead (`model.rs`,
`LookaheadCtx::emit`) does something categorically different: at the end of layer
`L` it applies layer `L+1`'s own post-attention norm and router weights to layer
`L`'s output hidden state, and prefetches the experts that ranking names.

It costs one extra `E×D` matvec per layer against weights already resident
(under 1 % of a decode step at K3 shapes) and needs no stored artifact, no
warm-up and no format change. The router is kept at full precision by
[contract](../todo.md), which is what makes the ranking trustworthy.

**Why it is a different question from the one `predict.rs` answers.** The
authoritative router at layer `L+1` sees `rmsnorm(x + attn_{L+1},
post_ln_{L+1})`; the look-ahead sees `rmsnorm(x, post_ln_{L+1})`. The single
missing term is that layer's own attention delta, and the residual stream
dominates it. Measured by [WASTE](https://github.com/sqliteai/waste) on their K3
container over 1092 layer transitions (`LEARNED.md` §29/§34), recall@16 of the
next layer's actual routed set:

| predictor | recall@16 |
|---|---|
| same-layer expert ids | 1.7 % |
| co-occurrence, held out (≈ our `Automaton`) | 29.0 % |
| the previous token's set (≈ our `Momentum`) | 29.5 % |
| **next layer's router on this layer's hidden state** | **59.0 %** |

And it is *steeply ranked*, which is what makes a narrow window the right policy
rather than a compromise: 92.2 % precision at rank 1, 81.4 % cumulative at 6,
59.0 % at 16. So `COLI_ROUTER_LOOKAHEAD_N=6` is not "prefetch 16 and waste 41 %"
— it is roughly five useful reads and one wasted one per layer.

**Those are their numbers, on their model.** Nothing here was adopted on their
say-so. **One row has now been reproduced on a peregrine container** (2026-08-09,
`bench-data/2026-08-09-prefetch-causes/`): `peregrine route-stats` over a 24-token
real-text trace of GLM-5.2 int4 measures consecutive-token overlap at **33.55 %**
against an independence null of **3.12 %** — 10.7× the null, over ~1 725 layer
transitions. That is the "previous token's set" row (WASTE: 29.5 %), and it
reproduces.

The consequence is larger than the row. This repo carried the opposite as an
*inference* — that routing entropy was high enough to explain the 0.6 % warm-cache
hit rate, colibrì's neutral PILOT prefetch and MTP's net loss — while flagging in
`benchmarks.md` that the overlap had never actually been measured. **It has now,
and the entropy story is wrong**: routing is strongly predictable from the previous
token, so a low hit rate has to be explained by capacity or policy, not by the
router.

Still unreproduced: the router-lookahead row (59.0 %), which is the interesting one
and the only arm that cannot be scored from a trace — `route_ranks` needs the
hidden state, and a trace records only routed ids. `COLI_PREDICT_EVAL=1` is the
instrument for it, and **it is currently reachable only from the stdio `GEN` path**
— there is no hook in `forward_rows_inner`, so neither `peregrine bench` nor
`peregrine-serve` can produce the number, which is why it never has.

**It cannot change a token.** The authoritative router still runs at layer `L+1`
and still decides; the look-ahead only starts I/O. `router_lookahead_cannot_move_a_token`
pins a streamed decode (which has the look-ahead) bit-identical to a resident one
(which does not).

**Decode only, and that is measured.** A decode layer claims `k` cache slots and
leaves a real idle window — its attention — for the speculative reads to land in.
A prefill chunk claims the union over every position in the chunk, so the
speculative records are the freshest unpinned entries and are exactly what
eviction takes first. WASTE built the chunk-path version and measured the
signature of a prefetch thrown away and re-fetched: demand hit rate tripled,
total bytes read rose 6.9 %, wall clock did not move (§36). They removed the hook
rather than defaulting it off, and so have we. There is also no window to fill
there — a chunk layer's readers are busy continuously, so a speculative read does
not move a read into idle time, it moves it in front of another read.

Speculative reads are accounted separately from demand misses (`insert_prefetched`
→ `prefetch_used` / `prefetch_wasted`, never `misses`). Folding them together
would make a look-ahead that guessed wrong look like a cache that performed
badly, and would change the meaning of every hit-rate figure in the engine.

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
    onto the newest frame (routing phase shift → trust recency). "Heavy" is
    derived, not chosen: `predict::phase_boost(depth)` returns the full momentum
    scale `depth·(depth+1)/2`, which outranks any expert absent from the newest
    frame by construction. It shipped as a hardcoded `2` until 2026-08-08 — at
    depth 4 that merely *tied* an expert that had just dropped out, so the
    feature did approximately nothing while both its unit tests passed on a
    hand-built `boost: 100`. Opt-in via `COLI_PREDICT_SOURCE=phase-aware`.
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
  (`COLI_PREFETCH_LANES`, **default 1 — the pool is a single lane unless you
  raise it**). The lane is keyed on a sequence id assigned at admission, not on
  the sequence's index in the active set: that index slides down whenever an
  earlier sequence retires, which until 2026-08-08 migrated live streams between
  lanes mid-flight and split their queued reads across two rings.
  `peregrine-serve`'s `batch.rs` is the only caller that uses a lane other than
  0 — see [`COLI_PREFETCH_LANES`](configuration.md) for why the other emitters
  deliberately do not.
- **Verification** — opt-in `COLI_PREFETCH_VERIFY` re-reads and byte-compares
  every speculative load (a `verify_mismatch` counter, never a panic) and logs
  used/wasted/accuracy at shutdown.

### Reading the shutdown counters

```
[ecache]   hits= misses= disk_reads= prefetch_reads= hit_rate=
[ecache]   resident: N slots, X GB of Y GB budget (Z% full)
[prefetch] used= wasted= unclassified= accuracy=…(of C classified) yield=…(of R issued)
[prefetch] resident-unused: N slots, X GB of Y GB budget (Z%)
```

Four things that are easy to misread, all learned the hard way on 2026-08-09:

- **`accuracy` is not yield.** It is `used/(used+wasted)`, and `wasted` only
  increments **on eviction** — a prefetched slab still resident is in neither
  term. Quote `yield` (used per *issued*) alongside it or not at all; on the first
  serving-path run they read 21.9 % and 3.2 % for the same fetches.
- **`unclassified` mixes units.** It is `reads − used − wasted`, but
  `prefetch_reads` counts read *operations* (an expert re-predicted after eviction
  is read again) while `used`/`wasted` count slot *events*. Treat it as a loose
  upper bound. `resident-unused` is the unambiguous one.
- **`resident` tells you which failure you have.** A ~0 % hit rate with the cache
  **100 % full** is an eviction/ordering problem; the same hit rate with it near
  empty is an admission problem. Nothing else in the output distinguishes them,
  and both prior investigations guessed.
- **Neither line separates the two emitters.** `WarmCache` tracks one
  `from_prefetch` bool and both `PrefetchCtx::emit_layer` and `LookaheadCtx::emit`
  feed the same lane, so `used`/`wasted`/`yield` are a blend of the history
  predictor and the router look-ahead. Isolating them needs
  `COLI_ROUTER_LOOKAHEAD=0` as its own arm, or per-emitter tagging.

And one about the workload rather than the counters: **`hit_rate` is over all
lookups, including prefill**, which has no cross-token reuse by construction. A
short completion on a long prompt is mostly prefill, so its hit rate is capped far
below what the routing supports — a 12-token prompt with a 2-token completion caps
at ~3 % however well the cache behaves.

## The warm RAM cache (`peregrine-io/src/warmcache.rs`)

Budgeted by `COLI_ECACHE_GB` (default: 10 % of available RAM, capped at
2 GiB). Holds quantized expert bytes verbatim — a hit returns a byte-identical
slab.

**Size it against one decode token's working set.** That is
`sparse_layers × topk × bytes_per_expert` — on GLM-5.2 int4, 75 × 8 × 18.9 MB ≈
**11.3 GB**. Under LRU a slab has to survive a full cycle back to its own layer to
be reused on the next token, so below that figure cross-token reuse is
structurally impossible *however good the predictor is*, and above it reuse
appears immediately. Measured 2026-08-09 on a real container, prefetch off in both
arms so only the budget differed:

| budget | slots | hit rate | disk reads |
|---|---:|---:|---:|
| 4.29 GB | 227 | 1.9 % | 9751 |
| 12.88 GB | 681 | **5.7 %** | **9380** |

This was the only knob tested in that pass that reduced `disk_reads` at all — for
comparison, the prefetcher issued ~420 reads to save ~20. Multiply the figure by
the number of concurrent decode streams whose routed sets do not overlap, and read
it against the `COLI_ECACHE_GB` row in [configuration](configuration.md): more
cache is still not monotonically better, because past the point where it competes
with the resident trunk a hit becomes a page fault.

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
  Heat does not enter the victim score at the default.
- **LFRU tier scoring** (`peregrine-io/src/tier.rs`) computes
  `(heat << 8) | recency` with hysteresis. It was written, tested and
  unreachable — the `[R]` defect class in [BAD_PATTERNS](BAD_PATTERNS.md) — and
  this page once described it, in the present tense, as the policy in force.
  **Wired 2026-08-06 behind `COLI_CACHE_LFRU=1`**: `lfru_score` becomes the
  second component of the victim key and `decay` halves accumulated frequency
  every 4096 hits. Priority stays primary, so LFRU reorders *within* a
  protection class rather than overriding `COLI_PREFETCH_PROTECT`.

  **Frequency comes from the cache, not from `HeatTable`.** A `Slot` counts its
  own hits. Sourcing it from the model's heat table would have been the obvious
  wiring and the wrong one: that table is constructed only when a GPU tier is
  (`model.rs`), so on every CPU-only run — which is every run on a box without
  `COLI_GPU` — the policy would have silently degraded to the LRU it was meant
  to replace, while the knob read as enabled. A hit *is* a routing that found
  its expert resident, which is the same quantity, measured over exactly the
  slots the victim choice ranges over.

  Still unreachable, deliberately: `pick_lfru` and `pick_swap` are the
  **fixed-slot swap** form of the same policy — a pinned set of constant size,
  which is the GPU tier's shape (`reheat`), not a byte-budgeted cache's. They
  are a `reheat` question and are tracked there, not force-fitted here.
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

## Borrowed negative results

Measurements from two parallel projects solving the same problem — WASTE
([sqliteai/waste](https://github.com/sqliteai/waste), C, Apache-2.0) and deltafin
([gavamedia/deltafin](https://github.com/gavamedia/deltafin), Rust) — that close
off directions peregrine has open or has not yet tried. **None of them were taken
on faith as numbers**; they are recorded as hypotheses with evidence attached,
and each names the local instrument that would settle it here. They are kept
because a failed experiment someone else already paid for is worth as much as a
successful one, and is much easier to lose.

| Direction | Their finding | What it means here |
|---|---|---|
| **Per-expert bit allocation** — quantize rarely-routed experts harder | No lever. Measured per expert, per layer, per matrix on K3 *and* Kimi-Linear, the value of the third bit varies by only **1.01–1.15×** (WASTE §20). Activation frequency is heavy-tailed; *sensitivity* is not. The LP the literature proposes has nothing to optimize over | Do not build a heat-driven mixed-precision expert format. Heat is still a good residency and eviction signal — it is not a quantization signal |
| **Truncating the routed tail** — drop low-gate-weight experts | The router is flat. Rank 1 to rank 16 is a factor of **4.6**, and ranks 9–16 carry **33.3 %** of the gate mass (WASTE §23). No subset of layers behaves differently | Bounds how far `COLI_ROUTE_MIN_SHARE` can usefully go. Our own `COLI_GATE_STATS` measures exactly this locally — and already reports *no* tail on a flat router |
| **Statistical cross-layer prediction** | Co-occurrence from layer `L` scores 29.0 % recall@16 — indistinguishable from "reuse the previous token's set" (29.5 %), which the cache exploits for free (WASTE §29) | This is the class `Automaton`, `MacroTable` and `CoActivation` belong to. Their measured ceiling is roughly half the router look-ahead's. **`COLI_PREDICT_EVAL=1` is how to check whether ours are beating that baseline at all** — and whether `automaton.json` / `macrostates.json` are earning the pipeline that builds them |
| **Look-ahead on the prefill path** | Loses. Bytes read up 6.9 %, hit rate up 3×, wall clock flat (WASTE §36) | Why `COLI_ROUTER_LOOKAHEAD` is decode-only. Reopening it needs a *total bytes read* measurement, not a hit-rate one — the hit rate improves precisely while it is failing |
| **A bigger expert cache** | Non-monotonic and violently so: 17.3 GB → 0.63 tok/s, 23.3 GB → **0.07 tok/s**, with hit rate *rising* (36.2 → 38.4 %) and bytes read *falling* across the cliff (WASTE §16/§39). The engine was inside its budget and the machine was not | `cache_cliff_warning` now says this at load. The deeper point: the cache's own telemetry cannot see the cliff, so no amount of hit-rate instrumentation would have caught it |
| **Cache floor = one token's working set** | Was true, then stopped being the constraint — because the look-ahead only needs a record to survive from one *layer* to the next, not one *token* to the next. A cache far too small to hold a token's working set now runs within 10 % of one five times its size (WASTE §39) | A reason to re-measure `COLI_ECACHE_GB` sizing *after* enabling the look-ahead rather than before. The two interact |
| **Speculative decoding under streaming** | Pays for deltafin (which streams the spine, so drafts amortize a per-token read peregrine also has) and is refused by WASTE (which keeps the trunk resident, so there is nothing to amortize) | peregrine streams experts but keeps the trunk resident — between the two. Our MTP path (`COLI_DRAFT`) is measured on its own terms; this says which of their conclusions is the relevant analogy, not what ours will be |
| **`mlock`** | Does not raise the ceiling; removes variance. Wiring the *trunk* is what mattered, not the cache (WASTE §30–32) | Exactly what `COLI_MLOCK` does, and why it uses `MCL_CURRENT` rather than `MCL_FUTURE` |
