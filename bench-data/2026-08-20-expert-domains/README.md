# Do different domains route to different experts? — 2026-08-21

**Verdict: no, not separably by `TokenClass`. The expert map is global heat.**
Phase 1 of the expert-mapping plan was a gate; it does not clear, so Phases 2
(`expert_map.json`) and 3 (output-neutral consumers) were **not built**. Recorded
as a closed negative.

The one apparent signal — everything involving `json` — tracks how
*self-similar the json document is*, not what domain it belongs to.

## What was measured

Four corpora, one `dump-routes --weights` trace each on GLM-5.2, N=256, same
model, same flags, same binary. Only the corpus differs.

| domain | bytes | `TokenClass` | wall |
|---|---|---|---|
| prose | 4 327 | **Prose** | 4 435 s |
| code | 7 303 | **Code** | 3 676 s |
| json | 6 000 | **Json** | 4 483 s |
| techdoc | 4 640 | **Code** | 3 555 s |

Each trace is split into 4 contiguous chunks; the statistic is the Jaccard
overlap of each chunk-pair's top-K routed `(layer, expert)` slots, compared
`within` label against `across` label, against a label-permuted null (500 reps).

`peregrine expert-map prose=… code=… json=… techdoc=…` reproduces the table.

## Why the obvious reading is wrong

Every pair beats the label-permuted null at **z ≈ +3 to +4**. That number means
nothing on its own, and the tool says so in its own output. Routing has **34.7 %
consecutive-position overlap against a 3.1 % independence null**, so *any*
contiguous grouping beats a shuffle — including contiguous halves of a **single**
prose trace, which read z = +11 to +19 with no domain effect present at all.
That confound is why this measurement was rebuilt once already.

The right comparator is a **same-`TokenClass`, different-document** pair, which
isolates "different document" from "different domain".

**`techdoc` classifies as `Code`, not `Prose`** — it is IT/cyber technical
documentation, dense with paths, flags and identifiers. That makes
`code vs techdoc` the control, and it makes `prose vs techdoc` a genuine
cross-class pair.

## The result

Excess over the `code vs techdoc` control (σ = the pair's own null spread):

| pair | classes | K=100 | K=600 | K=2400 |
|---|---|---|---|---|
| code vs json | Code/Json | +0.101 (+1.3σ) | +0.090 (+1.3σ) | +0.083 (+1.5σ) |
| json vs prose | Json/Prose | +0.074 (+1.1σ) | +0.114 (+1.7σ) | +0.129 (+2.1σ) |
| json vs techdoc | Json/Code | +0.063 (+0.9σ) | +0.072 (+1.1σ) | +0.086 (+1.6σ) |
| code vs prose | Code/Prose | +0.020 (+0.4σ) | +0.037 (+0.8σ) | +0.040 (+1.0σ) |
| **prose vs techdoc** | **Prose/Code** | **−0.045 (−1.0σ)** | **−0.011 (−0.3σ)** | **+0.003 (+0.1σ)** |
| *code vs techdoc* | *Code/Code* | *— control —* | *— control —* | *— control —* |

### The result that does not depend on picking a control

Subtracting a control is only as good as the claim that the control pair shares a
domain — and "Rust source" versus "IT documentation" is arguable even though both
classify as `Code`. So here is the same data with no control at all, sorted:

| K | non-json pairs, ascending | json pairs |
|---|---|---|
| 100 | 0.1304 `prose·techdoc` **(cross-class)** · 0.1754 `code·techdoc` *(same-class)* · 0.1954 `code·prose` **(cross-class)** | 0.2388 – 0.2766 |
| 600 | 0.1412 `prose·techdoc` **(cross)** · 0.1517 `code·techdoc` *(same)* · 0.1891 `code·prose` **(cross)** | 0.2236 – 0.2661 |
| 2400 | 0.1192 `code·techdoc` *(same)* · 0.1218 `prose·techdoc` **(cross)** · 0.1588 `code·prose` **(cross)** | 0.2022 – 0.2485 |

**Class membership does not order the gaps.** At K=100 and K=600 the one
same-class pair sits *in the middle of* the cross-class pairs; at K=2400 it is the
lowest. Crossing a `TokenClass` boundary does not move the number.

**"Involves json" does.** Every json pair is disjointly above every non-json pair
at all three core sizes — no overlap anywhere. The partition that explains this
table is not *same domain vs different domain*; it is *json vs not-json*. And
json is the most self-similar document (below).

Two further things kill the domain hypothesis:

1. **`prose vs techdoc` — different domain *and* different class — separates no
   more than the same-class control**, and at K=100 separates *less*. If domain
   drove expert selection this is the pair that should show it most.
2. **One cell out of fifteen reaches 2σ** (json vs prose, K=2400, +2.1σ), with no
   correction for fifteen comparisons. That is what chance produces.

## Where the "json effect" comes from

The pooled `within` values decompose exactly into per-document self-similarity
(6 equations, 4 unknowns, **max residual 0.0000**):

| domain | K=100 | K=600 | K=2400 |
|---|---|---|---|
| **json** | **0.395** | **0.412** | **0.446** |
| code | 0.369 | 0.360 | 0.336 |
| prose | 0.285 | 0.335 | 0.333 |
| techdoc | 0.281 | 0.277 | 0.289 |

`json` is the most internally repetitive document at every core size — a JSON
file repeats its own structure — and it is the *only* document whose
self-similarity **rises** with K. Every pair showing excess over the control
involves `json`, and the size of the excess tracks this column rather than any
notion of domain. That is a document-homogeneity confound, not specialization.

## What this does and does not close

**Closed:** a `TokenClass`-conditioned expert map has nothing in it to act on.
`TopicProfiles` accumulating per class (Phase 0a) is still worth having — it
repairs a counter that recorded nothing under `peregrine-serve` — but nothing
downstream should be built on the assumption that its five buckets separate
experts.

**Not closed, and worth saying plainly:** this tests *one* key. `TokenClass` is a
5-bucket character-ratio heuristic over 512 chars — the coarsest possible domain
label, and coarse enough that a technical-documentation corpus lands in `Code`.
A null under this key does **not** show experts are unspecialised; it shows they
are not separable *by this key*. A semantic key (embedding-clustered documents,
or task labels from a served workload) is a different experiment and remains
open.

**Also unchanged:** the hot core is real and already exploited. ~600 slots
explain the entire stable long-range structure, 335 slots cover 55.8 % of a
token's picks, and residency/prefetch already rank on exactly that. What this
measurement rules out is *conditioning that ranking on content class*.

## Files

- `corpora/{prose,code,json,techdoc}.txt` — inputs
- `routes-*.json` — the four traces (gate weights included)
- `dump-*.log` — per-trace engine logs
- `trace.sh` — regenerates the traces; `trace.out` is its log
