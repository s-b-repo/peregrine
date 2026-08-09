# M2 — does the router carry structure? Yes: 10.7× the independence null

Date: 2026-08-09 · GLM-5.2 int4 (256 experts/layer, top-8, 75 sparse layers) ·
24-token trace over **real text** · raw output in `M2-route-stats.txt`.

```
routed set size   8.0
overlap |A∩B|/|A| 0.3355  (33.55%)
jaccard           0.2212
independence null 0.0312  (3.12%)
```

**Consecutive tokens share a third of their routed experts, where independent
draws would share 3 %.** That is 10.7× the null over ~1 725 layer-transitions
(23 consecutive pairs × 75 sparse layers) — the same order of sample as the
1 092 transitions WASTE's numbers come from.

## Why this matters more than the ratio

This is the **first time routing overlap has been measured on a peregrine
container.** The repo has been carrying the opposite claim as an *inference*, and
said so honestly: `docs/benchmarks.md:38-44` attributes the 0.6 % warm-cache hit
rate, colibrì's neutral PILOT prefetch and MTP's net loss to routing entropy, then
flags "**Attributing all three to routing entropy is an inference, not a
measurement** — the routing overlap has never been measured. Run `peregrine
route-stats` … before relying on the entropy story." `todo.md` §13 repeats it:
caching cannot win at this capacity ratio "whatever the router does, and whether a
*better predictor* could is unmeasured until `route-stats` runs".

It has now run. **The entropy story is wrong.** Routing is strongly predictable
from the previous token, and 33.55 % is close enough to WASTE's 29.5 %
"previous token's set" baseline that their result reproduces here rather than
merely being borrowed.

So the ~0.4 % hit rate is **not** caused by unpredictable routing. Whatever is
causing it, a better predictor is not obviously the fix and a worse router is not
the excuse.

## Union growth — speculation and batching are cheaper than independent draws

```
w   consecutive   strided (B indep. seqs)   independent null
2       1.664            1.917                   1.969
3       2.229            2.704                   2.907
4       2.755            3.391                   3.816
6       3.720            4.575                   5.550
16      7.760            7.760                  12.745
```

A γ=1 speculative verify pays **1.66×** the bytes, not 2×. Batching 16 sequences
pays **7.76×**, not 12.7×. Both sit well below the null, i.e. the sharing that
continuous batching and speculation rely on is real and measurable — consistent
with the 4.4× aggregate gain at B=16 already recorded in `benchmarks.md`.

## The arithmetic this sets up

Overlap of 33.55 % is an **upper bound on what a one-step cache can hit**
(`peregrine-tools/src/lib.rs`: `mean_overlap` is documented as "the quantity a
perfect one-step cache would hit"). Reaching it requires a routed expert to
*survive* from token *t* to token *t+1*.

One token routes 75 × 8 = 600 experts ≈ **11.3 GB** at 18.9 MB each. Under LRU a
slab admitted at layer *L* of token *t* must survive a full 11.3 GB cycle to be
hit at layer *L* of token *t+1*. At `COLI_ECACHE_GB=4` the cache holds ~217
experts — **36 % of one token's working set** — so by the time the forward reaches
layer 75 everything from the early layers is gone, and the measured hit rate is
~0.3–0.4 % against an available 33.55 %.

That gap is what M3 and M4 exist to attribute: capacity alone predicts it, and so
does the protection inversion, and the two call for different fixes.

## Method note

The trace **must** come from `--text`. `dump-routes` used a synthetic corpus of
uniform-random token ids until 2026-08-09, and random ids route randomly: this
same report over a synthetic trace would show overlap ≈ the 3.12 % null and read
as proof that prediction is hopeless, when it would only be proof that the corpus
was noise. The subcommand now takes `--text` (shared with `flip-rate`, which had
learned this already) and warns loudly without it.
