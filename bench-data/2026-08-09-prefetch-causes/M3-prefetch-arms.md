# M3 — what actually produces (and prevents) warm-cache hits

Date: 2026-08-09 · GLM-5.2 int4, 358 GB container · `peregrine-serve`, B=1,
one identical 2-token request per arm · `COLI_ECACHE_GB=4`, memo and route-stats
persistence off · raw output in `*.counters.txt` / `*.log`.

**All three arms emitted an identical completion**, checked rather than assumed —
these knobs are correctness-neutral and the harness verifies it.

| arm | hits | hit_rate | disk_reads | prefetch_reads | resident-unused | decode_s |
|---|---:|---:|---:|---:|---:|---:|
| `default` (idle box) | **19** | 0.3 % | 6325 | 425 | 0.32 GB (7.5 %) | 301 |
| `default` (contended †) | **30** | 0.5 % | 6314 | 423 | 0.25 GB (5.7 %) | 325 |
| `protect_off` | **9** | 0.1 % | 6335 | 401 | 0.55 GB (12.8 %) | 286 |
| `prefetch_off` | **0** | 0.0 % | 6344 | 0 | 0.00 GB | 227 |

† ran while a stale process from M2 was draining 93 GB of speculative reads (see
`unbounded-queue-blocks-shutdown.md`). It is kept in the table rather than
discarded because the two `default` runs together are the **only estimate of
run-to-run spread** available here, and it is large: 19 vs 30 hits on an
identical request. Read every single-run difference below against that.

Disk-read counts, by contrast, are stable to 0.5 % across all four runs
(6314–6344) — they are counts of cache misses, not rates, so they survive
contention. **The conclusions below rest on the read counts, not the hit counts.**

## 1. Every hit in this engine comes from prefetch

`prefetch_off` scores **0 hits out of 6344 lookups. Exactly zero.** The demand
path's own cache contributes no cross-token reuse whatsoever.

That is not a defect, it is arithmetic, and it is the arithmetic M2 set up. One
token routes 600 experts ≈ **11.3 GB**; the cache is **4 GB**. Under LRU a slab
admitted at layer *L* must survive a full 11.3 GB cycle to be hit at layer *L* of
the next token, and 4 GB cannot hold 11.3 GB. So the only hits available are the
ones prefetch lands *just in time* — warmed and consumed inside the same forward,
before the cycle evicts them.

This is the real shape of the "0.4 % hit rate". It is not a cache that is
performing badly; it is a cache doing nothing at all, with a prefetcher scoring a
handful of just-in-time saves on top of it.

## 2. The protection inversion was the leading hypothesis and it is refuted

Going in, the strongest candidate was that `Model::protect_from` gives
`prio ≥ 1` to the predictor's set — the same set the prefetcher warms — while a
demand-loaded slab that was actually routed but not predicted stays at `prio 0`
and is evicted first. A never-used speculative slab outranking a real one looks
wrong on its face.

**The prediction was that turning protection off would raise the hit rate. It did
not.** `protect_off` scored 9 hits, below both `default` runs (19 and 30), with
prefetch yield at 2.2 % against 4.5–7.1 %, and it more than doubled the *unused*
speculative footprint (0.32 → 0.55 GB) — i.e. without the predictor's ranking,
pure LRU retains whichever slab was warmed most recently rather than the one about
to be needed.

**How firmly: the hypothesis is unsupported, not the converse proven.** One run
per arm against a `default` spread of 19–30 does not establish that protection
*helps* by any particular factor; it establishes that removing it did not do the
predicted thing, and that the direction of every secondary counter (yield,
resident-unused) agrees. Calling this "protection helps 3×" would be reading a
single sample through a 58 % spread.

Worth stating plainly because it is the reason to measure before fixing: the
defect analysis was sound, the code does exactly what the analysis said, and the
effect still did not go the way the analysis implied.

## 3. Prefetch is net-negative on read volume, by 14×

Demand reads barely move across arms — 6314 / 6325 / 6335 / 6344, a spread of
0.5 %. Prefetch saves **19–30** demand reads and issues **~425** to do it.

```
default (idle)  6325 demand + 425 speculative = 6750 total
prefetch_off    6344 demand +   0 speculative = 6344 total
```

**Prefetch costs ~406 net extra reads (+6.4 %) and buys a 0.3 % reduction in
demand reads** — roughly **14–22 reads issued per read saved**. At ~18.9 MB per
expert that is ~8.0 GB of device traffic to avoid ~0.36 GB. Every arm reads the
same bytes to produce the same answer; the speculative ones are pure addition.

This is the one conclusion that does not depend on the noisy hit counts: even
taking the *best* observed prefetch result (30 saves) against the *same* ~425
issued, the trade is 14:1 against.

Wall clock agrees directionally — `prefetch_off` was fastest at 227 s against
301 s for an idle-box `default`, a 25 % difference — but with one run per arm on a
box whose `default` hit count moved 58 %, that is corroboration, not evidence.

## What this means for "improve the hit rate"

At `COLI_ECACHE_GB=4` there is **no policy change that helps**, and the two most
plausible ones are now measured: protection is already helping, and turning
prefetch down would take the hit rate to zero rather than improving it. The
binding constraint is that the cache cannot hold one token's working set, so
nothing survives to be reused, and M2 showed **33.55 %** of a token's experts
*are* reusable in principle.

That makes the next question a capacity question, not an algorithm question —
which is M4, and the threshold to cross is 11.3 GB.

## Two caveats on the counters themselves

- `unclassified` in the `[prefetch]` line is `reads − used − wasted`, which
  **mixes units**: `prefetch_reads` counts read *operations* (the same expert
  re-predicted after eviction is read again), while `used`/`wasted` count
  slot *events*. That is why 352 "unclassified" coexists with 13 resident-unused
  slots. Both numbers are right in their own unit; the subtraction is a loose
  upper bound, not a population.
- `resident-unused` is the one that is unambiguous, and it is small
  (5.7 % of budget on `default`). The speculative footprint is *not* crowding the
  cache out — which is a third way the pollution story fails to hold up here.
