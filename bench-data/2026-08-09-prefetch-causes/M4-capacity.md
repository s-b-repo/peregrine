# M4 — capacity: tripling the cache did not move the hit rate

Date: 2026-08-09 · same request as M3 (12-token prompt, 2-token completion) ·
run under `systemd-run --user --scope -p MemoryMax=34G -p MemorySwapMax=0`, per
the box profile's rule for memory-hungry runs.

## The prediction, and what happened

One decode token routes 75 × 8 = 600 experts at ~18.9 MB = **11.3 GB**. Under LRU
a slab must survive a full 11.3 GB cycle to be reused at the same layer of the
next token. At `COLI_ECACHE_GB=4` that is impossible, so the prediction was that
crossing the threshold would turn cross-token reuse on.

`[ram]` projected peak 29.7 GB against 35.1 GB available, and it fit.

| arm | cache | hits | hit_rate | disk_reads | prefetch_reads | decode_s |
|---|---:|---:|---:|---:|---:|---:|
| `default` | 4.29 GB | 2 / 19 / 30 † | 0.0–0.5 % | 6325–6342 | ~423 | 239–325 |
| `bigcache` | 12.88 GB | 5 | 0.1 % | 6339 | 377 | 238 |
| `bigcache_noprefetch` | 12.88 GB | 5 | 0.1 % | 6339 | 0 | 303 / 382 |

† three runs of an identical request. The spread is the point: hit counts at this
scale are noise, and no difference smaller than ~30 hits means anything.

**Tripling the cache did not move the hit rate**, and the two 12.88 GB arms are
byte-identical (hits 5, misses 6339, disk_reads 6339) whether prefetch runs or
not. Capacity is refuted as a *sufficient* explanation, and prefetch is refuted as
the thing crowding the demand set out.

## Occupancy is what closed it

A near-zero hit rate has two opposite causes that hits/misses cannot separate: a
**full** cache is evicting the working set before reuse; an **empty** one is not
admitting at all. Neither binary reported occupancy, so both prior sessions
guessed. It is now printed:

```
4.29 GB  cache:  227 slots,  4.29 GB of  4.29 GB budget (100.0% full)
12.88 GB cache:  681 slots, 12.88 GB of 12.88 GB budget (100.0% full)
```

**Admission works and both caches are completely full.** 227 and 681 slots are
exactly 4.29 / 18.9 MB and 12.88 / 18.9 MB, so the accounting is sound and the
slabs are the expected size. Whatever is wrong is an *ordering* problem inside a
full cache, not a missing insert.

## The measurement's own flaw, which is most of the answer

This request is **81 % prefill**. The 12-token prompt's per-layer union is ~69
experts across 75 sparse layers ≈ 5 144 lookups; the two decode tokens contribute
~1 200. Prefill reads each expert in a chunk's union exactly once and has no
cross-token reuse to exploit, and two decode tokens offer exactly **one**
cross-token transition.

Even perfect reuse of M2's 33.55 % overlap would therefore have produced
~200 hits out of 6 344 lookups — a **3.2 % ceiling** on this workload. The
0.1–0.5 % measured is against that ceiling, not against 33.55 %.

So "the hit rate is 0.4 %" is substantially a statement about the request shape.
`M4-decode8/` repeats the 4 GB vs 12.88 GB comparison at 8 decode tokens with
prefetch held off in both arms, where decode lookups (~4 800) roughly match
prefill and seven transitions are available.

## Repeated at 8 decode tokens — and capacity is the lever after all

Same two cache sizes, prefetch held **off** in both so only the budget differs,
`max_tokens=8` so seven cross-token transitions are available and decode lookups
(~4 800) roughly match prefill (~5 144) instead of being swamped by it:

| cache | slots | hits | hit_rate | disk_reads | decode_s |
|---|---:|---:|---:|---:|---:|
| 4.29 GB | 227 | 193 | 1.9 % | 9751 | 356 |
| **12.88 GB** | **681** | **564** | **5.7 %** | **9380** | 353 |

**2.9× the hits, 3× the hit rate, and 371 fewer disk reads (−3.8 %).**

Two things make this the result the whole investigation was after:

1. **`disk_reads` finally moved.** Every configuration in M3 and in the 2-token
   M4 sat pinned at 6314–6344 — seven runs, one number, nothing changed the bytes.
   Cache capacity is the first knob tested that reduces them.
2. **The threshold behaves as predicted.** 681 slots is above one token's 600;
   227 is well below. At 227 only the tail of a token's set can survive to the
   next, at 681 the whole set can, and the hit rate tracks that.

Set against the alternative: **prefetch issues ~420 reads to save ~20**, while
going from 4.29 to 12.88 GB saves 371 reads and issues none. On this workload the
cache budget is worth roughly 18× what the prefetcher is, per read saved.

It is still only 5.7 % against a ~16 % ceiling for this workload (33.55 % overlap
× the 48 % of lookups that are decode), so capacity has not been *exhausted* at
12.88 GB either — one token's set fits, consecutive tokens' unions do not.

## What would still be worth knowing after that

If reuse stays near zero with decode dominating, the remaining suspect is eviction
order *within* a forward, and the cheap probe already exists but is not wired to
the serving path: `ecache_disk_reads_for_layer` gives a per-layer miss breakdown,
and the stdio binary prints a "busiest layers" line from it. A reuse failure
concentrated in early layers would look very different from one spread evenly.
