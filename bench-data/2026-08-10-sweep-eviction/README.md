# The eviction key, not the cache size

Date: 2026-08-10 · GLM-5.2 int4, 358 GB container · `peregrine-serve`, B=1,
8-token completion on a 12-token prompt · raw output in `*.counters.txt`.

Follows `bench-data/2026-08-09-prefetch-causes`, which concluded cache **capacity**
was the lever. A review pointed out that MoE expert access is a deterministic
front-to-back layer sweep and that LRU is pathological against exactly that — and,
separately, that my "prefetch off" arms were never clean LRU measurements, because
`Model::protect_from` runs even at `COLI_PREFETCH_WARM_PATHS=0` and puts
`pack_prio(predictor_score, heat)` into the eviction key.

The second point was right, and it is worth more than the first.

## Result: `COLI_PREFETCH_PROTECT` is on by default and is costing 40 % of the hits

All three arms below are 12.88 GB (681 slots, above one token's 600-expert working
set), prefetch fully off, identical request:

| arm | eviction key | hits | hit_rate | disk_reads | of available reuse |
|---|---|---:|---:|---:|---:|
| `bigcache_noprefetch` | `(pack_prio(score,heat), used)` | 564 | 5.7 % | 9380 | 40 % |
| `bigcache_bare` | `(0, used)` — pure LRU | **945** | **9.5 %** | **8999** | **67 %** |

**+68 % hits and 381 fewer disk reads, from unsetting a default-on knob.**
Available reuse over 7 transitions is `7 × 600 × 0.3355 ≈ 1409`, from the 33.55 %
consecutive-token overlap measured on 2026-08-09.

The mechanism is not subtle once stated: `protect_from` assigns `prio ≥ 1` to
every expert the *predictor* names, and `prio` is the **primary** ascending key in
the victim tuple (`warmcache.rs:795`). At a 1.1 % capacity ratio the predicted set
(~2500 keys) far exceeds the cache, so the priority term stops discriminating
between candidates and instead systematically outranks recency — the cache is
ordered by a speculative signal rather than by what the sweep actually touched.
Removing it does not make the cache smarter; it stops making it dumber.

## The sweep hypothesis: confirmed, in its strongest form

Running the same bare configuration at the **default** budget settles it. Pure
LRU, nothing else in the victim key, 227 slots against a 600-expert working set:

```
arm=bare  [ecache] hits=0 misses=9944 disk_reads=9944 hit_rate=0.0%
          [ecache] resident: 227 slots, 4.29 GB of 4.29 GB budget (100.0% full)
```

**Zero hits out of 9944 lookups**, with the cache 100 % full the entire time.
That is not "few"; it is the exact prediction of "LRU retains the complement of
what is needed next", and it is what a deterministic front-to-back layer sweep
does to a recency policy that cannot hold a whole pass.

The full matrix, all at 8 decode tokens, prefetch off:

| cache | slots (vs 600 needed) | `prio` in key | pure LRU |
|---|---|---:|---:|
| 4.29 GB | 227 — **38 %**, cannot hold a pass | 193 | **0** |
| 12.88 GB | 681 — **113 %**, holds a whole pass | 564 | **945** |

Read the two columns together and the whole picture falls out:

- **LRU is pathological precisely below the threshold and fine above it.** At 681
  slots the previous token's entire set fits, there is no complement to retain,
  and pure LRU captures 67 % of available reuse. At 227 slots it captures none.
- **The predictor priority is an accidental scan-resistance hack.** It *helps*
  below the threshold (193 vs 0) — any deviation from pure LRU beats LRU there,
  and the predicted set happens not to be the recency tail. It *hurts* above it
  (564 vs 945), where LRU is already doing the right thing and a speculative
  signal only displaces it. One default, opposite signs either side of a
  capacity threshold nobody was tracking.

So neither "capacity" nor "policy" alone was the answer: **which one binds depends
on whether the budget clears one token's working set**, and the engine currently
has no notion of that quantity.

### Sweep-aware eviction: implemented, and it under-delivers

`COLI_CACHE_SWEEP=1` evicts the **highest layer** instead of the least-recently-used
slot. Same arm, same budget, only the victim rule differs:

| 4.29 GB, prefetch off, protect off | hits | hit_rate | disk_reads | decode_s |
|---|---:|---:|---:|---:|
| pure LRU | 0 | 0.0 % | 9944 | 379 |
| `COLI_CACHE_SWEEP=1` | **71** | 0.7 % | 9873 | 350 |

It does what it was built to do — a structural zero becomes non-zero, with 71
fewer disk reads and 8 % less wall time — but the prediction was
`227/600 × 1409 ≈ 535`, and 71 is an eighth of that. **The prediction was wrong,
and the reason is admission, not eviction.**

Evicting the highest layer does not stop a high layer from being *admitted*. When
the sweep reaches layer 50, that expert is inserted, and `evict_to_budget` exempts
the just-inserted key from its own admission — so the victim is the highest
*other* resident, which is a band member. The layer-50 slab is then evicted by the
next insert, having already cost a slot the next pass needed. Every layer above
the band costs the band roughly one member per pass, and with ~50 such layers
against a 28-layer band, most of it is gone before the pass ends.

The band is not stable because nothing prevents doomed admissions. **The policy
that should work is admission control** — refuse to cache experts above a layer
threshold at all, so the band is never displaced — and that is an `insert_inner`
change, not a victim-order one. Recorded rather than attempted: it is a different
mechanism from the one this arm tested, and it deserves its own measurement.

Keep the result in proportion. The two larger effects here are still
`COLI_PREFETCH_PROTECT=0` (564 → 945 at 12.88 GB) and clearing the working-set
threshold (0 → 945). Sweep eviction is a third-order fix for the sub-threshold
case, and at 0.7 % it does not rescue a cache that is simply too small.

## Prefetch, at 8 tokens and the default budget

The last arm answers a question left open on 2026-08-09, where every prefetch-on
run was 2-token and showed demand hits of exactly zero:

| 4.29 GB, 8 tokens | hits | of which demand | disk_reads | speculative reads | decode_s |
|---|---:|---:|---:|---:|---:|
| prefetch off | 193 | 193 | 9751 | 0 | **356** |
| prefetch **on** | 196 | 183 | 9748 | **4034** | **502** |

**Prefetch buys 3 hits — inside noise — for 4034 extra reads and +41 % wall
time**, at a 0.3 % yield (13 used, 3641 wasted). It does *not* suppress demand
reuse: 183 of the 196 hits came from the demand path, so the earlier
`hits == prefetch_used` signature was an artefact of the 2-token workload, not a
property of prefetch.

## Standing caveats

- One run per arm. The 2026-08-09 pass established that hit counts on this box
  swing widely (`default` gave 2, 19 and 30 on three identical runs) — though at
  these magnitudes (564 vs 945) the gap is far outside that band.
- `disk_reads` counts **experts** (six regions each), not regions or device
  requests.
- Arms are correctness-neutral and the harness asserts identical completions
  across them.
