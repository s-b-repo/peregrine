# Serving-path prefetch: first counters, and why the lane sweep did not run

Date: 2026-08-08 · Box: Ryzen 5 5500 (6C/12T), 46 GB RAM, RTX 3060 (unused — CPU
build), LUKS NVMe · Model: `GLM-5.2-colibri-int4-with-int8-mtp` (358 GB int4) ·
`COLI_ECACHE_GB=4`, `COLI_MEMO_ENTRIES=0`, CPU-only `peregrine-serve`.

## What this is

The plan was to sweep `COLI_PREFETCH_LANES` ∈ {1,2,4,8} over the serving path and
set an honest default. **The sweep did not run** — see the budget below. What did
come out of the attempt is worth more than a half-powered sweep would have been:
three defects that made the measurement impossible in the first place, now fixed,
plus the first numbers anyone has seen from the serving path's prefetch.

## The first serving-path counters

One request, `max_tokens=2`, cold-ish cache:

```
run 1 (original reporting):
[ecache]   hits=14 misses=3510 disk_reads=3510 prefetch_reads=433 hit_rate=0.4%
[prefetch] used=14 wasted=50 accuracy=21.9% fadvise=0 verify_mismatch=0

run 2 (same request, corrected reporting):
[ecache]   hits=10 misses=3514 disk_reads=3514 prefetch_reads=412 hit_rate=0.3%
[prefetch] used=10 wasted=67 unclassified=335 accuracy=13.0% (of 77 classified)
           yield=2.4% (of 412 issued) fadvise=0 verify_mismatch=0
```

**Two identical 2-token requests, and `accuracy` moved 21.9 % → 13.0 %.** The
yield moved far less (3.2 % → 2.4 %), which is what you would expect if the
classified subset is small and its composition is mostly an artefact of what
happened to be evicted before shutdown. Treat single-request runs as a smoke test
of the instrumentation, not as data.

**These lines had never been printable from `peregrine-serve`.** The stdio binary
has reported them since the feature landed; the HTTP server never did, so the one
path that has per-sequence prefetch — and therefore the only path where prefetch
*lanes* mean anything — was unobservable. Reporting now happens in the engine
thread, which is the only thing that owns the `Model`.

### Three things that line does not say, and one of them is a trap

**`accuracy=21.9%` has a survivorship-biased denominator.** It is
`used/(used+wasted)` = 14/64, and `prefetch_wasted` is incremented **only on
eviction** (`warmcache.rs:734,775`, gated on `from_prefetch && !ever_hit`). A
prefetched slab still resident at shutdown is neither used nor wasted, so it never
enters the ratio. 433 reads were issued and **64 were classified** — 85 % of the
evidence is missing. Measured as used-per-issued the effective yield is
14/433 = **3.2 %**. Neither number is wrong; quoting 21.9 % alone is. The shutdown
line now prints both plus the unclassified count, in both binaries.

This is not only a reporting problem: **`PrefetchTuner` (`COLI_PREFETCH_TUNE`)
EWMAs the same used/wasted pair**, so the distance controller is steering on the
classified 15 %. Whether that biases it is a real open question and is not
answered here — the tuner was deliberately left alone rather than re-based on a
hunch.

**`fadvise=0` is correct, not a bug.** The hint tier is off by default —
`hint_paths: env_usize("COLI_PREFETCH_HINT_PATHS", 0)` (`model.rs`). The counter
was honest and `docs/configuration.md` was wrong, listing both warm and hint paths
as "on" when they are counts and the hint one defaults to zero. Doc corrected.

**The cache is probably not the dominant cost here.** `docs/configuration.md`
records this engine's io_uring lane at **0.84 GB/s against colibrì's 2.02 GB/s**
from 8 blocking-pread threads on the same LUKS drive. At 3510 reads for 2 tokens,
a 2.4× read-throughput gap outweighs anything a 4 GB cache can do against a 358 GB
container. `COLI_IO_ENGINE=pread` exists precisely to test that dm-crypt
hypothesis and is pinned byte-identical across engines, so it can be A/B'd without
risking output. **That is the measurement to run before any further prefetch work
on this box** — see runbook §1a.

### Capacity vs. algorithm

4 GB of cache against a 358 GB container is a **1.1 % capacity ratio**, and a
0.4 % hit rate is close to what capacity alone predicts. Read this file with that
distinction in mind: nothing here shows the caching or prediction *algorithms*
underperforming, and a result that only says "this box needs more RAM" is not an
argument for changing them. The lane sweep, the tuner's evidence base, and the
io-engine gap are the parts that could be algorithmic; the hit rate on its own
is not.

## Why the sweep did not run

Measured on this box, both numbers first-hand:

| Configuration | Cost |
|---|---|
| 1 stream, 4 tokens | **3 min 23 s** (~50 s/token) |
| 8 concurrent streams, 4 tokens | **>50 min, killed unfinished** |

At 8 streams the load average reached **27** on a 12-thread box and I/O pressure
`some avg60=65 %` — the state the box profile records as putting plasmashell into
uninterruptible sleep. Eight distinct prompts route to eight distinct expert sets,
so the per-step union is far wider than one stream's; concurrency makes this
regime worse, not better, when the container does not fit in RAM.

A 4-arm × 3-repeat × 32-token sweep is therefore a multi-day run here, on a
machine someone is using. It was stopped deliberately, not abandoned.

**Shortening it was not an option.** The box swings ±45 % on ~50 s runs and only
settles to ±3 % past ~200 s, so a sweep small enough to fit could not separate
the arms — it would produce four numbers and no result, which is the outcome this
repo's own §1 notes warn is worse than none.

## What the default is, and why it did not change

`COLI_PREFETCH_LANES` stays at **1**. No measurement was taken, so no change is
justified; a default of 4 with a plausible story attached is exactly what the
audit that started this was cleaning up. The docs now state the real default and
what raising it costs (a thread and an io_uring ring per lane).

## To actually run it

`scripts/bench-serve-lanes.sh` + `scripts/bench-serve-lanes.py` exist and are
committed, and §9 of the validation runbook has the procedure and the budget.
Run them on a box that can hold more of the container resident. Two details in
the harness worth keeping if you rewrite it:

- It counts `usage.completion_tokens`, not SSE deltas. A token that only extends
  an unfinished multi-byte character emits no delta, and text held back as a
  possible partial `<tool_call>` marker emits none either — a first attempt
  measured **0 tokens from two healthy streams** for exactly that reason.
- It runs arms in **rotating order** across repeats, because there is no
  passwordless sudo here to drop the page cache between them.
