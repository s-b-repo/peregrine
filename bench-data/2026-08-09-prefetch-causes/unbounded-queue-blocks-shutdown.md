# The unbounded prefetch queue makes shutdown drain reads nothing will use

Found 2026-08-09 while running M2, not looked for. Observed on the real
GLM-5.2 container, `peregrine dump-routes … 24 --text corpus.txt`, defaults
otherwise (`COLI_ROUTE_STATS_PERSIST=0`, so persistence is not the cause).

## What happened

| moment | cumulative `read_bytes` |
|---|---|
| `wrote 24 forwards of routing trace` printed | **492 GB** |
| ~10 min later, process still alive | **585 GB** |
| still climbing when killed | ~15 MB/s, competing with the next arm |

**93 GB of speculative reads were issued *after* the work was finished**, and the
process could not exit until they completed. It had to be `kill -9`d, at 10 GB
resident, while the next measurement was already running and the box had 5 GB
available and a full 16 GB swap.

## Why

`PrefetchMsg::Stop` is an ordinary message on the lane's **unbounded**
`crossbeam_channel` (`model.rs`). `Model::drop` → `PrefetchPool::stop()` sends
`Stop` and then **joins** the thread, so `Stop` is processed strictly *after*
every `Warm` already queued. Nothing anywhere drains or cancels the queue:
`PrefetchPool::barrier()` waits for it, and the worker loop's only early-out is a
`cache.contains(key)` dedup, which is not a throttle.

The backlog is large because breadth is unbounded — `COLI_PREFETCH_WARM_PATHS`
defaults to `usize::MAX`, so the `break` in `PrefetchCtx::emit_layer` is
unreachable and every candidate the predictor names is streamed. The serving
path's `enqueue_seq_prefetch` compounds it by looping **all 75 sparse layers in
one burst** per sequence per tick.

So the queue grows without bound during the run, and at shutdown the engine pays
for all of it at ~19 MB per item, sequentially — the worker submits one expert
(6 regions) per `submit_and_wait`.

## Why this is worth fixing regardless of what the prefetch arms say

Every other cost of over-broad prefetch is arguable: the reads *might* pay off,
the bandwidth *might* be spare. These do not. They are issued after the last
token, their results are dropped, and they hold a multi-gigabyte process alive.

It is also user-visible in the ordinary case. `Model::drop` runs on every clean
exit, so any operator stopping a server or a bench run waits out the backlog — and
on the serving path `Model::drop` is what writes `route_stats.json`, so this sits
directly in front of a persistence step.

## The fix, and the one thing to be careful about

`Stop` should take priority over queued `Warm`/`Hint` work: either a separate
control channel selected first, or an `AtomicBool` the worker checks each
iteration and which makes it drop the rest of the queue.

The care: `PrefetchPool::barrier()` exists so tests can wait for prefetch to
land, and it must keep meaning "everything queued has been processed". A shutdown
that discards work and a barrier that waits for it are different operations and
must not be collapsed into one — `prefetch_barrier`, `prefetch_from_history` and
the warm-cache tests depend on the waiting one.
