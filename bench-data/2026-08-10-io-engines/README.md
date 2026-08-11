# IO engine A/B — harness fixed, numbers VOID

Date: 2026-08-10. **Do not quote the `.raw` files in this directory.** They were
taken on a box carrying a second concurrent workload and are not a measurement of
anything. They are kept only because the two failure modes they exposed are worth
recording.

## What the harness now does that it did not

`scripts/bench-io-engines.sh` exists because `iobench --reps` cannot answer the
engine question on its own: it re-reads **one** file, so rep 1 is a cold device
read and reps 2..N are served from the page cache. A buffered arm measured that
way is reporting RAM.

Two fixes, both needed:

1. **Cold reads, enforced.** Each rep gets its own shard *and* the shard is
   evicted first with `posix_fadvise(POSIX_FADV_DONTNEED)` — no root required.
   "Fresh shard" alone is not enough: it only means "not cached by this run", and
   a first attempt drew **1.41 GB/s** from a shard an earlier run had left
   resident against **0.86** for the same engine on a cold one.
2. **Undersized shards skipped.** `rings × iters × blk` must fit inside the shard
   or the offsets wrap and the pass re-reads its own cache. Shards here are **not
   uniform**: among 2.69 GB siblings there is one of 0.89 GB and one of **16
   bytes** (`out-00136.safetensors` — a valid safetensors file with an empty
   header and zero tensors, not corruption). The 0.89 GB one silently produced a
   wrapped, inflated sample; the 16-byte one produced `0.00 GB/s`.

## Why the numbers are void

Both passes ran while a **second session** was decoding the 358 GB container on
the same machine — load average 84, `some avg60=71 %` I/O pressure. The first pass
(page-cached, ~0.86 GB/s for both buffered arms) and the second (cold, 0.48–0.51)
differ by more than the effect being measured, and the second's 40–48 % spread is
larger than any engine gap worth reporting.

What they hint at, and what a clean run should test: **cold device reads land near
0.5 GB/s**, which is where the *serving path* already measures (397–518 MB/s from
`disk_reads × 18.9 MB`). If that holds on a quiet box it retires the claim that
the engine runs 1.6–2.1× below its own device — the gap would be against a
page-cache figure, not a device one.

## To take it properly

Quiesce the box first — no other decode running — then:

```bash
REPS=7 scripts/bench-io-engines.sh bench-data/<date>-io-engines
```

Read the two **buffered** arms against each other. The O_DIRECT arm is reported
but is not comparable to them: `COLI_IO_ENGINE=pread` silently disables O_DIRECT,
so uring-with against pread-without is a two-variable comparison.
