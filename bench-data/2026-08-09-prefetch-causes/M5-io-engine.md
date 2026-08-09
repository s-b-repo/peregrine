# M5 — the io_uring/pread gap does not survive a controlled comparison

Date: 2026-08-09 · `cargo run --release -p peregrine-io --example iobench` ·
64 MB blocks, 8 deep, 8 rings (4.2 GB per arm) · **a different shard per arm**, so
no arm reads what a previous one warmed.

```
io_uring O_DIRECT  x8 rings: 4.2 GB in 5.19s -> 0.81 GB/s
io_uring buffered  x8 rings: 4.2 GB in 3.57s -> 1.19 GB/s
pread x8 threads   x8 rings: 4.2 GB in 3.51s -> 1.20 GB/s
```

## The gap was a confound

`docs/configuration.md` motivates `COLI_IO_ENGINE=pread` with "peregrine's
io_uring lane measured 0.84 GB/s against colibrì's 2.02 GB/s from 8 blocking-pread
threads on the same LUKS drive", and `bench-data/2026-08-08-serve-prefetch`
repeated it as the biggest lever on the table.

**`pread` implies no O_DIRECT** (`concurrent.rs`: the direct path is skipped for
that engine). So the recorded pair compares io_uring **with** O_DIRECT against
pread **without** it — two variables, which is the exact confound
`validation-runbook.md` §1a warns about and which the one attempted reproduction
already recorded as inconclusive.

Comparing like with like, **`pread` and `io_uring` are the same rate**: 1.20 vs
1.19 GB/s, a 1 % difference on single runs. The syscall shape is not costing
anything on this box.

What the numbers do show is that **O_DIRECT is the slow arm** — 0.81 GB/s against
1.19 buffered, a 32 % penalty — which is consistent with `COLI_DIRECT` already
defaulting to off and with its documented note that direct I/O regresses on
page-cache-warm runs. It also reproduces the historical 0.84 GB/s figure almost
exactly (0.81), on different hardware, which suggests that number was always
measuring O_DIRECT rather than "the io_uring lane".

## Consequence

The read path is not the lever, and `COLI_IO_ENGINE=pread` should not be expected
to buy anything. The dm-crypt hypothesis it was built to test is not supported by
this measurement — at 1.2 GB/s through LUKS on 8 threads, decryption is not the
binding constraint at these block sizes either.

**Caveats.** Single run per arm; the two buffered arms benefit from kernel
readahead that O_DIRECT by definition does not, which is part of why O_DIRECT
loses and is not a defect in the comparison — it is the comparison. Sequential
64 MB reads are not the engine's actual access pattern (6 regions per expert,
~6 MB and ~8–24 KB each), so this bounds the *device*, not the streaming lane.
