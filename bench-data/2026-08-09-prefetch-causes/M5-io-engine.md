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

## Follow-up: `iobench` was timing its own setup, and that turned out not to matter

Chasing the LUKS sector-size question (M1) surfaced that `iobench` started its
clock before spawning its rings, so `File::open`, `Reactor::new` and
`vec![0u8; blk]` — which reserves *and zeroes* `blkMB × iters × rings` bytes, the
same order as the bytes read — were all inside the measured window. That looked
like it would understate throughput, and `dd bs=1M iflag=direct` on the same file
reaching **1.5 GB/s** against `iobench`'s 0.83 seemed to confirm it.

**It did not.** The timer is now split, and `io` lands within ~2 % of `wall`
(2.54 vs 2.58 s) — setup was never a material part of the measurement, and the
historical figures are not an allocation artefact. Recorded because the hypothesis
was wrong and the check is what showed it; the split is kept so any future setup
cost is visible rather than charged to the device.

The `iobench`-vs-`dd` gap is real and is **access pattern**: 64 concurrent 32 MB
reads split into 128 KB block-layer requests (`max_hw_sectors_kb=128`)
oversubscribe a 255-deep queue, while `dd` is sequential at depth 1 with
readahead. Both are honest; they measure different things, and the streaming
lane's own pattern is much closer to `iobench`'s.

Re-measured after the fix, on fresh shards — the ranking is unchanged:

```
uring O_DIRECT  0.69 GB/s      uring buffered  0.84 GB/s      pread x8  0.85 GB/s
```

---

## Addendum 2026-08-09: both conclusions survive repeats; the harness had two defects

Everything above was measured **one pass per arm**, at **8 rings** where the
engine ships 4 (`COLI_IO_RINGS`, `model.rs::io_rings`). Both were wrong to do, and
`iobench` has been changed so neither is the default. Re-measured against a real
GLM-5.2 shard on the root LUKS/NVMe, 5 reps, at the shipped ring count:

| arm | median GB/s | spread | verdict |
|---|---|---|---|
| uring buffered | **1.12** | 5.4 % | — |
| pread ×8 | 1.06 | 10.6 % | **gap 5.7 %, inside the noise floor — indistinguishable** |
| uring O_DIRECT | 0.86 | 10.3 % | **−23 %, outside both spreads — a real regression** |

**Both headline conclusions hold, and now for stated reasons rather than
coincidence.** `pread` and `uring` are the same rate — not because 1.20 and 1.19
happened to land close on one pass each, but because their difference is smaller
than the noise floor, which is now measured rather than assumed. And O_DIRECT is
genuinely the slow arm: a 23 % gap that no overlap of the two distributions can
account for.

### Defect 1: one pass is not a measurement

Five identical passes against tmpfs returned 2.58 / 2.13 / 1.78 / 2.27 / 2.34 GB/s
— 35 % spread, on a box carrying ~36 % background CPU load. `iobench` now defaults
to `REPS=5`, reports median with min/max/spread, and prints an explicit note when
the spread exceeds 15 % of the median. Its `RINGS` default moved 1 → 4 to match the
shipped configuration.

### Defect 2: the offset walk silently benchmarks the page cache

Offsets are `(r * iters + i) * blk % flen`. When `rings × iters × blk` exceeds the
file, they **wrap**, and the later reads of a pass hit regions that same pass just
pulled into the page cache. Nothing said so. A 15-rep sweep on a 2.5 GiB shard read
like a clean 2× win:

```
rings=4   0.94 GB/s     rings=8   1.39 GB/s     rings=12  1.82 GB/s     rings=16  1.76 GB/s
```

At 12 rings that is 96 reads of 64 MB over **40 distinct slots** — 58 % of each
pass re-reading its own cache. Re-run at equal work with no wrap, the same sweep
is flat within noise (0.87 / 0.92 / 1.04 / 1.00), and the 2× disappears. `iobench`
now computes the distinct-slot count up front and warns, with the re-read
percentage, before it prints a single rate. **`COLI_IO_RINGS=4` needs no change.**

### The number that actually matters: the device is the constraint

The same harness against tmpfs — identical code path, no device — reaches
**3.77 GB/s** at 4 rings, against **1.12 GB/s** through the storage path.

**The engine has ~3.4× more headroom than the drive is delivering.** That reframes
where the next win is: not in the read path — which M5 has now twice failed to find
a lever in — but in the device under it. A PCIe 3.0 x4 NVMe saturating at ~3.4 GB/s
lands almost exactly on the engine's own ceiling, which is also why anything faster
would be wasted on this box (Ryzen 5 5500 is Cezanne; nothing negotiates PCIe 4.0).
See `[[cortix-box-profile]]` for the slot topology — the one unshared CPU-direct M.2
link currently holds a 128 GB Intel 600p.

**Caveat on the ceiling figure.** 3.77 GB/s is what tmpfs yields *with the VM and
VNC helper running*. It is a floor on the engine's capability, not its maximum —
quiescing that load would raise it, and that measurement has not been taken.
