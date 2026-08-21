[« Docs index](README.md)

# Measurement discipline

How to get a number out of this engine that means something.

This page exists because the repo has been wrong four times in one direction: a
plausible measurement, taken once, that pointed at the wrong conclusion. Each time
the fix was not a better guess but a better instrument. If you are about to tune a
knob or publish a figure, read this first — every trap below has already cost a
day.

**The short version.** Take medians, not single runs. Check that your benchmark is
reading the device and not its own page cache. And when a counter is summed over
threads, never read it as wall time.

---

## 1. One run is not a measurement

Five *identical* passes of `iobench` against a tmpfs file — no device in the path
at all, nothing to vary but the scheduler — returned:

```
2.58   2.13   1.78   2.27   2.34  GB/s        median 2.27, spread 35 % of median
```

That is a **35 % spread with zero variables**. The cause is contention, not the
storage: this box runs an OSX-KVM guest and a VNC helper holding ~36 % of all
twelve threads before a benchmark starts.

Two published conclusions were drawn from single passes inside a distribution that
wide, and both were wrong:

- A ring-count sweep taken one-pass-per-point put the software ceiling at
  **2.99 GB/s**. Re-run with 5 reps per point, the same configuration measures
  **3.77 GB/s** — the original had simply landed low.
- `M1-storage-config.md` argued from a one-pass sweep that throughput "plateaus at
  4 rings and declines", and used the non-scaling as evidence *against* a
  CPU-bound crypto path. That curve reproduces on tmpfs with no dm-crypt and no
  device, so it was never evidence about crypto. [The argument is
  withdrawn](../bench-data/2026-08-09-prefetch-causes/M1-storage-config.md).

### What the tools do about it

`iobench` defaults to **`REPS=5`** and reports the distribution, not a number:

```
$ iobench shard.safetensors 64 8 4 0 uring 15
...
median 1.12 GB/s over 15 reps (min 1.11, max 1.17, spread 5.4% of median)
```

When the spread exceeds 15 % of the median it says so explicitly:

```
NOTE: spread is 25.8% of the median, so any A/B gap smaller than that is not
resolvable on this box right now. Quiesce background load or raise REPS before
concluding anything from a difference that size.
```

**Read that note as a hard gate.** A 6 % difference between two arms, against a
25 % spread, is not a small effect — it is *no measured effect*. Say "unresolved",
not "slightly better".

Between reps the file is cooled with `POSIX_FADV_DONTNEED`, so rep *N* is not
reading what rep *N−1* warmed. It is advisory and best-effort — a no-op on tmpfs,
irrelevant under O_DIRECT — and if it ever silently stopped working the reps would
converge and the reported spread would collapse, which is a visible symptom rather
than a hidden one.

---

## 2. The benchmark that measures its own cache

`iobench` walks offsets as `(ring * iters + i) * blk % flen`. That `% flen` is the
trap: when the working set exceeds the file, offsets **wrap**, and the back half of
a pass re-reads regions the front half just pulled into the page cache.

It looks exactly like a scaling win. A 15-rep sweep on a 2.5 GiB shard read:

```
rings=4  0.94 GB/s     rings=8  1.39     rings=12  1.82     rings=16  1.76
```

— an apparently clean **2× from ring count**. At 12 rings that is 96 reads of
64 MB over **40 distinct slots**: 58 % of each pass re-reading its own cache. Held
to equal work with no wrap, the same sweep is flat within noise:

```
4×24  0.87      8×12  0.92      12×8  1.04      16×6  1.00   GB/s
```

There was no 2×. The harness now computes the distinct-slot count before it prints
any rate:

```
WARNING: 96 reads of 64 MB over a 2.7 GB file is only 40 distinct slots — offsets
wrap, so ~58% of each pass re-reads what it just cached. This measures the page
cache, not the device.
```

**Rule:** keep `rings × iters × blk` **below** the file size, or shrink `BLK_MB`.
Against tmpfs the wrap is harmless (everything is RAM either way) — the warning
still fires, and there it can be ignored.

---

## 3. Thread-summed counters are not wall time

The `[lane]` counters (`io_us`, `cpu_us`, `gpu_us`, `reduce_us`) are accumulated by
every lane thread independently. **A sum of 17 s across four io rings is equally
"four rings busy for 4.3 s" and "one ring busy for 17 s."** On its own the counter
cannot tell you whether a lane was saturated or idle, which is precisely the
question you are usually asking.

`LaneTimings::lane_wall_us` is the denominator that resolves it — the wall clock of
the 3-lane region itself. With it, the shutdown report prints duty cycles:

```
[lane] 3 forwards: io 54.7s (73%) cpu 19.5s (26%) gpu 0.0s (0%) reduce 1.0s (1%)
[lane] moe wall 58.2s over 3 forwards (19.38s each); io duty 24% of 4 rings, cpu 0.3 workers busy
```

**`io duty 24% of 4 rings` is what found the claim bug.** It says: of the four
rings, on average 0.94 were doing anything. The percentages on the first line said
"I/O dominates at 73 %", which was true and useless — it is a share of *busy* time,
and three of the four rings were not busy at all. Only the duty cycle exposed it.
See [the concurrent scheduler](concurrent-scheduler.md#the-three-lanes).

**How to read the block:**

| line | means | watch for |
|---|---|---|
| `io/cpu/gpu/reduce` percentages | share of lane **busy** time | says nothing about idleness |
| `moe wall … each` | wall clock per forward inside the MoE lane | against a decode token's total: how much of a token is even in this lane |
| `io duty N% of R rings` | ring occupancy | **< ~50 % means the lane is starved, not the device** |
| `cpu N workers busy` | mean concurrently-active CPU workers | ≪ pool size means the io side is not feeding them |

---

## 4. Separate prefill from decode, always

`scripts/bench-prefetch-arms.sh` reports one `decode_s` covering the whole request.
On a short request that number is **mostly prefill**: at `max_tokens=4` a GLM-5.2
request is ~68 % prefill (5,144 expert lookups against 2,400). A change that only
affects decode — which is most cache and prefetch work — is diluted to invisibility.

`peregrine-gen` separates them:

```
── peregrine-gen ──────────────────────────────────────────────
  generated    4 tokens, 12 chars
  ttft         2m 29s  (prefill + first token)
  total        3m 42s
  decode rate  16.08 s/tok (0.062 tok/s)  (excludes ttft)
  inter-token  min 15.0s  p50 15.5s  p95 17.6s  max 17.6s
```

TTFT is excluded from the interval percentiles on purpose: prefill happens once,
and folding it in drags every percentile with it.

**The spread is a diagnostic, not noise.** On a disk-bound MoE lane a token served
from the warm cache and one that streams a full ~11.3 GB routed union differ by an
order of magnitude, so `min`/`p95` carry information the mean destroys.
`--json` writes every interval for offline analysis. See
[Tools](tools.md#peregrine-gen).

---

## 5. Checklist before publishing a number

- [ ] **Repeats**, and the spread reported beside the median.
- [ ] Gap between arms **larger than** the spread — otherwise report it unresolved.
- [ ] Working set **inside** the file; no wrap warning.
- [ ] Duty cycles checked, not just busy-time shares.
- [ ] Prefill separated from decode.
- [ ] **One variable at a time.** The `0.84` vs `2.02 GB/s` figure this repo
      published for a year compared uring-*with*-O_DIRECT against pread-*without*
      it — two variables — and the conclusion drawn from it was wrong. See
      [Benchmarks](benchmarks.md) and
      [`M5-io-engine.md`](../bench-data/2026-08-09-prefetch-causes/M5-io-engine.md).
- [ ] Long runs where possible. Identical configs swung ±45 % on ~50 s runs here;
      200 s+ runs average to ±3 %.
- [ ] Wrap memory-hungry runs so an OOM is contained rather than letting the
      killer pick off the VM:
      `systemd-run --user --scope -p MemoryMax=34G -p MemorySwapMax=0 …`

## Related pages

[Validation runbook](validation-runbook.md) · [Benchmarks](benchmarks.md) ·
[Performance tuning](performance-tuning.md) · [Tools](tools.md) ·
[The concurrent 3-lane scheduler](concurrent-scheduler.md)

## The byte ledger — 11.3 GB/token is not one number

`COLI_UNION_STATS=1` now prints a `[ledger]` block at shutdown decomposing the
figure this project quotes into the numbers it is actually made of. Measured on
the real GLM-5.2 int4 container, 15 tokens across B=1 and B=4:

| column | GB/token | |
|---|---:|---|
| **requested** | **11.349** | `Σ keff` — the arithmetic figure |
| **unique (union)** | 7.188 | 36.7 % removed by the batch union |
| cache-served | — | 3.9 % of unique |
| **from disk** | **6.905** | what the drive actually moved |
| prefetch waste | — | 12.8 % of disk traffic |

**The arithmetic figure is confirmed and it is not the disk traffic.** 11.349
GB/token matches what this repo has always quoted; the drive moved **6.905
GB/token**, 39 % less. Quoting the top row as if it were the bottom one
overstates disk traffic by 1.64×, and this documentation did exactly that.

### How the columns are defined, and why the denominators differ

- **requested** is routed selections × per-expert bytes. It double-counts every
  expert that more than one row in a batch selected.
- **unique** is the batch union: at B>1 one read serves every row that chose it.
  This is the byte identity underneath the batching claim, which had only ever
  been published as a throughput ratio.
- **prefetch waste** is a share of **disk traffic**, not of `requested`. The
  latter is ~1.6× larger and would report 7.8 % where the truth is 12.8 %. The
  flattering denominator is the wrong one.
- **re-read after eviction** prints `NOT MEASURED`. Distinguishing a re-read
  from a first read needs per-slab eviction history the warm cache does not
  retain, and those bytes sit inside `from disk` unlabelled. Named rather than
  folded in silently — a missing column that looks present is worse than one
  that is labelled absent.

### The rule the report enforces

Every saving printed is a **byte** figure, and the block refuses to present one
without naming the gate that has to qualify it. The precedent travels with it:
`COLI_ROUTE_MIN_SHARE` cut 12.5 % of reads and cost **27.9 %** of top-1
predictions. A byte saving with no quality figure beside it is not a result.

Proposed by `cwahq` (2026-08-21), who argued the ledger should precede the
factorization work. It should have: `peregrine-basisfit` prices against the
*requested* denominator, which this table shows overcounts by 1.58×.
