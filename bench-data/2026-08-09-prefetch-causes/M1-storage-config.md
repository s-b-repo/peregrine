# M1 — storage configuration (runbook §1b)

Date: 2026-08-09 · Box: Ryzen 5 5500, 46 GB RAM, model on `/dev/mapper/luks-3208…`
(LUKS2 → ext4 → `nvme0n1p2`).

Checked first because §1b records the largest unverified effect on the table: a
512-byte LUKS sector measured ~10 % of raw throughput where 4096 restored ~5×,
and an I/O scheduler costs 14–57 % versus `none` on NVMe. If either were wrong
here, every other number in this directory was taken against a crippled device.

## Result: the two tunable items are already optimal; one is unresolved

| Item | Value | Verdict |
|---|---|---|
| `nvme0n1` scheduler | `[none]` mq-deadline kyber bfq | **optimal** — nothing to win |
| `nvme1n1` scheduler | `[none]` mq-deadline kyber bfq | optimal |
| `nvme0n1` `max_hw_sectors_kb` | 128 | **hardware limit, not tunable** — `max_sectors_kb` cannot exceed it |
| `nvme0n1` `max_sectors_kb` | 128 | at the hardware ceiling |
| `dm-0` (LUKS) `logical_block_size` | 512 | suggestive, **not conclusive** — see below |
| `dm-0` `max_sectors_kb` | 128 | inherited from the backing device |
| LUKS **encryption sector size** | **unknown** | needs root |

## Confirmed: the LUKS sector size is 512 bytes

```
$ sudo cryptsetup luksDump /dev/nvme0n1p2 | grep -i sector
        sector: 512 [bytes]
```

So §1b's *precondition* holds. Whether its recorded ~5× applies here is a separate
question, and the evidence gathered so far is genuinely mixed:

| observation | reading |
|---|---|
| `dd bs=1M iflag=direct` through LUKS: **1.5 GB/s** | not obviously crippled |
| O_DIRECT throughput vs ring count: 0.60 / 0.81 / 0.86 / 0.83 / 0.69 GB/s at 1 / 2 / 4 / 8 / 12 rings | **plateaus at 4 rings and declines** — does not behave like a CPU-bound crypto path, which would keep scaling on 12 threads |
| CPU during `dd` at 1.5 GB/s | ~4.4 of 12 cores busy, against a 1.9-core desktop baseline → **~2.5 cores net**, i.e. ~1.7 cores per GB/s |

~1.7 cores per GB/s is high for AES-NI (which should do ~1–2 GB/s per core) and is
the kind of overhead 8× more crypto operations per byte would produce — but it is
not conclusive, and the non-scaling argues the other way.

**The reformat is not a knob.** `cryptsetup` cannot change sector size in place;
it means backing up, `luksFormat --sector-size 4096`, and restoring. This volume
is 878 GB used of 906 GB with 22 GB free and holds the OS, and `/mnt/backup` is a
696 GB array — smaller than the data. That is a multi-hour, whole-system operation
with nowhere obvious to stage it.

**Do not reformat on the evidence above.** The measurement that would settle it is
cheap, read-only, and needs one sudo:

```bash
sudo dd if=/dev/nvme0n1p2 of=/dev/null bs=1M count=2000 iflag=direct
```

That reads the *encrypted* partition directly, bypassing dm-crypt entirely.
Against the 1.5 GB/s measured through the mapper:

- **raw ≈ 1.5–1.7 GB/s** → dm-crypt costs nothing, sector size is irrelevant here,
  close the item.
- **raw ≈ 3 GB/s or more** → the crypto layer is halving throughput, and
  `--sector-size 4096` becomes worth costing out properly.

## What this rules in

`max_hw_sectors_kb=128` means a 6 MB expert weight region is split into ~48
block-layer requests regardless of engine or queue depth. That is normal for NVMe
(MDTS), not a defect, and it is the same for every arm — so it cannot explain a
difference *between* arms, only the absolute rate. Worth knowing when reading any
GB/s figure taken here against the 0.84 / 2.02 GB/s pair from a different box.

---

## Addendum 2026-08-09: the ring-scaling argument above is withdrawn

The section "Confirmed: the LUKS sector size is 512 bytes" rests part of its case
on this row:

> O_DIRECT throughput vs ring count: 0.60 / 0.81 / 0.86 / 0.83 / 0.69 GB/s at
> 1 / 2 / 4 / 8 / 12 rings — **plateaus at 4 rings and declines** — does not
> behave like a CPU-bound crypto path, which would keep scaling on 12 threads

**That inference does not hold, for three independent reasons.**

**1. The curve reproduces with no crypto and no device.** Run against a tmpfs
file — dm-crypt nowhere in the path, nothing but RAM — `iobench` produces the
same shape: a peak at 4 rings and a decline after.

| rings | tmpfs median GB/s (5 reps) | spread |
|---|---|---|
| 1 | 1.12 | 77 % |
| 2 | 2.11 | 95 % |
| **4** | **3.77** | 16 % |
| 8 | 3.45 | 17 % |
| 12 | 3.00 | 15 % |
| 16 | 3.16 | 9 % |

A shape that appears identically with and without dm-crypt is not evidence about
dm-crypt. The cause is CPU: this box runs an OSX-KVM guest and a VNC helper that
hold ~36 % of all twelve threads before a benchmark starts, so past ~4 rings the
harness contends with them.

**2. Every number in that row was a single pass.** Five identical passes at the
same settings returned 2.58 / 2.13 / 1.78 / 2.27 / 2.34 GB/s — a spread of 35 % of
the median. Differences of 0.86 vs 0.83 vs 0.69 are far inside that. `iobench`
now defaults to 5 reps and prints the spread beside the median.

**3. Held to equal work with no offset wrap, the device curve is flat.** The
original sweep varied total bytes with ring count *and* wrapped its offsets (see
M5's addendum). Fixed — 96 reads of 16 MB at every point, no wrap, 15 reps:

| rings × iters | device median GB/s | spread |
|---|---|---|
| 4 × 24 | 0.87 | 11 % |
| 8 × 12 | 0.92 | 15 % |
| 12 × 8 | 1.04 | 30 % |
| 16 × 6 | 1.00 | 23 % |

0.87 → 1.04 across a 4× change in ring count, against per-arm spreads of 11–30 %.
There is no plateau-and-decline to explain, because there is no resolvable trend.

**What this changes.** The dm-crypt hypothesis is neither supported nor refuted by
ring scaling — that evidence is simply withdrawn, and the row should be struck
from the mixed-evidence table above. **The recommended settling measurement is
unchanged and still outstanding**, still read-only, still one sudo:

```bash
sudo dd if=/dev/nvme0n1p2 of=/dev/null bs=1M count=2000 iflag=direct
```

The other two rows of that table stand: `dd` at 1.5 GB/s and the ~1.7-cores-per-GB/s
CPU figure are unaffected by any of this.
