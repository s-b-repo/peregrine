# 4-drive split bench (2026-09-22)

GLM-5.2-r5 shards are **split** across 4 HDDs (`sda/sdb/sdc/sdd` at
`/srv/m-sd?/GLM-5.2-r5`, joined by `model_paths.json`), not mirrored — so
`COLI_IO_MIRROR_ROOTS` does not apply here. The parallelism lever is
multi-ring + device-pure claims (`COLI_IO_DEVICE_SCHED`).

Harness: `iobench FILE 32 4 1 1 uring 3` (32 MB blocks, 4 reads = 128 MB,
1 ring, O_DIRECT so reps stay device-bound; no root for cache drops).
One ~1 GB shard per drive (sda trunk 5 GB, sdb/sdd ~1.1 GB experts,
sdc 189 MB experts — working set fits all four without wrap).

## Single-drive baselines (sequential)

| Drive | File | Median |
|---|---|---:|
| sda | trunk-sda-00000 (5.0 GB) | **0.31 GB/s** (spread 4.5%) |
| sdb | experts-l10-sdb (1.06 GB) | **0.29 GB/s** (spread 7.4%) |
| sdc | experts-l10-sdc (189 MB) | **0.09 GB/s** (spread 35% — noisy, slow drive) |
| sdd | experts-l10-sdd (1.12 GB) | **0.34 GB/s** (spread 6.0%) |
| **sum** | | **≈ 1.03 GB/s** |

## Four drives at once (4× iobench concurrently)

| Drive | Median under contention |
|---|---:|
| sda | 0.24 GB/s |
| sdb | 0.25 GB/s |
| sdc | 0.09 GB/s |
| sdd | 0.29 GB/s |
| **aggregate** | **≈ 0.87 GB/s (85% of the 1.03 sum)** |

## Reading

- 4 concurrent drives deliver **~2.8× the fastest single drive** (0.87 vs
  0.31). Each drive dips ~15% under contention (shared io-wq CPU across 4
  processes on 12 cores) — expected, still strongly net-positive.
- **sdc is the straggler** (0.09 GB/s, 35% spread): a device-mixed claim
  reaps behind it, which is exactly the mechanism device-pure claim groups
  (`COLI_IO_DEVICE_SCHED`, on by default when shards span >1 device) exist
  to stop. sdc's 75 small shards also suggest a placement review.
- Absolute rates are HDD-class (spinning seeks on 32 MB random reads via
  O_DIRECT); the NVMe (`/srv/model600p`) path in `model_paths.json` is not
  measured here.
