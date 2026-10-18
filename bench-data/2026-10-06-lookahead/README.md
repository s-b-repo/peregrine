# 2026-10-06 — router look-ahead on/off, GLM-5.3-Flash-4ssd (3 SSDs)

Serial arms, one at a time, no overlap. Same binary
(`target-cuda/release/peregrine`), same container, CUDA profile as
`scripts/serve-glm53-flash-cuda.sh` minus the two knobs under test.

    COLI_MODEL=/srv/m-sdd/GLM-5.3-Flash-4ssd COLI_STREAM=1 COLI_GPU=1
    COLI_GPU_INT4=1 COLI_GPU_PINNED=1 COLI_CUDA_GRAPH=1 COLI_CUDA_FUSED_REDUCE=1
    COLI_GPU_DENSE=1 COLI_MOE_ENGINE=concurrent COLI_IO_BATCH=8
    COLI_IO_DEVICE_SCHED=1 COLI_SSD_AWARE_SCHED=1 COLI_ECACHE_GB=6
    COLI_DRAFT=0 COLI_PREFETCH_STALE_DROP=1 COLI_FUSE_PREFILL=1
    COLI_BENCH_STEPS=<steps> COLI_UNION_STATS=1

Arm naming: `ab_n64_*` = `COLI_ROUTER_LOOKAHEAD_N=64` (the previously shipped
arm), `ab_n0hint_*` / `b8b_*` / `b16_*` = `COLI_ROUTER_LOOKAHEAD_N=0
COLI_PREFETCH_HINT_PATHS=64`. `.out` is the bench table + byte ledger +
lane/ssdclock summary; `.err` is the startup block.

Aggregate tok/s, per run:

| batch | arm | run 1 | run 2 | run 3 | run 4 | median |
|---:|---|---:|---:|---:|---:|---:|
| 1 | N=64 | 0.093 | 0.087 | 0.101 | 0.096 | 0.095 |
| 1 | off | 0.157 | 0.169 | 0.181 | 0.060 | 0.163 |
| 8 | N=64 | 0.176 | 0.210 | — | — | 0.193 |
| 8 | off | 0.281 | 0.334 | — | — | 0.308 |
| 16 | N=64 | 0.197 | 0.279 | 0.275 | — | 0.275 |
| 16 | off | 0.147 | 0.352 | 0.271 | — | 0.271 |

B=1 run 4 of the "off" arm (0.060) is the one outlier; 3 of 4 sit at
0.157–0.181 and the B=8 pairs reproduce the direction, so the B=1 gap is not
an artifact of that run — but it is also not a clean 4/4.

**The shipped configuration, as edited.** `scripts/serve-glm53-flash-cuda.sh`
after the change (`COLI_ROUTER_LOOKAHEAD=0`, `COLI_PREFETCH_HINT_PATHS=64`,
`COLI_PREFETCH_STALE_DROP=1` — the last was already there), B=1, 3 runs:

| run | aggregate tok/s | GB/token from disk |
|---|---:|---:|
| 1 | 0.244 | 3.139 |
| 2 | 0.148 | 3.139 |
| 3 | 0.246 | 3.139 |
| median | **0.244** | 3.139 |

That is **2.6x the shipped N=64 median (0.095)** on this box, at 6 decode
steps per run. The spread on this box is large — the same configuration has
produced 0.148 and 0.246 back to back — so quote the median and the count, never
the best run.

Also measured in the same session and **not** acted on:

- `COLI_IO_BATCH` 8/16/32/64: no resolvable effect (0.105–0.127 tok/s, noise).
- `COLI_ROUTER_LOOKAHEAD_N` 0/16/64/128 single reps: 0.194/0.117/0.127/0.092.
- `COLI_ECACHE_GB` 6/12 at B=1: 0.154 tok/s and 2.767 GB/token from disk (vs
  3.341 GB/token at 6 GB) — bytes improve, tok/s inside the noise.
- `COLI_ECACHE_GB` 6/12/18 at B=16: byte-identical ledgers, `cache-served`
  0.0% in all three, `re-read (evicted)` 22.0% in all three. The warm cache
  serves nothing on the batched path; that is an open finding, not a result.
- `dd iflag=direct bs=8M`: 370 / 536 / 560 MB/s on sda / sdb / sdd.
  `iobench` through the engine's own reactor: 0.43–0.46 GB/s on sdd for every
  block size 1–64 MB and every ring count 1–16.
