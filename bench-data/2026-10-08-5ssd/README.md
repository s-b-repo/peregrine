# 2026-10-08 — GLM-5.3-Flash on five SSDs, and the cache policies

Everything here is one serialized arm at a time on an idle box, interleaved
where two configurations were compared. Raw runs are in this directory; the
summary is in `docs/deployment-glm53-flash.md`.

## The stripe

`scripts/reshard-glm53-5ssd.sh` re-striped the container across five SSDs using
**measured** O_DIRECT bandwidth as the weights, then byte-verified every tensor
(`verification passed (byte-identical)`) and staged the result into
`/srv/m-sdd/GLM-5.3-Flash-5ssd` with per-device dirs named `glm53-experts-5ssd`.
The three-drive container (`GLM-5.3-Flash-4ssd`) was never modified.

Measured read bandwidth (O_DIRECT, 8 MiB blocks, median of 12):

| device | median | min | max | spread |
|---|---:|---:|---:|---:|
| sda (Kingston A400) | 434 MB/s | 123 | 501 | **307 %** |
| sdb (Acer RE100) | 537 | 534 | 543 | 1 % |
| sdc (Crucial 120G, ex-Kali) | 562 | 560 | 563 | 0 % |
| sdd (Crucial 1T) | 561 | 534 | 563 | 5 % |
| sde (Seagate 120G, ex-Windows) | 551 | 548 | 551 | 0 % |

`sda` is the outlier and the measurement script says so out loud: its rate swings
by 3×, so any stripe weight derived from a single `dd` is a guess. The weights
actually used came from a run where sda measured 255 MB/s, which gave it a 10.4 %
share — defensible, since every token waits on its slowest drive, but it is the
one weight in this set that deserves a re-measure before anyone re-stripes again.

## Three drives → five, interleaved, 6 decode steps

| | tok/s (median of 3) | runs | GB/token from disk | slowest device's learned rate |
|---|---:|---|---:|---:|
| 3 SSD | 0.187 | 0.182 / 0.187 / 0.197 | 3.139 | 185 MB/s |
| 5 SSD | **0.227** | 0.207 / 0.227 / 0.232 | 3.139 | 379 MB/s |

**+21 %**, 3 of 3 reps, against an 11 % spread on the winning arm. Bytes per
token are identical, so this is bandwidth and nothing else — and the mechanism is
visible in the per-device rates: with five devices every one of them sits above
379 MB/s, where with three, one was at 185 MB/s and gated every token. Total
array bandwidth rose 1.47 → 2.65 GB/s (+80 %) and only 21 % of that became
tokens, which is the same engine-side inefficiency measured on 2026-10-06 and is
still the biggest open gap.

## Cache size: a real win on bytes, unresolvable on tok/s

Interleaved, 6 SSD, 6 decode steps:

| `COLI_ECACHE_GB` | tok/s median | runs | GB/token from disk | cache-served |
|---|---:|---|---:|---:|
| 6 | 0.268 | 0.260 / 0.268 / 0.297 | 3.139 | 25.9 % |
| 14 | 0.278 | 0.277 / 0.278 / 0.300 | **2.704** | 36.2 % |

The byte figure is deterministic and reproducible — 3.139 GB/token in all 11 CUDA
runs at 6 GB, 2.704 in all 4 at 14 GB, re-checked on a freshly rebuilt binary: doubling the cache takes 14 % off the bytes a token
reads, because one token routes 4.04 GB and the cache now holds ~1.3 passes
instead of ~0.6. The tok/s difference (+4 %) is **inside this box's noise** — the
same configuration has produced 0.148 and 0.246 back to back in an earlier
session — so the honest claim is the byte saving plus "no measurable harm".

## Cache policies that are implemented, off by default, and now measured

| arm | tok/s | GB/token | verdict |
|---|---:|---:|---|
| `COLI_CACHE_LFRU=1` | 0.220 / 0.289 | **3.581** | **negative**: 14 % *more* bytes than the default |
| `COLI_JOINT_EVICTION=1` | 0.174 | 3.139 | no byte change, no measurable gain |
| `COLI_ECACHE_GB=12` | 0.265 | 2.767 | same direction as 14 GB |
| `COLI_ECACHE_GB=14 COLI_CACHE_LFRU=1` | 0.290 | 2.714 | LFRU's byte penalty shrinks as the cache grows |

The LFRU result is the interesting one, and it is a negative: the policy that
FlashMoE (arXiv 2601.17063) reports as **+51 % hit rate over LRU and LFU** on an
SSD-offloaded MoE *loses* 14 % of the bytes here. `tier::lfru_score` ranks by
`heat << 8 | (255 - age)`, so one routing-frequency count outweighs every recency
advantage — correct in principle, wrong in practice when `heat` is stale or sparse
relative to a cache holding several passes. Leave it off. That is worth knowing
*before* someone reads the paper and turns it on.

## A measurement trap worth writing down

A release build issued without `--features cuda` produces a binary that prints
one advisory line (`gpu=unavailable (CUDA backend not built)`) and then runs to
completion, yielding a perfectly plausible tok/s on the **CPU** engine — 0.247
tok/s and 3.066 GB/token instead of 2.704, with the same env and the same model.
It cost a confused detour here. `COLI_GPU=1` is a request, not a guarantee, so
every harness in this directory now greps stderr for `gpu=available` and exits
non-zero instead of recording the number; `scripts/bench-lookahead-arms.sh` does
the same, under `REQUIRE_GPU=0` for a CPU run you meant.

## What did not change

- `COLI_IO_BATCH` 8/16/32/64: no resolvable effect, on either container.
- Bytes per token: 3.139 GB on both containers, as it must be — same model, only
  the placement differs.