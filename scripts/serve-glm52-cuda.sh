#!/usr/bin/env bash
# peregrine-serve: GLM-5.2, CUDA, tuned from this repo's own measurements.
#
# Every knob set below cites the bench-data run that licenses it. Knobs that
# *sound* like wins and measured otherwise are listed at the bottom, set to
# their safe values with the number that disqualified them — so nobody
# re-enables them from first principles.
#
# Model layout: 5-way bandwidth-proportional split, no RAID, OS drive excluded.
#   sda 537 MB/s -> 22.7%   sdb 514 -> 21.9%   sdc 91 -> 3.9%
#   sdd 547 MB/s -> 23.0%   nvme1n1 669 -> 28.5%    aggregate 2.36 GB/s
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${COLI_MODEL:-/srv/m-sda/GLM-5.2-r5}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8131}"
MAX_BATCH="${MAX_BATCH:-16}"

# ── measured-faster ───────────────────────────────────────────────────────────
# Batching is the largest lever in the repo: 4.4x aggregate at B=16
# (0.056 -> 0.244 tok/s, docs/benchmarks.md:285). The knee is at or below B=32 —
# B>=32 arms die on the 3600 s per-stream guillotine. Do not raise past 16.
: "${MAX_BATCH:=16}"

# Warm expert cache. 8 GB is the best B=16 serve number in the tree
# (0.088 tok/s / 5794 s, bench-data/2026-08-15-ecache-auto/ecache8-rep1.json).
# Do NOT use =auto: it sized against host MemAvailable inside the harness cgroup
# and was OOM-killed at 0/16 streams. Do NOT exceed ~half MemAvailable — a hit
# that has been paged out is a page fault, and throughput collapses while the
# cache's own hit rate keeps climbing.
# 6, not the 8 that measured best on the old 3-drive layout: the fifth ring
# above costs ~1.6 GB of stream buffer and this box has 29 GB available, so
# something had to give. Trading cache for the ring is the right way round here
# — the cache tops out at 5.7% hit rate even at 12.88 GB (one token routes
# 10.85 GB, so nothing that fits in RAM changes the shape), whereas an unhomed
# device is a whole drive contributing only opportunistically. Revisit if the
# [ram] boot line shows headroom.
export COLI_ECACHE_GB="${COLI_ECACHE_GB:-6}"

# Speculation, floored. +37% at B=16 (0.060 -> 0.082 median, REPEATS=3) and
# 22% fewer disk expert-reads (289059 vs 369955), docs/benchmarks.md:747.
# The floor is what makes it pay: COLI_DRAFT=5 WITHOUT COLI_SPEC_CONF measured
# 1.57x SLOWER (24.3 vs 15.5 s/token) because the read union grew 2.63x.
# These two travel together — never set COLI_DRAFT alone.
export COLI_DRAFT="${COLI_DRAFT:-5}"
export COLI_SPEC_CONF="${COLI_SPEC_CONF:-0.65}"

# Drop speculative prefetches that are already stale: +6.9% at B=16
# (0.072 -> 0.077 median, REPEATS=3), speculative reads -68%. Already the code
# default since 2026-08-16; set explicitly because docs/configuration.md still
# documents it as off and someone will "fix" that by unsetting it.
export COLI_PREFETCH_STALE_DROP="${COLI_PREFETCH_STALE_DROP:-1}"

# Expert-read batch size. The ceil-divide fix took decode 21.83 -> 16.08 s/tok
# and io duty 24% -> 84%. 8 (not the 16 default) is what every 2026-08-13..15
# arm used: it cuts the stream reserve 11.5 -> ~6.4 GB, which is what lets the
# 8 GB warm cache coexist with 4 rings on a 46 GB box.
export COLI_IO_BATCH="${COLI_IO_BATCH:-8}"

# KV session store: cold 2620.5 s -> warm 391.9 s (6.7x), output byte-identical
# across the restart; re-proven 2504.6 -> 389.0 s on the async writer.
# This is the single biggest win for *coding*, where you resend a long file or
# repo context every turn. It is prefill-only — it does nothing for decode.
# Placed on the OS NVMe deliberately: that drive is the fastest here (1.12 GB/s)
# and, now that the model is off it, no longer competes with expert streaming.
export COLI_KV_STORE_DIR="${COLI_KV_STORE_DIR:-/home/cortix/.cache/peregrine-kv}"
export COLI_KV_STORE_MB="${COLI_KV_STORE_MB:-40000}"
mkdir -p "$COLI_KV_STORE_DIR"

# Keep RSS flat across the io/compute worker pools.
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

# ── CUDA ──────────────────────────────────────────────────────────────────────
# 1.09x at B=16 (0.244 vs 0.224, tight across 3 reps, docs/benchmarks.md:277).
# The win is lane overlap, NOT residency: 62 residents against ~19200 experts
# routed per B=16 step is 0.3% residency. A 12 GB 3060 cannot hold one token's
# 11.3 GB working set, so do not expect the tier to grow into a win.
#
# ACCURACY NOTE — this is the one setting here that changes token values.
# GPU experts compute in f32 where the CPU path is int4, so the GPU arm is more
# accurate but NOT bit-identical to a CPU baseline. Unset COLI_GPU if you need
# bit-exactness against the CPU reference.
export COLI_GPU="${COLI_GPU:-1}"
# Disk -> pinned host page -> VRAM, two DMAs and no userspace copy (2026-08-22).
# Default on; set explicitly so the A/B control is visible.
export COLI_GPU_PINNED="${COLI_GPU_PINNED:-1}"
export COLI_GPU_UPLOAD_DEPTH="${COLI_GPU_UPLOAD_DEPTH:-4}"

# ── this layout specifically ──────────────────────────────────────────────────
# Neither the ring count nor device-pure claims are set here, on purpose.
#
# As of 2026-08-22 the engine derives both from the shard->device map: claims are
# grouped per physical device (default on whenever the shards span more than one
# device), and the ring count defaults to ONE RING PER DEVICE. That second half
# is what only matters once you have five drives: `ring_homes()` shares rings
# across the per-device claim groups proportionally, so with 5 groups and the
# historical 4 rings one group gets **zero home rings** and is reached only when
# some other ring runs dry and steals from it — a whole drive running part-time.
#
# Setting COLI_IO_RINGS explicitly overrides the device count AND skips the RAM
# back-off that walks it down when stream buffers would eat too much of
# MemAvailable, so a hardcoded number here would silently defeat both. Read the
#   [io] device-pure claims on (5 devices, N shard fds, one ring per device)
# boot line instead — it says outright when a device did not get a home ring.
#
# Still UNMEASURED: no bench-data arm has run device-pure claims. A/B against
# COLI_IO_DEVICE_SCHED=0 before believing it.

# ── measured NOT to help: left off on purpose, with the disqualifying number ──
# COLI_DIRECT=1        O_DIRECT measured 0.86 vs 1.12 GB/s buffered = -23%,
#                      outside both spreads. Kernel readahead is worth more than
#                      the page-cache bypass at 0.6% reuse.
# COLI_IO_ENGINE=pread 1.06 vs 1.12 GB/s — a 5.7% gap BELOW the noise floor.
# COLI_REGBUF=1        was inert for a year; now wired but read_fixed_many copies
#                      out of the registered buffer, and it ENOMEMs at the
#                      default 8 MB RLIMIT_MEMLOCK.
# COLI_SPEC_GDN=1      1.55x SLOWER, and Qwen-hybrid only — inert on GLM anyway.
# COLI_ROUTE_MIN_SHARE flip_rate 0.279 at tau=0.05. Fails its quality gate.
# COLI_MOE_ENGINE=sched slower by construction: no GPU lane, no warm cache,
#                      no prefetch. It exists as a second-implementation A/B.
# COLI_CACHE_SWEEP=1   predicted ~535 extra hits, measured 71.
# Prefetch on/off      CONTESTED: the prose says off is slower (23.8 vs 21.9),
#                      but no bench-data file backs that pair, while the archived
#                      arms say the reverse (356 s off vs 502 s on at 8 tokens).
#                      Left at default because it is genuinely unresolved.

echo "== peregrine-serve — GLM-5.2, CUDA, 5-drive split =="
echo "model   : ${MODEL_DIR}"
echo "listen  : http://${HOST}:${PORT}/v1   batch=${MAX_BATCH}"
echo "gpu     : ${COLI_GPU} (pinned=${COLI_GPU_PINNED})  ecache=${COLI_ECACHE_GB}G"
echo "spec    : draft=${COLI_DRAFT} conf=${COLI_SPEC_CONF}   io_batch=${COLI_IO_BATCH}"
echo "kvstore : ${COLI_KV_STORE_DIR} (${COLI_KV_STORE_MB} MB)"
echo "auth    : ${PEREGRINE_API_KEY:+bearer key configured}${PEREGRINE_API_KEY:-NO KEY — refusing to start}"
echo "===================================================="

if [ -z "${PEREGRINE_API_KEY:-}" ]; then
    echo "ERROR: PEREGRINE_API_KEY is unset. This server is reachable from the" >&2
    echo "       Cloudflare tunnel; starting it without bearer auth would expose" >&2
    echo "       an unauthenticated endpoint. Export a key and retry." >&2
    exit 1
fi

exec "${REPO_DIR}/target/release/peregrine-serve" \
    --model "${MODEL_DIR}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-batch "${MAX_BATCH}"
