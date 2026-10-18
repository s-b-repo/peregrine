#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-/srv/m-sdd/GLM-5.3-Flash-5ssd}"
BIN="${BIN:-$REPO_DIR/target-cuda/release/peregrine-serve}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8133}"
MODEL_ID="${MODEL_ID:-GLM-5.3-Flash}"
MAX_BATCH="${MAX_BATCH:-16}"
KV_STORE_DIR="${KV_STORE_DIR:-/srv/model600p/peregrine-kv}"
KV_STORE_MB="${KV_STORE_MB:-12000}"

[[ -x "$BIN" ]] || { printf 'missing %s; run cargo build with CUDA enabled\n' "$BIN" >&2; exit 1; }
[[ -f "$MODEL_DIR/config.json" && -f "$MODEL_DIR/model_paths.json" ]] || { printf 'missing four-drive GLM model at %s\n' "$MODEL_DIR" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { printf 'nvidia-smi not found\n' >&2; exit 1; }
mkdir -p "$KV_STORE_DIR"

export COLI_MODEL="$MODEL_DIR"
export COLI_STREAM=1
# RSS ceiling. A B=16 run at COLI_ECACHE_GB=10 stacked ~20 GB of RSS on this 46 GB
# host: MemAvailable went 28.7 -> 3.0 GB and the box started swapping. Refuse
# rather than swap. (An earlier note here blamed "~16 GB of deleted tmpfs files"
# held by browsers/Steam. That does not reproduce: deleted-but-still-open
# descriptors across the top RSS processes total under 1 MB. The reclaimable holder
# is baloo_file at ~4.4 GB RSS -- which also competes for the bench stripe's
# bandwidth, since nothing is excluded from indexing /srv. docs/io-and-storage.md.)
export COLI_RSS_GUARD_GB="${COLI_RSS_GUARD_GB:-28}"
# The VRAM *expert tier* is admissible on this arch now — the SwiGLU clamp gate
# (model.rs:4846) admits it, and it builds: "GPU tier holds 771 experts in VRAM".
# It was still a 39% LOSS at B=1 here: 771/12096 experts = 6.4% residency placed
# round-robin (cold knapsack), and disk traffic was unchanged at 4.239 GB/token.
# Opt in with EXPERT_TIER=1 once VRAM can hold the hot set (or at wide batch);
# leave it off for single-stream decode. docs/gpu-vendors.md has the numbers.
export COLI_GPU_INT4=1
export COLI_GPU_F32_FRAC=0
export COLI_GPU_EXPERT_HEADROOM_MB="${COLI_GPU_EXPERT_HEADROOM_MB:-1792}"
export COLI_GPU_PINNED=1
export COLI_GPU_DEVICES="${COLI_GPU_DEVICES:-0}"
if [[ "${EXPERT_TIER:-0}" == "1" ]]; then
    export COLI_GPU=1
else
    unset COLI_GPU
fi
export COLI_CUDA_GRAPH=1
export COLI_CUDA_FUSED_REDUCE=1
export COLI_GPU_DENSE=1
export COLI_GPU_DENSE_HEADROOM_MB="${COLI_GPU_DENSE_HEADROOM_MB:-256}"
export COLI_GPU_DENSE_MAX_BATCH="${COLI_GPU_DENSE_MAX_BATCH:-1}"
# Router look-ahead: OFF on this container. The window pre-streams every
# predicted candidate (COLI_PREFETCH_WARM_PATHS defaults to "all"), so on a
# streaming model the speculative reads share each device's ring with the reads
# the token actually waits for — and the ledger prices that at 12.9% of disk
# traffic wasted, on top of the queue delay.
#
# Re-measured 2026-10-06 against the old N=64 setting, interleaved arms, CUDA
# profile, same box (aggregate tok/s; medians of 4 / 2 / 3 runs — raw runs in
# bench-data/2026-10-06-lookahead/):
#   B=1   N=64 0.095  ->  off 0.163   (+72%; 3 of 4 clean runs, one 0.060 outlier)
#   B=8   N=64 0.193  ->  off 0.308   (+60%, both pairs)
#   B=16  N=64 0.275  ->  off 0.271   (no resolvable difference; an earlier
#                                          single run read +34% the other way)
# The earlier claim on this line — "N=64 is the measured pick on this box (+14%
# over off)" — does not reproduce here, and it was measured against a *four*-drive
# stripe that included the 5.4k RPM HDD. With three SSDs the speculative reads
# are pure loss. Re-measure per container; do not trust this line on a new box.
export COLI_ROUTER_LOOKAHEAD="${COLI_ROUTER_LOOKAHEAD:-0}"
# Kept for interface parity: it only has an effect when the look-ahead above is
# on, so `COLI_ROUTER_LOOKAHEAD=1 COLI_ROUTER_LOOKAHEAD_N=64` still reproduces the
# old arm exactly.
export COLI_ROUTER_LOOKAHEAD_N="${COLI_ROUTER_LOOKAHEAD_N:-64}"
# Cheap tier-2 prefetch for the same candidates: a page-cache `fadvise` instead of
# a device read, so warming the cache no longer occupies the ring that the token's
# own reads are queued on. Suppressed under O_DIRECT, hence inert rather than
# harmful here; kept so a buffered lane still gets it.
export COLI_PREFETCH_HINT_PATHS="${COLI_PREFETCH_HINT_PATHS:-64}"
export COLI_MOE_ENGINE=concurrent
export COLI_IO_BATCH="${COLI_IO_BATCH:-8}"
export COLI_IO_DEVICE_SCHED=1
export COLI_SSD_AWARE_SCHED=1
# 14 GB, not 6: one token routes 4.04 GB, so a 6 GB cache holds ~0.6 of a pass
# and 14 GB holds ~1.3, which takes the bytes a token reads from disk from 3.139
# to 2.704 GB (-14 %, reproducible). The tok/s effect (+4 %) is inside this box's
# run-to-run noise, so the byte saving is the claim. Needs `MemAvailable` for it:
# the [ram] projection line at startup is the thing to check.
export COLI_ECACHE_GB="${COLI_ECACHE_GB:-14}"
export COLI_DRAFT="${COLI_DRAFT:-0}"
export COLI_PREFETCH_STALE_DROP=1
export COLI_FUSE_PREFILL=1
export COLI_KV_STORE_DIR="$KV_STORE_DIR"
export COLI_KV_STORE_MB="$KV_STORE_MB"
export COLI_KV_BUDGET_MB="${COLI_KV_BUDGET_MB:-6000}"
export COLI_PAR_THREADS="${COLI_PAR_THREADS:-12}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

printf 'GLM model: %s\n' "$MODEL_DIR"
# Was "sda+sdb+sdc+sdd"; sdc is a 5.4k RPM HDD that capped the whole stripe at
# ~40 MB/s on its share of every token (the 0.083 tok/s baseline). model_paths.json
# now lists only the three SSDs, proportionally weighted by measured read speed.
printf 'Storage: sda+sdb+sdc+sdd+sde (five SSDs, re-striped 2026-10-08 by measured bandwidth)\n'
printf 'CUDA devices: %s\n' "$COLI_GPU_DEVICES"
printf 'Max batch: %s; 1 tok/s target is aggregate throughput\n' "$MAX_BATCH"
printf 'Endpoint: http://%s:%s/v1\n' "$HOST" "$PORT"

exec "$BIN" \
    --model "$MODEL_DIR" \
    --host "$HOST" \
    --port "$PORT" \
    --model-id "$MODEL_ID" \
    --max-batch "$MAX_BATCH" \
    "$@"
