#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-/srv/m-sdd/Ornith-1.5-35B-A3B-4drive}"
BIN="${BIN:-$REPO_DIR/target-cuda/release/peregrine-serve}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8140}"
MODEL_ID="${MODEL_ID:-Ornith-1.5-35B-A3B}"
MAX_BATCH="${MAX_BATCH:-8}"

[[ -x "$BIN" ]] || { printf 'missing %s; run scripts/setup-ornith35.sh build\n' "$BIN" >&2; exit 1; }
[[ -f "$MODEL_DIR/config.json" && -f "$MODEL_DIR/model_paths.json" && -f "$MODEL_DIR/manifest.json" ]] || { printf 'missing four-drive model at %s; run scripts/setup-ornith35.sh all\n' "$MODEL_DIR" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { printf 'nvidia-smi not found\n' >&2; exit 1; }
command -v jq >/dev/null || { printf 'jq not found\n' >&2; exit 1; }
mapfile -t MODEL_PATHS < <(jq -r '.paths[]' "$MODEL_DIR/model_paths.json")
[[ ${#MODEL_PATHS[@]} -eq 4 ]] || { printf 'model_paths.json must list four drives\n' >&2; exit 1; }
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    [[ -f "$MODEL_PATH/.peregrine-ornith35-layout" ]] || { printf 'missing Ornith layout marker in %s\n' "$MODEL_PATH" >&2; exit 1; }
done

export COLI_MODEL="$MODEL_DIR"
export COLI_STREAM=1
export COLI_GPU=1
export COLI_GPU_INT4=1
export COLI_GPU_F32_FRAC=0
export COLI_GPU_PINNED=1
export COLI_GPU_DEVICES="${COLI_GPU_DEVICES:-0}"
export COLI_PAR_THREADS="${COLI_PAR_THREADS:-12}"
export COLI_QWEN_THINK="${COLI_QWEN_THINK:-1}"
export COLI_IO_BATCH="${COLI_IO_BATCH:-8}"
export COLI_IO_DEVICE_SCHED=1
export COLI_SSD_AWARE_SCHED=1
export COLI_ECACHE_GB="${COLI_ECACHE_GB:-8}"
export COLI_KV_BUDGET_MB="${COLI_KV_BUDGET_MB:-6000}"
export COLI_CUDA_GRAPH=1
export COLI_MOE_ENGINE=concurrent
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

printf 'Ornith model: %s\n' "$MODEL_DIR"
printf 'CUDA devices: %s\n' "$COLI_GPU_DEVICES"
printf 'Experts: four-drive streaming + int4 VRAM tier\n'
printf 'Endpoint: http://%s:%s/v1\n' "$HOST" "$PORT"

exec "$BIN" \
    --model "$MODEL_DIR" \
    --host "$HOST" \
    --port "$PORT" \
    --model-id "$MODEL_ID" \
    --max-batch "$MAX_BATCH" \
    "$@"
