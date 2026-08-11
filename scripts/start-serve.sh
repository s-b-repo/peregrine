#!/usr/bin/env bash
# Launch peregrine-serve HTTP server for Open Code / OpenAI-compatible clients.
# Model: GLM-5.2 split across 3 drives:
#   1. Main NVMe:  /home/cortix/models/GLM-5.2
#   2. RAID0:      /srv/modelstripe/GLM-5.2-r4
#   3. 600p NVMe:  /srv/model600p/GLM-5.2-r4 (via model_paths.json)

set -euo pipefail

MODEL_DIR="${COLI_MODEL:-/home/cortix/models/GLM-5.2}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8180}"
MODEL_ID="${MODEL_ID:-glm-5.2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "== Starting peregrine-serve =="
echo "Model Dir: ${MODEL_DIR}"
echo "Listening: http://${HOST}:${PORT}/v1"
echo "Model ID:  ${MODEL_ID}"
echo "==============================="

# Stop any previously running peregrine-serve instance if requested or stale
if [ "${KILL_EXISTING:-1}" = "1" ]; then
    pkill -9 -x peregrine-serve 2>/dev/null || true
    sleep 0.5
fi

# Build release binary with CUDA features if missing
if [ ! -f "${REPO_DIR}/target/release/peregrine-serve" ]; then
    echo "Building CUDA release binary..."
    cargo build --release --manifest-path "${REPO_DIR}/Cargo.toml" -p peregrine-serve --features cuda
fi

export COLI_MODEL="${MODEL_DIR}"
# COLI_GPU is opt-in: presence of the var enables GPU VRAM expert tier.
# The RTX 3060 (12GB) cannot hold 10.9GB resident + routing tier — unset to force CPU.
unset COLI_GPU
# To re-enable GPU: export COLI_GPU=1
export COLI_IO_BATCH="${COLI_IO_BATCH:-8}"
export COLI_RAM_OVERCOMMIT="${COLI_RAM_OVERCOMMIT:-1}"
export COLI_FUSE_PREFILL="${COLI_FUSE_PREFILL:-1}"
# The 744B GLM-5.2 exceeds resident RAM — stream routed experts from disk across all 3 drives.
export COLI_STREAM="${COLI_STREAM:-1}"
# No RAM warm cache (set to 0): with 38 GB usable and 10.9 GB resident weights,
# keeping a large cache would cause page faults under memory pressure. Instead,
# rely on the OS page cache (COLI_DIRECT unset = default) which caches fast NVMe
# reads automatically and can be reclaimed by the kernel under pressure.
# Set COLI_ECACHE_GB=11 to cache one full routing pass (10.85 GB) in RAM.
export COLI_ECACHE_GB="${COLI_ECACHE_GB:-0}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

echo "API key: ${PEREGRINE_API_KEY:+configured}${PEREGRINE_API_KEY:-NOT SET — bearer auth disabled}"
echo "Stream: ${COLI_STREAM} | Warm cache: ${COLI_ECACHE_GB} GB | GPU: ${COLI_GPU:-off}"
echo "==============================="

exec "${REPO_DIR}/target/release/peregrine-serve" \
    --model "${MODEL_DIR}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --model-id "${MODEL_ID}" \
    --api-key "${PEREGRINE_API_KEY:-}" \
    "$@"
