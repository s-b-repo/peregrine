#!/usr/bin/env bash
# Launch a native checkpoint or an Omni suite with the OpenAI-compatible API.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
features=()
if [ "${PEREGRINE_CUDA:-0}" = "1" ]; then
    features+=(--features cuda)
fi
exec cargo run --locked --release --manifest-path "${REPO_DIR}/Cargo.toml" \
    -p peregrine-serve "${features[@]}" -- "$@"
