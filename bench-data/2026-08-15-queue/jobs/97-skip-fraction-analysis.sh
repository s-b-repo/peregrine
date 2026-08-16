#!/usr/bin/env bash
# The 08-13 "reads skipped by bounds" number, unblocked by the trace-format fix
# (3c3246a) + the --bounds sidecar loader (26f0f8f): loads the existing
# expert_bounds.json instead of recomputing bounds from the container, so the
# only model-dir touch is one ~MB JSON read — legal here because the queue
# serializes it after the serve sweeps. Debug binary on purpose: pure parsing
# and arithmetic, and the release binary must not be rebuilt out of turn.
# Verdict caveat (documented at the parser): dump-routes traces carry no gate
# weights, so read the g·C column; the gate-only column is degenerate.
set -u
cd "$(dirname "$0")/../../.."
cargo build -p peregrine-tools > /dev/null 2>&1 || exit 1
target/debug/peregrine-skipbound \
  --bounds /home/cortix/models/GLM-5.2/expert_bounds.json \
  --trace bench-data/2026-08-13-int3g64/routes.json
