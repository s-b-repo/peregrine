#!/usr/bin/env bash
# Track A device-pure claims A/B at B=16, REPEATS=3 (publishable per
# docs/measurement.md): off-arm re-measures the 0.86 GB/s delivered baseline,
# on-arm flips COLI_IO_DEVICE_SCHED=1. Primary metrics: delivered GB/s and io
# duty from the [lane] counters, then tok/s. Requires job 90's rebuild (the
# knob does not exist in the 11:26 binary).
set -u
cd "$(dirname "$0")/../../.."
export COLI_MODEL=${COLI_MODEL:-/home/cortix/models/GLM-5.2}
REPEATS=3 MAX_TOKENS=32 PORT=8146 nice -n 5 scripts/bench-serve-envarms.sh \
  bench-data/2026-08-15-queue/device-sched-b16 16 \
  bench-data/2026-08-15-queue/arms/device-off.env \
  bench-data/2026-08-15-queue/arms/device-on.env
