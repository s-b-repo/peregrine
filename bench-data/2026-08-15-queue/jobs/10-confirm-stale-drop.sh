#!/usr/bin/env bash
# REPEATS=3 confirmation of COLI_PREFETCH_STALE_DROP at B=16 — enable (move to
# ../jobs/) only if the REPEATS=1 screen in bench-data/2026-08-15-stale-drop/b16
# showed a win. Publishable per docs/measurement.md: gap must exceed spread.
set -u
cd "$(dirname "$0")/../../.."
export COLI_MODEL=${COLI_MODEL:-/home/cortix/models/GLM-5.2}
REPEATS=3 MAX_TOKENS=32 PORT=8146 nice -n 5 scripts/bench-serve-envarms.sh \
  bench-data/2026-08-15-queue/confirm-stale-drop-b16 16 \
  bench-data/2026-08-15-stale-drop/arms/stale-off.env \
  bench-data/2026-08-15-stale-drop/arms/stale-on.env
