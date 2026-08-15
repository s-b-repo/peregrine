#!/usr/bin/env bash
# REPEATS=3 confirmation of COLI_ECACHE_GB=auto vs fixed 8 at B=16 — enable
# only if the stage-5 REPEATS=1 screen (bench-data/2026-08-15-ecache-auto)
# showed a win. Gap must exceed spread or the verdict stays "unresolved".
set -u
cd "$(dirname "$0")/../../.."
export COLI_MODEL=${COLI_MODEL:-/home/cortix/models/GLM-5.2}
REPEATS=3 MAX_TOKENS=32 PORT=8146 nice -n 5 scripts/bench-serve-envarms.sh \
  bench-data/2026-08-15-queue/confirm-ecache-auto-b16 16 \
  bench-data/2026-08-15-ecache-auto/arms/ecache8.env \
  bench-data/2026-08-15-ecache-auto/arms/ecache-auto.env
