#!/usr/bin/env bash
# Stage-5 rerun with the cgroup-clamp fix (commit 6704288): the 08-15 arm OOM'd
# because auto sizing read the root cgroup's "max" inside a 34G scope. REPEATS=1
# screen, same arms as the original so the datasets read against each other.
# Requires job 90's rebuild (the fix is in the binary, not the env).
set -u
cd "$(dirname "$0")/../../.."
export COLI_MODEL=${COLI_MODEL:-/home/cortix/models/GLM-5.2}
REPEATS=1 MAX_TOKENS=32 PORT=8146 nice -n 5 scripts/bench-serve-envarms.sh \
  bench-data/2026-08-15-queue/ecache-auto-refix-b16 16 \
  bench-data/2026-08-15-ecache-auto/arms/ecache8.env \
  bench-data/2026-08-15-ecache-auto/arms/ecache-auto.env
