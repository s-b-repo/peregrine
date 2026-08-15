#!/usr/bin/env bash
# Runs after the REPEATS=3 confirmations (jobs 10/20), which are the last users
# of the 11:26 binary: rebuilds target/release from the integrated tree
# (stale-drop + Seam 1/2 device-sched + serve async boundaries + knob
# migration + auto-ecache cgroup clamp), then reruns the kvstore smoke — the
# async checkpoint writer moved save timing off the engine thread, so the
# smoke's timing lines need re-reading (output identity is unchanged by
# design; the identity check is the smoke's point).
set -u
cd "$(dirname "$0")/../../.."
if ! cargo build --release > bench-data/2026-08-15-queue/rebuild.log 2>&1; then
  echo "release rebuild FAILED — refusing to run further queue jobs against a stale binary"
  touch bench-data/2026-08-15-queue/SKIP
  exit 1
fi
export COLI_MODEL=${COLI_MODEL:-/home/cortix/models/GLM-5.2}
PORT=8146 scripts/kvstore-smoke.sh bench-data/2026-08-15-queue/kvstore-smoke-async
