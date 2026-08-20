#!/usr/bin/env bash
# Multi-domain routing traces on GLM-5.2, for the expert-map gate.
#
# One trace per domain, same N, same model, same flags — the only thing that
# differs is the corpus. `--weights` so the frames carry gate mass (the shape
# `dump-routes` could not write until 2026-08-20).
#
# Timing is irrelevant to this measurement: the output is *which experts route*,
# which is deterministic given the corpus. Contention on the box changes how long
# it takes and nothing about what it records.
set -u
MODEL=${MODEL:-/home/cortix/models/GLM-5.2}
D=$(cd "$(dirname "$0")" && pwd)
N=${N:-256}

for c in prose code json techdoc; do
  out="$D/routes-$c.json"
  if [ -s "$out" ]; then
    echo "$(date +%H:%M:%S) $c already traced, skipping" >&2
    continue
  fi
  echo "$(date +%H:%M:%S) tracing $c (N=$N)" >&2
  t0=$(date +%s)
  # ROUTE_STATS_PERSIST off: a trace run must not write its own routing back
  # into the checkpoint's route_stats.json and contaminate the next domain.
  nice -n 10 env COLI_STREAM=1 COLI_ROUTE_STATS_PERSIST=0 \
    "$D/../../target/release/peregrine" dump-routes "$MODEL" "$out" "$N" \
    --text "$D/corpora/$c.txt" --weights > "$D/dump-$c.log" 2>&1
  rc=$?
  echo "$(date +%H:%M:%S) $c rc=$rc wall=$(( $(date +%s) - t0 ))s" >&2
  [ $rc -eq 0 ] || echo "FAILED: $c — see dump-$c.log" >&2
done
echo "$(date +%H:%M:%S) ALL DONE" >&2
