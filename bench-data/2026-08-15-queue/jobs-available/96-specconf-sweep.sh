#!/usr/bin/env bash
# Job 96 — spec-conf floor x depth sweep (ranked #2 in the ideas pass).
#
# Why: the +35% stage-4 screen ran ds4's untuned default floor. Every drafted
# token pruned at low confidence is a verify row's expert union not streamed,
# so the floor x depth surface is the cheapest unexplored byte lever. Arms
# {0.5, 0.65, 0.8} x {COLI_DRAFT 5, 6}; the d5-c065 arm reproduces stage 4's
# candidate, doubling as the cross-run consistency anchor. REPEATS=1 screen —
# the winner (if any beats d5-c065 outside the +-3% band) gets its own
# REPEATS=3 confirmation slot, not this job.
#
# ENABLE AFTER: job 20 confirms the +35% screen (a dead screen makes this moot).
# Primary: tokens_per_s per arm. Secondary: the [spec] accept-rate and
# spec_conf_stops lines — envarms' counters.txt grep misses [spec]/[kvstore],
# so this job extracts them from the retained server logs into spec-summary.txt.
# Cost estimate: 6 arms x ~2 h = ~12 h.
set -uo pipefail
cd "$(dirname "$0")/../../.."

OUT=bench-data/2026-08-15-specconf-sweep
A=bench-data/2026-08-15-queue/arms
stamp() { echo "== [$(date '+%F %T')] $*"; }

stamp "job 96: spec-conf floor x depth sweep, 6 arms, B=16, REPEATS=1"
REPEATS=1 MAX_TOKENS=32 PORT=8151 nice -n 5 scripts/bench-serve-envarms.sh \
  "$OUT" 16 \
  "$A/spec-d5-c050.env" "$A/spec-d5-c065.env" "$A/spec-d5-c080.env" \
  "$A/spec-d6-c050.env" "$A/spec-d6-c065.env" "$A/spec-d6-c080.env" \
  || stamp "sweep FAILED (partial arms may still be readable)"

# The accept-rate table envarms' counters grep doesn't keep.
{
  for log in "$OUT"/*.server.log; do
    [ -e "$log" ] || continue
    echo "--- $(basename "$log" .server.log)"
    grep -E '^\[spec\]' "$log" || echo "(no [spec] line — server may not have shut down cleanly)"
  done
} > "$OUT/spec-summary.txt" 2>/dev/null
stamp "job 96 done — tokens_per_s per arm in $OUT/*.json, accept rates in spec-summary.txt"
