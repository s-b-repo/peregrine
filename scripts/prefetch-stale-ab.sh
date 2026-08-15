#!/usr/bin/env bash
# COLI_PREFETCH_STALE_DROP A/B, queued behind the 2026-08-15 overnight chain.
#
# Waits for overnight-2026-08-15.log to print "chain complete" (so the rebuild
# below can never swap binaries under a mid-sweep serve arm), rebuilds
# target/release from the working tree, then runs two envarms sweeps:
# B=16 (where the 08-13 counters showed 98.6% of speculative reads wasted)
# and B=1 (where the prediction is a near-no-op — see the README).
#
#   setsid nohup scripts/prefetch-stale-ab.sh > stale-ab-2026-08-15.log 2>&1 &
#
# Cancel any time with: touch bench-data/2026-08-15-stale-drop/SKIP
set -u
cd "$(dirname "$0")/.."
DATA=bench-data/2026-08-15-stale-drop
CHAIN_LOG=overnight-2026-08-15.log
stamp() { echo "== [$(date '+%F %T')] $*"; }

stamp "waiting on the overnight chain ('chain complete' in $CHAIN_LOG); touch $DATA/SKIP to cancel"
while :; do
  [ -e "$DATA/SKIP" ] && { stamp "SKIP present — exiting without measuring"; exit 0; }
  grep -q 'chain complete' "$CHAIN_LOG" 2>/dev/null && break
  # A dead chain that never printed the sentinel still frees the box; don't wait
  # forever on a log line that can no longer appear.
  pgrep -f 'overnight-2026-08-15.sh' >/dev/null \
    || { stamp "chain script gone without 'chain complete' — box is free, proceeding"; break; }
  sleep 300
done
[ -e "$DATA/SKIP" ] && { stamp "SKIP present — exiting without measuring"; exit 0; }

stamp "rebuilding target/release (safe now: no serve arm mid-sweep can see a binary swap)"
if ! cargo build --release > "$DATA/build.log" 2>&1; then
  stamp "release build FAILED — aborting (see $DATA/build.log)"
  exit 1
fi

export COLI_MODEL=${COLI_MODEL:-/home/cortix/models/GLM-5.2}
stamp "A/B at B=16 (REPEATS=1, the wasted-read regime)"
REPEATS=1 MAX_TOKENS=32 PORT=8145 nice -n 5 scripts/bench-serve-envarms.sh \
  "$DATA/b16" 16 "$DATA/arms/stale-off.env" "$DATA/arms/stale-on.env" \
  || stamp "B=16 sweep FAILED (continuing to B=1)"

stamp "A/B at B=1 (REPEATS=1, predicted near-no-op)"
REPEATS=1 MAX_TOKENS=32 PORT=8145 nice -n 5 scripts/bench-serve-envarms.sh \
  "$DATA/b1" 1 "$DATA/arms/stale-off.env" "$DATA/arms/stale-on.env" \
  || stamp "B=1 sweep FAILED"

stamp "done — compare tok/s, disk_reads and the [prefetch] stale_dropped=/used= split in $DATA"
