#!/usr/bin/env bash
# Phase 3: jobs enabled after phase-2 globbed its list (96 specconf-sweep, 98
# gap A/B, 99 topic A/B — all unlocked once job 20 confirmed spec-conf). Waits
# on phase-2's "phase 2 drained" sentinel, then runs any jobs/*.sh without a log.
set -u
cd "$(dirname "$0")/.."
QDIR=bench-data/2026-08-15-queue
P2_LOG=bench-queue-phase2-2026-08-15.log
stamp() { echo "== [$(date '+%F %T')] $*"; }
stamp "waiting on phase 2 ('phase 2 drained' in $P2_LOG)"
while :; do
  [ -e "$QDIR/SKIP" ] && { stamp "SKIP present — exiting"; exit 0; }
  grep -q 'phase 2 drained' "$P2_LOG" 2>/dev/null && break
  pgrep -f 'bench-queue-phase2-2026-08-15.sh' >/dev/null \
    || { stamp "phase-2 runner gone without its sentinel — treating box as free"; break; }
  sleep 300
done
shopt -s nullglob
for job in "$QDIR"/jobs/*.sh; do
  log="$QDIR/$(basename "$job" .sh).log"
  [ -e "$log" ] && continue
  [ -e "$QDIR/SKIP" ] && { stamp "SKIP present — stopping"; exit 0; }
  stamp "running $job"
  bash "$job" > "$log" 2>&1 || stamp "$job FAILED (continuing)"
done
stamp "phase 3 drained"
