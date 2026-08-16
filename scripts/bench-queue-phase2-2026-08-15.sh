#!/usr/bin/env bash
# Phase 2 of the 2026-08-15 bench queue: the phase-1 runner globbed jobs/ once
# at launch, so jobs enabled later (90 rebuild+smoke, 93 ecache rerun, 95
# device-sched A/B) queue here instead. Waits for phase 1's "queue drained"
# sentinel (or its dead pid), then runs every jobs/*.sh that has no log yet,
# in lexical order, same semantics as phase 1.
#
#   setsid nohup scripts/bench-queue-phase2-2026-08-15.sh > bench-queue-phase2-2026-08-15.log 2>&1 &
#
# Cancel with: touch bench-data/2026-08-15-queue/SKIP
set -u
cd "$(dirname "$0")/.."
QDIR=bench-data/2026-08-15-queue
P1_LOG=bench-queue-2026-08-15.log
stamp() { echo "== [$(date '+%F %T')] $*"; }

stamp "waiting on phase 1 ('queue drained' in $P1_LOG)"
while :; do
  [ -e "$QDIR/SKIP" ] && { stamp "SKIP present — exiting"; exit 0; }
  grep -q 'queue drained' "$P1_LOG" 2>/dev/null && break
  pgrep -f 'bench-queue-2026-08-15.sh' >/dev/null \
    || { stamp "phase-1 runner gone without its sentinel — treating the box as free"; break; }
  sleep 300
done

shopt -s nullglob
for job in "$QDIR"/jobs/*.sh; do
  log="$QDIR/$(basename "$job" .sh).log"
  [ -e "$log" ] && continue # phase 1 already ran it
  [ -e "$QDIR/SKIP" ] && { stamp "SKIP present — stopping before $job"; exit 0; }
  stamp "running $job"
  bash "$job" > "$log" 2>&1 || stamp "$job FAILED (continuing; see its log)"
done
stamp "phase 2 drained"
