#!/usr/bin/env bash
# Serialized bench queue for the post-chain confirmation wave (2026-08-15).
#
# One measurement owns the box at a time. This runner waits until BOTH the
# overnight chain and the queued stale-drop A/B are finished (sentinel or dead
# process), then executes bench-data/2026-08-15-queue/jobs/*.sh in lexical
# order. Jobs are enabled by moving them in from jobs-available/ — only arms
# whose REPEATS=1 screen came back positive earn a REPEATS=3 confirmation slot.
#
#   setsid nohup scripts/bench-queue-2026-08-15.sh > bench-queue-2026-08-15.log 2>&1 &
#
# Cancel with: touch bench-data/2026-08-15-queue/SKIP
# A job that fails logs and yields to the next one (a lost slot is a rerun).
set -u
cd "$(dirname "$0")/.."
QDIR=bench-data/2026-08-15-queue
CHAIN_LOG=overnight-2026-08-15.log
STALE_LOG=stale-ab-2026-08-15.log
stamp() { echo "== [$(date '+%F %T')] $*"; }

wait_free() { # $1 sentinel-log  $2 sentinel-text  $3 pgrep-pattern
  while :; do
    [ -e "$QDIR/SKIP" ] && return 1
    grep -q "$2" "$1" 2>/dev/null && return 0
    pgrep -f "$3" >/dev/null \
      || { stamp "$3 gone without '$2' — treating the box as free"; return 0; }
    sleep 300
  done
}

stamp "waiting on the overnight chain"
wait_free "$CHAIN_LOG" 'chain complete' 'overnight-2026-08-15.sh' \
  || { stamp "SKIP present — exiting"; exit 0; }
stamp "waiting on the stale-drop A/B"
wait_free "$STALE_LOG" '^== .*done —' 'prefetch-stale-ab.sh' \
  || { stamp "SKIP present — exiting"; exit 0; }

shopt -s nullglob
jobs=("$QDIR"/jobs/*.sh)
stamp "box is free; ${#jobs[@]} enabled job(s)"
for job in "${jobs[@]}"; do
  [ -e "$QDIR/SKIP" ] && { stamp "SKIP present — stopping before $job"; exit 0; }
  stamp "running $job"
  bash "$job" > "$QDIR/$(basename "$job" .sh).log" 2>&1 \
    || stamp "$job FAILED (continuing; see its log)"
done
stamp "queue drained"
