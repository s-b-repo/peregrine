#!/usr/bin/env bash
# cron-watchdog-loop.sh — User-space cron replacement for loop-watchdog.
#
# Since this environment lacks root access to install/start crond (cronie's
# crond requires root for /var/run/crond.pid and /var/spool/cron), this
# script acts as a mini-cron: it runs loop-watchdog every 2 minutes and
# self-terminates after the 1-week window (using the same START_MARKER
# the watchdog script manages).
#
# This complements (not replaces) the systemd user timer
# loop-watchdog.timer, which is the primary scheduling mechanism.
set -euo pipefail

ROOT="/home/cortix/peregrine"
LOG="$ROOT/auto-improve-loop.log"
PID_FILE="$ROOT/cron-watchdog-loop.pid"
START_MARKER="$ROOT/.watchdog-start.time"
ONE_WEEK_SECONDS=604800
INTERVAL_SECONDS=120

export PATH="/home/cortix/.local/bin:/home/cortix/.bun/bin:/home/cortix/.cargo/bin:/home/cortix/.opencode/bin:/home/cortix/.local/cronie/usr/bin:/usr/bin:/usr/local/bin:/bin"
export HOME="/home/cortix"

# Write our own PID file so we can be managed/killed
echo $$ > "$PID_FILE"
chmod 600 "$PID_FILE" 2>/dev/null || true

ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }
log() { echo "[$(ts)] [cron-repl] $*" | tee -a "$LOG"; }

log "=== Cron-replacement loop started (PID $$), interval=${INTERVAL_SECONDS}s ==="

cleanup() {
    log "Cron-replacement loop exiting (signal received)."
    rm -f "$PID_FILE" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

while true; do
    # ── 1-week window check (same marker the watchdog uses) ──────────────
    now=$(date +%s)
    if [ -f "$START_MARKER" ]; then
        start=$(cat "$START_MARKER" 2>/dev/null || echo "$now")
    else
        echo "$now" > "$START_MARKER"
        start="$now"
    fi
    elapsed=$((now - start))
    if [ "$elapsed" -ge "$ONE_WEEK_SECONDS" ]; then
        log "Cron-replacement: 1-week window complete (elapsed=${elapsed}s). Exiting."
        # Also ensure systemd timer is cleaned up
        systemctl --user stop loop-watchdog.timer 2>/dev/null || true
        systemctl --user disable loop-watchdog.timer 2>/dev/null || true
        # Clean up crontab entry if crontab is available
        if command -v crontab >/dev/null 2>&1; then
            crontab -l 2>/dev/null | grep -v 'loop-watchdog\.sh' | crontab - 2>/dev/null || true
            log "Cron-replacement: crontab entry removed."
        fi
        rm -f "$START_MARKER" 2>/dev/null || true
        break
    fi

    # ── Run the watchdog ────────────────────────────────────────────────
    # If the systemd timer is the active primary scheduler, skip running the
    # watchdog here to avoid duplicate executions and log duplication.
    # loop-watchdog.sh already writes to $LOG via its internal `tee -a`, so we
    # must NOT pipe through an outer `tee -a` (that would double every line).
    if systemctl --user is-active loop-watchdog.timer >/dev/null 2>&1; then
        log "Cron-replacement: systemd timer active — skipping watchdog run (heartbeat only)."
    else
        "$ROOT/scripts/loop-watchdog.sh" >> "$LOG" 2>&1 || true
    fi

    # ── Sleep for the interval, but check for early exit every 5s ─────────
    remaining=$INTERVAL_SECONDS
    while [ "$remaining" -gt 0 ]; do
        sleep 5
        remaining=$((remaining - 5))
        # Allow quick exit if PID file is removed (external shutdown signal)
        if [ ! -f "$PID_FILE" ]; then
            log "PID file removed — exiting early."
            break 2
        fi
    done
done
