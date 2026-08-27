#!/usr/bin/env bash
# crontab-backup.sh — Lightweight background loop that runs the watchdog
# every 2 minutes as a fallback when neither cron nor systemd timers are
# available.  It reads /home/cortix/peregrine/scripts/loop-watchdog.sh
# and invokes it, logging to the same auto-improve-loop.log file.
set -euo pipefail

WATCHDOG="/home/cortix/peregrine/scripts/loop-watchdog.sh"
LOG="/home/cortix/peregrine/auto-improve-loop.log"
INTERVAL=120   # 2 minutes

export PATH="/home/cortix/.local/bin:/home/cortix/.bun/bin:/home/cortix/.cargo/bin:/home/cortix/.opencode/bin:/usr/bin:/usr/local/bin:/bin"
export HOME="/home/cortix"

ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }

echo "[$(ts)] [crontab-backup] Starting 2-minute watchdog loop (PID $$)." >> "$LOG"

while true; do
    bash "$WATCHDOG" >> "$LOG" 2>&1 || echo "[$(ts)] [crontab-backup] Watchdog exited with error" >> "$LOG"
    sleep "$INTERVAL"
done
