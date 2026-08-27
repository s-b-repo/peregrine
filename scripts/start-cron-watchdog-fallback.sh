#!/usr/bin/env bash
# start-cron-watchdog-fallback.sh — Launcher that daemonizes cron-watchdog-loop.sh
# via setsid so it survives beyond the calling session.
set -euo pipefail
export PATH="/home/cortix/.local/cronie/usr/bin:/home/cortix/.local/bin:/home/cortix/.bun/bin:/home/cortix/.cargo/bin:/home/cortix/.opencode/bin:/usr/bin:/usr/local/bin:/bin"
export HOME="/home/cortix"

if pgrep -f "cron-watchdog-loop.sh" >/dev/null 2>&1; then
    echo "cron-watchdog-loop.sh is already running (PID $(pgrep -f 'cron-watchdog-loop.sh'))"
    exit 0
fi

# Only start the fallback when the systemd timer is NOT active.
# Running both simultaneously causes duplicate watchdog executions and
# triplicate log entries (systemd runs the script + cron-watchdog-loop
# runs it with an extra tee). The fallback is meant for when systemd
# timers are unavailable.
if systemctl --user is-active loop-watchdog.timer >/dev/null 2>&1; then
    echo "systemd timer is active — cron-watchdog-loop fallback NOT needed."
    exit 0
fi

# setsid + disown fully detaches the process into its own session
setsid bash /home/cortix/peregrine/scripts/cron-watchdog-loop.sh </dev/null >>/home/cortix/peregrine/auto-improve-loop.log 2>&1 &
disown
echo "cron-watchdog-loop.sh started (PID $!)"
