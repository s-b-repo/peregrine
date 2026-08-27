#!/usr/bin/env bash
# Quick status check for the peregrine watchdog system
set -uo pipefail
LOG="/home/cortix/peregrine/auto-improve-loop.log"
PID_FILE="/home/cortix/peregrine/auto-improve-loop.pid"
START_MARKER="/home/cortix/peregrine/.watchdog-start.time"

echo "=== 1. SYSTEMD TIMER (Primary) ==="
systemctl --user is-active loop-watchdog.timer 2>&1
systemctl --user is-enabled loop-watchdog.timer 2>&1
echo "Schedule: every 2 minutes (OnCalendar=*:0/2)"
echo "Next trigger:" $(systemctl --user list-timers 2>&1 | grep loop-watchdog | awk '{print $4, $5}')
echo ""

echo "=== 2. CRON-WATCHDOG-LOOP (Fallback) ==="
if pgrep -f "cron-watchdog-loop.sh" >/dev/null 2>&1; then
    PID=$(pgrep -f 'cron-watchdog-loop.sh')
    echo "RUNNING (PID $PID)"
    ps -p "$PID" -o pid,etime,args --no-headers 2>&1
else
    echo "NOT RUNNING"
fi
echo ""

echo "=== 3. AUTO-IMPROVE LOOP ==="
if [ -f "$PID_FILE" ]; then
    LP=$(cat "$PID_FILE" | tr -d '[:space:]')
    if kill -0 "$LP" 2>/dev/null; then
        echo "RUNNING (PID $LP)"
        ps -p "$LP" -o pid,etime,args --no-headers 2>&1
    else
        echo "DEAD (PID $LP in file but process not alive)"
    fi
else
    echo "No PID file"
fi
echo ""

echo "=== 4. OLLAMA ==="
curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && echo "UP (:11434)" || echo "DOWN"
echo ""

echo "=== 5. 1-WEEK WINDOW ==="
start=$(cat "$START_MARKER" 2>/dev/null || echo 0)
now=$(date +%s)
elapsed=$((now - start))
remaining=$((604800 - elapsed))
echo "Started: $(date -d @$start '+%Y-%m-%d %H:%M:%S %Z' 2>/dev/null)"
echo "Elapsed: $((elapsed/3600))h $((elapsed%3600/60))m"
echo "Remaining: ~$((remaining/86400)) days"
echo ""

echo "=== 6. RECENT LOG (last 8 lines) ==="
tail -8 "$LOG"
echo ""

echo "=== 7. SUMMARY ==="
SYSTEMD_ACTIVE=$(systemctl --user is-active loop-watchdog.timer 2>/dev/null)
CRON_REPL=$(pgrep -f "cron-watchdog-loop.sh" >/dev/null 2>&1 && echo "ACTIVE" || echo "INACTIVE")
echo "Primary (systemd timer):  $SYSTEMD_ACTIVE"
echo "Fallback (cron-watchdog): $CRON_REPL"
echo "Loop PID file exists:    $([ -f "$PID_FILE" ] && echo yes || echo no)"
echo "Log file:                $LOG"
