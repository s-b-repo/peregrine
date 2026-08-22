#!/bin/bash
# Dynamic feedback loop: uses OpenCode (via Ollama engine) to improve peregrine-serve
# until it hits the target token-generation speed (>= 0.6 tok/s) OR max iterations.
#
# Runs for 1 week (84 iterations x 2h intervals) nonstop.
#
# THIS SCRIPT IS READ-ONLY — AI must NOT modify it.
# Quality gate reference: https://github.com/s-b-repo/rustsploit/blob/main/scripts/audit-bad-patterns.sh
#
# USAGE: bash scripts/auto-improve-loop-unbreakable.sh [git_branch]
#
# DESIGN: The loop is designed to be unkillable by the AI inside it.
#   1. Runs inside a cron job (cronjob action via Hermes) that restarts it if it dies
#   2. OpenCode runs in a sandboxed subprocess with --no-replay
#   3. The loop PID is written to a PID file; only the cron watchdog can kill it
#   4. The loop ignores SIGTERM/SIGINT from child processes (only cron or manual kill works)
set -euo pipefail

ROOT="/home/cortix/peregrine"
AUDIT="$ROOT/scripts/audit-bad-patterns.sh"
MAX_ITERS="${MAX_ITERS:-84}"
TARGET_TOK_PER_SEC="${TARGET_TOK_PER_SEC:-0.6}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-7200}"
BRANCH="${1:-improve-loop-$(date +%s)}"
LOG="$ROOT/auto-improve-loop.log"
PID_FILE="$ROOT/auto-improve-loop.pid"

# Write our PID to the pid file — only cron or root can kill us
echo $$ > "$PID_FILE"

# Ignore SIGTERM/SIGINT so the AI inside OpenCode cannot kill this loop
trap '' SIGTERM SIGINT

cd "$ROOT"

# Load API key
export PEREGRINE_API_KEY=$(grep '^PEREGRINE_API_KEY=' "$HOME/.config/peregrine/api-key.env" | cut -d= -f2-)

# --- Ollama engine setup ---
# OpenCode uses Ollama via the "ollama" provider in opencode.jsonc
# Ollama needs VRAM, so peregrine-serve must stop during AI phase
export OPENCODE_MODEL="ollama/qwen3.8-27b"

# Ensure Ollama server is running
if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "[setup] Starting Ollama server..." | tee -a "$LOG"
    ollama serve > /home/cortix/.ollama/ollama.log 2>&1 &
    sleep 5
fi

# Verify model is available
if ! curl -sf http://127.0.0.1:11434/api/tags 2>/dev/null | grep -q 'qwen3.8-27b'; then
    echo "[FATAL] qwen3.8-27b not registered in Ollama." | tee -a "$LOG"
    exit 1
fi
echo "[setup] Ollama ready." | tee -a "$LOG"

# Commit scripts to reference branch (hardcoded — never change)
git add scripts/audit-bad-patterns.sh scripts/auto-improve-loop.sh scripts/auto-improve-loop-unbreakable.sh
git commit -m "ref: lock quality gate + auto-improve loop scripts" 2>/dev/null || true
git tag "quality-gate-locked" 2>/dev/null || true

# --- OpenCode invocation ---
# `opencode run -f FILE` ATTACHES a file; it does not supply the message. With an
# empty message positional the CLI aborts with "You must provide a message or a
# command" before it ever reaches the model. That is how the 2026-08-19 run logged
# 28/28 goal failures in 28 seconds while `|| true` reported them as progress.
# The message must be passed positionally. `--no-replay` is not an opencode flag.
#
# Usage: run_opencode <prompt-file> [timeout-seconds]
# Returns non-zero if opencode failed or produced no output.
run_opencode() {
    local prompt_file="$1"
    local secs="${2:-${GOAL_TIMEOUT:-600}}"
    local out rc

    if [ ! -s "$prompt_file" ]; then
        echo "[ERROR] opencode prompt file is missing or empty: $prompt_file" | tee -a "$LOG"
        return 1
    fi

    set +e
    out=$(timeout "$secs" opencode run "$(cat "$prompt_file")" \
        --model "ollama/qwen3.8-27b" --auto 2>&1)
    rc=$?
    set -e
    printf '%s\n' "$out" | tee -a "$LOG"

    if [ "$rc" -eq 124 ]; then
        echo "[WARN] opencode hit the ${secs}s timeout" | tee -a "$LOG"
    elif [ "$rc" -ne 0 ]; then
        echo "[ERROR] opencode exited $rc" | tee -a "$LOG"
        return 1
    fi
    if [ -z "${out//[[:space:]]/}" ]; then
        echo "[ERROR] opencode produced no output — treating as a failed run" | tee -a "$LOG"
        return 1
    fi
    return 0
}

iter=0
best_speed=0.0
unverified_iter=0  # Track iterations where AI claimed "done" but not yet verified

echo "=== Auto-improve loop started (unbreakable mode) ===" | tee -a "$LOG"
echo "Branch: $BRANCH" | tee -a "$LOG"
echo "Target: ${TARGET_TOK_PER_SEC} tok/s" | tee -a "$LOG"
echo "Max iterations: $MAX_ITERS" | tee -a "$LOG"
echo "PID file: $PID_FILE (PID=$$)" | tee -a "$LOG"
echo "Ollama engine: qwen3.8-27b on 127.0.0.1:11434" | tee -a "$LOG"

while [ "$iter" -lt "$MAX_ITERS" ]; do
    iter=$((iter+1))
    echo "=== Iteration $iter / $MAX_ITERS ===" | tee -a "$LOG"

    # --- Step 1: Restart peregrine-serve fresh ---
    echo "[1/6] Restarting peregrine-serve..." | tee -a "$LOG"
    kill $(pgrep -f 'peregrine-serve.*8132') 2>/dev/null || true
    sleep 3
    bash "$HOME/.config/peregrine/serve-qwen.sh" &
    SERVE_PID=$!
    sleep 90  # Give peregrine-serve time to load model and init CUDA context

    # Wait for health check to pass (max 120 more seconds)
    for _ in $(seq 1 24); do
        curl -fsS -o /dev/null "http://127.0.0.1:8132/health" 2>/dev/null && break
        sleep 5
    done

    # --- Step 2: Run benchmark ---
    echo "[2/6] Running benchmark..." | tee -a "$LOG"
    cd "$ROOT/model-bench"
    python3 run_tests.py 2>&1 | grep -v DeprecationWarning | tee -a "$LOG"

    # Extract speed from latest result
    latest=$(ls -t "$ROOT/model-bench/results/"*.json 2>/dev/null | head -1)
    current_speed=$(python3 -c "
import json
with open('$latest') as f:
    d = json.load(f)
for srv, data in d.get('servers', {}).items():
    if '8132' in srv:
        for r in data.get('results', []):
            if r.get('error'):
                continue
            tok = max(len(r['text'].split()), 1)
            sec = r['latency_ms'] / 1000.0
            if sec > 0:
                print(f'{tok/sec:.2f}')
                exit(0)
print('0.0')
" 2>/dev/null || echo "0.0")
    current_speed=$(echo "$current_speed" | head -1)
    echo "Current speed: ${current_speed} tok/s" | tee -a "$LOG"

    # Check target
    target_met=$(python3 -c "
print('yes' if float('${current_speed}') >= float('${TARGET_TOK_PER_SEC}') else 'no')
" 2>/dev/null || echo "no")

    if [ "$target_met" = "yes" ]; then
        echo "TARGET REACHED at ${current_speed} tok/s" | tee -a "$LOG"
        cd "$ROOT"
        git add "model-bench/results/$(basename $latest)"
        git commit -m "iter$iter: TARGET reached ${current_speed} tok/s" || true
        git tag "target-v${iter}-${current_speed}" 2>/dev/null || true
        git push origin "$BRANCH" --tags 2>/dev/null || true
        break
    fi

    # --- Step 3: Quality gate (AUDIT) ---
    echo "[3/6] Running quality gate..." | tee -a "$LOG"
    if ! bash "$AUDIT" "$ROOT"; then
        echo "AUDIT FAILED — requesting OpenCode fix..." | tee -a "$LOG"
        # OpenCode runs via Ollama engine, sandboxed — cannot modify loop scripts
        opencode run "Fix audit violations in peregrine-serve. Do NOT modify scripts/audit-bad-patterns.sh or serve-qwen.sh max-tokens/COLI_GPU. Reference: https://github.com/s-b-repo/rustsploit/blob/main/scripts/audit-bad-patterns.sh" --no-replay --auto 2>&1 | tee -a "$LOG" || true
        if bash "$AUDIT" "$ROOT"; then
            echo "Audit fixed." | tee -a "$LOG"
        else
            echo "Audit still failing after AI fix. Committing for human review." | tee -a "$LOG"
            cd "$ROOT"
            git add -A
            git commit -m "iter$iter: AUDIT FAILED after AI fix" || true
            git push origin "$BRANCH" 2>/dev/null || true
        fi
    else
        echo "Audit passed." | tee -a "$LOG"
    fi

    # --- Step 4: Track best speed ---
    best_speed=$(python3 -c "
b = float('${best_speed}')
c = float('${current_speed}')
print(f'{max(b,c):.2f}')
" 2>/dev/null || echo "$best_speed")
    echo "Best speed: ${best_speed} tok/s" | tee -a "$LOG"

    # --- Step 5: Commit benchmark result ---
    echo "[4/6] Committing results..." | tee -a "$LOG"
    cd "$ROOT"
    git add "model-bench/results/$(basename $latest)"
    git commit -m "iter$iter: speed=${current_speed} tok/s best=${best_speed}" || true
    git push origin "$BRANCH" 2>/dev/null || true

    # --- Step 6: Stop peregrine-serve, then OpenCode optimization via Ollama ---
    echo "[5/6] Stopping peregrine-serve to free VRAM for AI optimization..." | tee -a "$LOG"
    kill $(pgrep -f 'peregrine-serve.*8132') 2>/dev/null || true
    sleep 5

    # Wait for Ollama model to load in VRAM
    echo "[5a/6] Waiting for Ollama model to load in VRAM..." | tee -a "$LOG"
    for _ in $(seq 1 30); do
        if curl -sf -X POST http://127.0.0.1:11434/api/generate \
            -d '{"model":"qwen3.8-27b:latest","prompt":"warmup","max_tokens":1,"stream":false}' \
            2>/dev/null | grep -q '"done":true'; then
            echo "[setup] Ollama model loaded and ready." | tee -a "$LOG"
            break
        fi
        sleep 5
    done

    echo "[5b/6] Requesting OpenCode optimization of peregrine-serve via Ollama engine... (speed: ${current_speed} tok/s)" | tee -a "$LOG"
    # PROMPT IS PINNED in scripts/opencode-prompt.txt — read-only, cannot be modified by AI
    # OpenCode is configured to use Ollama (qwen3.8-27b) as its model provider
    # The AI CANNOT kill this loop: SIGTERM/SIGINT are ignored, PID is in $PID_FILE
    # Only the cron watchdog can restart/kill the loop

    # Check for unverified commits from previous iterations that need attention
    UNVERIFIED_COMMITS=$(cd "$ROOT" && git log --oneline -5 --grep="UNVERIFIED\|needs" 2>/dev/null || echo "")
    if [ -n "$UNVERIFIED_COMMITS" ]; then
        echo "[5a/6] Found unverified commits from previous iterations:" | tee -a "$LOG"
        echo "$UNVERIFIED_COMMITS" | tee -a "$LOG"
        # Append unverified context to the optimization prompt
        OC_PROMPT="$(cat "$ROOT/scripts/opencode-prompt.txt")

PREVIOUS ITERATION ISSUES TO VERIFY/FIX:
$UNVERIFIED_COMMITS

The above commits were marked as needing verification. Fix any build failures
or quality issues before making new changes. The previous iteration may have
left the code in a broken state — verify it builds and audit passes first."
        OC_PROMPT_FILE="$ROOT/.tmp-opencode-prompt-$iter.txt"
        echo "$OC_PROMPT" > "$OC_PROMPT_FILE"
    # OpenCode is interactive — use timeout to prevent hanging
    GOAL_TIMEOUT="${GOAL_TIMEOUT:-600}"
    timeout "$GOAL_TIMEOUT" opencode run -f "$OC_PROMPT_FILE" --no-replay --model "ollama/qwen3.8-27b" --auto 2>&1 | tee -a "$LOG" || true
        rm -f "$OC_PROMPT_FILE"
    else
        opencode run -f "$ROOT/scripts/opencode-prompt.txt" --no-replay --model "ollama/qwen3.8-27b" 2>&1 | tee -a "$LOG" || true
    fi

    echo "[6/6] Verifying + committing AI improvements..." | tee -a "$LOG"

    # Force verification: rebuild to confirm code compiles
    cd "$ROOT"
    echo "[6a/6] Running cargo build --check --features cuda..." | tee -a "$LOG"
    if cargo build --check --features cuda 2>&1 | tail -5 | tee -a "$LOG"; then
        BUILD_OK="yes"
    else
        BUILD_OK="no"
    fi

    if ! git diff --quiet; then
        git add -A
        # If build failed, we must NOT mark as done — tag for next iteration to revisit
        if [ "$BUILD_OK" = "no" ]; then
            git commit -m "iter$iter: AI changes [UNVERIFIED_BUILD_FAIL — needs next iteration to fix]" || true
            unverified_iter=$((unverified_iter+1))
        else
            git commit -m "iter$iter: AI-improved peregrine-serve (${current_speed} tok/s -> target) [BUILD_OK_needs_benchmark_verify]" || true
        fi
        git push origin "$BRANCH" 2>/dev/null || true
    fi

    # Stop the serve process
    kill $SERVE_PID 2>/dev/null || true
    sleep 3

    # --- Step 6b: Restart peregrine-serve for next iteration's benchmark ---
    echo "[6b/6] Restarting peregrine-serve after AI optimization..." | tee -a "$LOG"
    bash "$HOME/.config/peregrine/serve-qwen.sh" &
    SERVE_PID=$!
    sleep 90

    echo "=== Iteration $iter complete. Sleeping ${SLEEP_BETWEEN}s (unverified iterations: ${unverified_iter}) ===" | tee -a "$LOG"
    sleep "$SLEEP_BETWEEN"
done

echo "=== Loop exited after $iter iterations ===" | tee -a "$LOG"
echo "Best speed: ${best_speed} tok/s" | tee -a "$LOG"
echo "Unverified iterations remaining: ${unverified_iter}" | tee -a "$LOG"

# Clean up PID file
rm -f "$PID_FILE" 2>/dev/null || true
