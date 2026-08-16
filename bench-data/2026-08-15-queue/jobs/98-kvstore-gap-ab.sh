#!/usr/bin/env bash
# Job 97 — kvstore async-writer latency A/B (COLI_KV_STORE_SYNC=1 vs unset).
#
# What it confirms: baf6295 moved checkpoint serialize+fsync off the engine
# thread. The claim is not tok/s (saves are rare per token) but the stall
# tail: with the synchronous path, every sequence retirement (~0.3-2 s of
# write+fsync) freezes every OTHER stream's next token. Primary metric:
# pooled p95/p99 inter-token gap and worst-stream p95 from
# scripts/bench-serve-gaps.py; [kvstore] saved=/dropped_busy= archived per arm.
# Prediction: async arm's p99 collapses toward its p50; saved= equal in both
# arms (dropped_busy may be nonzero in the async arm — that is the trade
# working as designed, each drop is a stall the sync arm would have taken).
#
# DEPENDS ON: branch wt-kvstore-gap merged (COLI_KV_STORE_SYNC knob + the gap
# client). This job rebuilds target/release itself at start — legal because
# the queue serializes jobs, so nothing is mid-sweep during it.
#
# Cost estimate (be honest when slotting): 6 boots (3 reps x 2 arms, rotated)
# + per round a 16-stream prefill of ~550-token prompts + 48 decode ticks —
# roughly 6-9 h on this box. Rank below tok/s levers; run when the queue is
# otherwise idle.
set -uo pipefail
cd "$(dirname "$0")/../../.."

MODEL=${COLI_MODEL:-/home/cortix/models/GLM-5.2}
BIN=target/release/peregrine-serve
PORT=${PORT:-8149}
MEMMAX=${MEMMAX:-34G}
BOOT_TIMEOUT=${BOOT_TIMEOUT:-1800}
OUT=bench-data/2026-08-15-kvstore-gap
mkdir -p "$OUT"
stamp() { echo "== [$(date '+%F %T')] $*"; }

stamp "job 97: rebuilding release from the current tree (queue-serialized, so no sweep is live)"
cargo build --release > "$OUT/build.log" 2>&1 || { stamp "build FAILED"; exit 1; }
grep -q "COLI_KV_STORE_SYNC" crates/peregrine-serve/src/kvstore.rs \
  || { stamp "tree lacks the sync knob (wt-kvstore-gap not merged) — aborting"; exit 1; }

PROMPT="$OUT/prompt.txt"
head -c 2500 bench-data/2026-08-13-route-min-share/corpus.txt > "$PROMPT"

run_arm() { # $1 = arm name (async|sync), $2 = rep
    local arm=$1 rep=$2
    local store="$OUT/store-$arm-$rep"
    rm -rf "$store"; mkdir -p "$store"
    local sync_env=()
    [ "$arm" = sync ] && sync_env=(COLI_KV_STORE_SYNC=1)
    local blog="$OUT/$arm-rep$rep.server.log"
    (
        env "${sync_env[@]}" COLI_KV_STORE_DIR="$store" COLI_MODEL="$MODEL" \
            COLI_ROUTE_STATS_PERSIST=0 COLI_DEBUG=1 MALLOC_ARENA_MAX=2 \
            systemd-run --user --scope -q -p MemoryMax="$MEMMAX" -p MemorySwapMax=0 \
            "$BIN" --model "$MODEL" --port "$PORT" --max-batch 16
    ) >"$blog" 2>&1 &
    local pid=$!
    local waited=0
    until curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
        kill -0 "$pid" 2>/dev/null || { stamp "$arm rep$rep: server died in boot"; return 1; }
        sleep 2; waited=$((waited+2))
        [ "$waited" -ge "$BOOT_TIMEOUT" ] && { stamp "$arm rep$rep: boot timeout"; kill -INT "$pid"; return 1; }
    done
    stamp "$arm rep$rep: serving after ~${waited}s; driving 16 gap streams"
    python3 scripts/bench-serve-gaps.py \
        --url "http://127.0.0.1:$PORT/v1/chat/completions" \
        -n 16 --prompt-file "$PROMPT" --tag "$arm-r$rep" --max-tokens 48 \
        > "$OUT/$arm-rep$rep.json" 2> "$OUT/$arm-rep$rep.client.log" \
        || stamp "$arm rep$rep: client FAILED"
    kill -INT "$pid" 2>/dev/null; wait "$pid" 2>/dev/null || true
    grep -E '^\[(kvstore|prefetch|ecache|lane)\]' "$blog" > "$OUT/$arm-rep$rep.counters.txt" || true
}

# Rotate arm order across reps, envarms-style, so drift cancels.
for rep in 1 2 3; do
    if [ $((rep % 2)) -eq 1 ]; then order="async sync"; else order="sync async"; fi
    for arm in $order; do run_arm "$arm" "$rep" || stamp "arm $arm rep $rep lost (continuing)"; done
done
stamp "job 97 done — compare gap_pooled_s.p95/.p99 and gap_worst_stream_p95_s across arms in $OUT"
