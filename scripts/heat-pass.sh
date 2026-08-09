#!/usr/bin/env bash
# Produce the routing-heat profile that `peregrine-requantize --tier-hot-frac`
# needs, and which the checkpoint does not currently have.
#
# `route_stats.json` in the model directory carries `"heat": null`. The reason is
# in `model.rs:2175`:
#
#     let heat = gpu.as_ref().map(|_| HeatTable::new(...));
#
# The heat table is built **only when the GPU tier is built**, i.e. only under
# `COLI_GPU=1`. So the data that drives a purely storage-side decision — which
# experts stay int4 and which drop to int2 — can only be produced by a GPU run.
# That coupling is incidental, not a design intent worth relying on; noted here
# because it is the reason this script exists rather than a plain serve run.
#
# `bench-prefetch-arms.sh` cannot be reused: it sets COLI_ROUTE_STATS_PERSIST=0
# on purpose, so arms do not warm each other through this very file.
#
#   usage: scripts/heat-pass.sh [tokens] [out-dir]
#
# Heat is a *ranking* over 19 200 experts (256/layer x 75 sparse layers) and each
# token samples only top-8 of 256 per layer, so a short run ranks almost nothing.
# Run it short once to prove the mechanism writes a non-null array, then long for
# a profile worth tiering on.
set -uo pipefail

MODEL=${COLI_MODEL:-/home/cortix/models/GLM-5.2-colibri-int4-with-int8-mtp}
BIN=${BIN_SERVE:-target/release/peregrine-serve}
PORT=${PORT:-8139}
TOKENS=${1:-1}
OUT=${2:-/tmp/heat-pass}
MEMMAX=${MEMMAX:-34G}

mkdir -p "$OUT"
[ -x "$BIN" ] || { echo "no server binary at $BIN (cargo build --release --bins)" >&2; exit 1; }

# Varied prompts: heat is a frequency ranking, and one prompt's routing is not a
# sample of the model's routing. Cycled if TOKENS demands more turns than these.
PROMPTS=(
  "Explain how a mixture-of-experts layer routes a token."
  "Write a short Rust function that reverses a linked list."
  "Summarise the causes of the 1929 financial crash."
  "What is the difference between a semaphore and a mutex?"
  "Describe photosynthesis to a ten-year-old."
)

echo "== heat pass: $TOKENS tokens/request, model $MODEL"
echo "== route_stats.json will be REWRITTEN at shutdown (back it up first)"

env COLI_MODEL="$MODEL" \
    COLI_GPU=1 \
    COLI_ROUTE_STATS_PERSIST=1 \
    COLI_MEMO_ENTRIES=0 \
    COLI_DEBUG=1 \
    MALLOC_ARENA_MAX=2 \
    systemd-run --user --scope -q -p MemoryMax="$MEMMAX" -p MemorySwapMax=0 \
    "$BIN" --model "$MODEL" --port "$PORT" >"$OUT/serve.log" 2>&1 &
pid=$!

waited=0
until curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    kill -0 "$pid" 2>/dev/null || { echo "!! exited during load:" >&2; tail -20 "$OUT/serve.log" >&2; exit 1; }
    sleep 2; waited=$((waited + 2))
    [ "$waited" -ge 900 ] && { echo "!! not healthy in 900s" >&2; kill -INT "$pid"; exit 1; }
done
echo "== loaded in ${waited}s"

# Every request pays a full prefill (~5 000 lookups), which dominates a
# low-token run, so the verification pass wants NPROMPTS=1.
NPROMPTS=${NPROMPTS:-${#PROMPTS[@]}}
i=0
for p in "${PROMPTS[@]:0:$NPROMPTS}"; do
    i=$((i + 1))
    t0=$SECONDS
    curl -s -m 7200 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"glm-5.2\",\"messages\":[{\"role\":\"user\",\"content\":$(printf '%s' "$p" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')}],\"max_tokens\":$TOKENS,\"temperature\":0.0}" \
        >"$OUT/req-$i.json"
    echo "== request $i done in $((SECONDS - t0))s"
done

# SIGINT, not SIGKILL: route_stats.json is written at Model::drop, which a hard
# kill skips entirely — the same defect main.rs:1045 documents for the counters.
echo "== SIGINT (route_stats.json is written at Drop)"
kill -INT "$pid" 2>/dev/null
wait "$pid" 2>/dev/null

python3 - "$MODEL/route_stats.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
h = d.get("heat")
if h is None:
    print("RESULT: heat is still null — the GPU tier did not build. Check serve.log for a CUDA fallback.")
    sys.exit(1)
n = len(h) if isinstance(h, list) else -1
flat = [x for row in h for x in row] if (n > 0 and isinstance(h[0], list)) else h
nz = sum(1 for x in flat if x)
print(f"RESULT: heat present — {n} rows, {len(flat)} entries, {nz} non-zero ({100.0*nz/max(len(flat),1):.1f}% of experts seen)")
PY
