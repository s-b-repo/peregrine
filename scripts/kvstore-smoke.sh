#!/usr/bin/env bash
# End-to-end smoke for the disk-persisted KV session store (COLI_KV_STORE_DIR):
# serve, send one long-prompt request, restart the server, send the identical
# request, and demand (a) byte-identical output text and (b) a [kvstore] load
# on the second boot. The identity check is the point — a disk-restored KV that
# changes even one token is a correctness bug, not a slow path.
#
#   usage: scripts/kvstore-smoke.sh <out-dir>
#   env:   MODEL BIN_SERVE PORT MEMMAX BOOT_TIMEOUT MAX_TOKENS
#
# Server lifecycle copied from bench-serve-envarms.sh: systemd scope with
# MemoryMax so an OOM is contained, SIGINT (never SIGKILL) so the shutdown
# counter lines this script greps for are actually printed.
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL=${COLI_MODEL:-/home/cortix/models/GLM-5.2}
BIN=${BIN_SERVE:-target/release/peregrine-serve}
PORT=${PORT:-8147}
MEMMAX=${MEMMAX:-34G}
BOOT_TIMEOUT=${BOOT_TIMEOUT:-1800}
MAX_TOKENS=${MAX_TOKENS:-8}
CLIENT_TIMEOUT=${CLIENT_TIMEOUT:-10800}

OUT=${1:?usage: kvstore-smoke.sh <out-dir>}
mkdir -p "$OUT"
STORE="$OUT/kvstore"
rm -rf "$STORE"
mkdir -p "$STORE"

# A deterministic long prompt (well past the 256-token save floor): the same
# corpus the flip-rate gate uses, so tokenization is stable across runs.
PROMPT_FILE="$OUT/prompt.txt"
head -c 6000 bench-data/2026-08-13-route-min-share/corpus.txt > "$PROMPT_FILE"

start_server() { # $1 = boot log
    (
        COLI_KV_STORE_DIR="$STORE" COLI_MODEL="$MODEL" COLI_ROUTE_STATS_PERSIST=0 \
        COLI_DEBUG=1 MALLOC_ARENA_MAX=2 \
        exec systemd-run --user --scope -q -p MemoryMax="$MEMMAX" -p MemorySwapMax=0 \
            "$BIN" --model "$MODEL" --port "$PORT" --max-batch 1
    ) >"$1" 2>&1 &
    SERVER_PID=$!
    local waited=0
    until curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "!! server exited during load; tail of $1:" >&2
            tail -5 "$1" >&2
            return 1
        fi
        sleep 2
        waited=$((waited + 2))
        [ "$waited" -ge "$BOOT_TIMEOUT" ] && { echo "!! not healthy in ${BOOT_TIMEOUT}s" >&2; kill -INT "$SERVER_PID"; return 1; }
    done
    echo "loaded in ~${waited}s"
}

stop_server() {
    kill -INT "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null || true
}

ask() { # $1 = response json path; prints wall seconds
    python3 - "$PROMPT_FILE" "$1" "$PORT" "$MAX_TOKENS" "$CLIENT_TIMEOUT" <<'PY'
import json, sys, time, urllib.request
prompt_file, out, port, max_tokens, timeout = sys.argv[1:6]
with open(prompt_file, encoding="utf-8", errors="replace") as f:
    prompt = f.read()
body = json.dumps({
    "model": "peregrine",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": int(max_tokens),
    "temperature": 0,
}).encode()
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=body, headers={"Content-Type": "application/json"})
t0 = time.monotonic()
with urllib.request.urlopen(req, timeout=float(timeout)) as r:
    resp = json.load(r)
secs = time.monotonic() - t0
with open(out, "w") as f:
    json.dump(resp, f, indent=2)
print(f"{secs:.1f}")
PY
}

echo "== kvstore smoke: cold run"
start_server "$OUT/server-cold.log" || exit 1
cold_s=$(ask "$OUT/resp-cold.json") || { stop_server; exit 1; }
stop_server
echo "cold request: ${cold_s}s"

echo "== kvstore smoke: restart run (same request)"
start_server "$OUT/server-warm.log" || exit 1
warm_s=$(ask "$OUT/resp-warm.json") || { stop_server; exit 1; }
stop_server
echo "warm request: ${warm_s}s (cold was ${cold_s}s)"

grep '^\[kvstore\]' "$OUT/server-cold.log" "$OUT/server-warm.log" || true

python3 - "$OUT/resp-cold.json" "$OUT/resp-warm.json" <<'PY'
import json, sys
a, b = (json.load(open(p))["choices"][0]["message"]["content"] for p in sys.argv[1:3])
if a != b:
    print(f"MISMATCH:\n  cold: {a!r}\n  warm: {b!r}")
    sys.exit(1)
print(f"outputs identical across the restart ({len(a)} chars)")
PY
rc=$?
saved=$(ls "$STORE"/*.pgkv 2>/dev/null | wc -l)
echo "store holds $saved checkpoint(s)"
[ "$saved" -ge 1 ] || { echo "!! nothing was persisted"; rc=1; }
exit $rc
