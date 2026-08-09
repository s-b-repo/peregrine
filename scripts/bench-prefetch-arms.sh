#!/usr/bin/env bash
# Isolate what causes the warm cache's ~0.4% hit rate, one arm per hypothesis.
#
# Each arm is one `peregrine-serve` process, one identical request, SIGINT, and
# the shutdown `[ecache]`/`[prefetch]` counters. Arms differ only in environment,
# and every knob here is correctness-neutral, so the emitted token stream must be
# identical across arms — the script checks that rather than assuming it.
#
# Why the metrics survive this box's constraints: `hit_rate` and `disk_reads`
# count misses in *peregrine's own* warm cache, not OS page-cache state. Unlike
# wall clock they are not distorted by the fact that there is no passwordless
# sudo here to drop the page cache between arms. Wall time is recorded but is the
# weakest column; read the counters.
#
#   usage: scripts/bench-prefetch-arms.sh <out-dir> [arm ...]
#   arms:  default protect_off prefetch_off bigcache
set -uo pipefail

MODEL=${COLI_MODEL:-/home/cortix/models/GLM-5.2-colibri-int4-with-int8-mtp}
BIN=${BIN_SERVE:-target/release/peregrine-serve}
PORT=${PORT:-8137}
MAX_TOKENS=${MAX_TOKENS:-2}
ECACHE=${COLI_ECACHE_GB:-4}
BIGCACHE=${BIGCACHE_GB:-14}
PROMPT=${PROMPT:-"Explain how a mixture-of-experts layer routes a token."}

OUT=${1:?usage: bench-prefetch-arms.sh <out-dir> [arm ...]}
shift
ARMS=${*:-"default protect_off prefetch_off"}
mkdir -p "$OUT"

[ -x "$BIN" ] || { echo "no server binary at $BIN (cargo build --release --bins)" >&2; exit 1; }

# Per-arm environment. `default` is deliberately empty: the shipped defaults are
# the configuration whose 0.4% hit rate is under investigation.
arm_env() {
    case "$1" in
    default) ;;
    # Candidate 1: the predictor's set is given cache priority >= 1 while a
    # demand-loaded slab that was routed but not predicted stays at 0, so a
    # never-used speculative slab outranks it in the victim order. Same reads
    # issued as `default` — only the eviction order changes.
    protect_off) echo "COLI_PREFETCH_PROTECT=0" ;;
    # Candidate 2: speculation competes for a device the demand path is starved
    # on. Removes essentially all speculative traffic. Both emitters must be
    # named: the router look-ahead is built independently of the history
    # predictor, so warm-paths alone still leaves it issuing per layer.
    prefetch_off)
        echo "COLI_PREFETCH_WARM_PATHS=0"
        echo "COLI_PREFETCH_HINT_PATHS=0"
        echo "COLI_ROUTER_LOOKAHEAD=0"
        ;;
    # Candidate 3: one token's routed set is ~11.3 GB (600 experts x 18.9 MB),
    # so a 4 GB cache cannot hold it and cross-token reuse is structurally
    # impossible. This crosses that threshold.
    bigcache) echo "COLI_ECACHE_GB=$BIGCACHE" ;;
    # The combination, and the one that separates the two. A cache above one
    # token's 11.3 GB working set should permit cross-token reuse — but only if
    # the demand set is what occupies it. Speculation adds ~7 GB per token on top
    # of the 11.3, so at 12.9 GB the two together still overflow and evict each
    # other. This arm gives the demand set the whole budget.
    bigcache_noprefetch)
        echo "COLI_ECACHE_GB=$BIGCACHE"
        echo "COLI_PREFETCH_WARM_PATHS=0"
        echo "COLI_PREFETCH_HINT_PATHS=0"
        echo "COLI_ROUTER_LOOKAHEAD=0"
        ;;
    *) echo "unknown arm: $1" >&2; return 1 ;;
    esac
}

run_arm() {
    # Declared separately: under `set -u`, a single `local a=$1 b=$a` is not a
    # reliable way to reference an earlier name in the same statement.
    local arm=$1
    local log="$OUT/$arm.log"
    local body="$OUT/$arm.json"
    local env_file="$OUT/$arm.env"
    echo "-- arm $arm" >&2
    { echo "COLI_MODEL=$MODEL"
      echo "COLI_ECACHE_GB=$ECACHE"
      echo "COLI_MEMO_ENTRIES=0"        # a memo hit would decode nothing at all
      echo "COLI_ROUTE_STATS_PERSIST=0" # else each arm warms the next via route_stats.json
      echo "COLI_DEBUG=1"
      echo "MALLOC_ARENA_MAX=2"
      arm_env "$arm" || return 1; } > "$env_file"

    env $(tr '\n' ' ' < "$env_file") "$BIN" --model "$MODEL" --port "$PORT" >"$log" 2>&1 &
    local pid=$!
    local waited=0
    until curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
        kill -0 "$pid" 2>/dev/null || { echo "  !! exited during load; see $log" >&2; tail -5 "$log" >&2; return 1; }
        sleep 2; waited=$((waited + 2))
        [ "$waited" -ge 900 ] && { echo "  !! not healthy in 900s" >&2; kill -INT "$pid"; return 1; }
    done

    local t0=$SECONDS
    curl -s -m 3600 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"glm-5.2\",\"messages\":[{\"role\":\"user\",\"content\":$(printf '%s' "$PROMPT" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')}],\"max_tokens\":$MAX_TOKENS,\"temperature\":0.0}" \
        >"$body"
    local elapsed=$((SECONDS - t0))

    # SIGINT, not SIGKILL: graceful shutdown is what prints the counters.
    kill -INT "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    { echo "arm=$arm load_s=$waited decode_s=$elapsed"
      grep -E '^\[(ecache|prefetch|prefix-cache)\]' "$log"; } > "$OUT/$arm.counters.txt"
    cat "$OUT/$arm.counters.txt"
    echo
}

for a in $ARMS; do run_arm "$a" || echo "  arm failed: $a" >&2; done

echo "== token-stream identity check (arms are correctness-neutral by construction)"
python3 - "$OUT" $ARMS <<'PY'
import json, os, sys
out, arms = sys.argv[1], sys.argv[2:]
seen = {}
for a in arms:
    p = os.path.join(out, f"{a}.json")
    try:
        with open(p) as fh:
            d = json.load(fh)
        seen[a] = d.get("choices", [{}])[0].get("message", {}).get("content")
    except (OSError, json.JSONDecodeError, IndexError, KeyError):
        seen[a] = None
vals = [v for v in seen.values() if v is not None]
if len(vals) < 2:
    print("  too few completed arms to compare")
elif all(v == vals[0] for v in vals):
    print(f"  OK — all {len(vals)} arms emitted an identical completion")
else:
    print("  MISMATCH — a correctness-neutral knob changed the output:")
    for a, v in seen.items():
        print(f"    {a}: {str(v)[:80]!r}")
PY
