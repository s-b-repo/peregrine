#!/usr/bin/env bash
# Aggregate decode throughput vs serve batch depth (MAX_BATCH = client concurrency).
#
# This is the goal metric of the 2-tok/s-aggregate campaign: total decoded
# tokens per second across B concurrent streams, measured end-to-end through
# peregrine-serve. bench-serve-lanes.sh sweeps a prefetch knob at fixed B; this
# sweeps B itself, because batch-union sharing (measured 4.98x at B=16) is the
# denominator that makes aggregate tok/s scale where single-stream cannot.
#
# Methodology (same box constraints as bench-serve-lanes.sh):
#   * No passwordless sudo -> the page cache cannot be dropped between arms.
#     Arms run in ROTATING order across repeats; report MEDIANS.
#   * A background VM swings ~50 s runs by +-45%, 200 s+ runs by +-3%.
#     Every arm here is 200 s+ by construction at current speeds.
#   * The server runs inside a systemd scope with MemoryMax: an OOM must be
#     contained, not left to the kernel to resolve against the VM.
#   * Servers get SIGINT, never SIGKILL: the [ecache]/[lane]/[ram] shutdown
#     counters this harness archives are printed on graceful shutdown, and
#     route_stats.json is rewritten at Drop (persistence is disabled here so
#     the checkpoint's copy stays pristine for the reshard tooling).
#   * REFS=1 first boots a quiet server and records temperature-0 transcripts
#     of 3 fixed prompts via peregrine-gen. These are the bit-identity
#     references every later lever must byte-match.
#
#   usage: scripts/bench-serve-batch.sh <out-dir> [B ...]
#   env:   MODEL BIN_SERVE BIN_GEN PORT MAX_TOKENS REPEATS BOOT_TIMEOUT
#          MEMMAX REFS REF_TOKENS
#   Arms inherit the caller's COLI_* environment; the swept variable is B.
set -uo pipefail

MODEL=${COLI_MODEL:-/home/cortix/models/GLM-5.2-colibri-int4-with-int8-mtp}
BIN=${BIN_SERVE:-target/release/peregrine-serve}
GEN=${BIN_GEN:-target/release/peregrine-gen}
PORT=${PORT:-8131}
MAX_TOKENS=${MAX_TOKENS:-32}
REPEATS=${REPEATS:-3}
BOOT_TIMEOUT=${BOOT_TIMEOUT:-1800}
MEMMAX=${MEMMAX:-34G}
# A non-streaming request spans prefill + all decode tokens; unfused deep
# batches measured >60 min end-to-end, so the guillotine must sit well past it.
CLIENT_TIMEOUT=${CLIENT_TIMEOUT:-10800}
REFS=${REFS:-0}
REF_TOKENS=${REF_TOKENS:-16}

OUT=${1:?usage: bench-serve-batch.sh <out-dir> [B ...]}
shift
BATCHES=${*:-"16 32"}

REF_PROMPTS=(
  "Explain how a mixture-of-experts layer routes a token."
  "Write a short Rust function that reverses a linked list."
  "Summarise the causes of the 1929 financial crash."
)

mkdir -p "$OUT"
[ -x "$BIN" ] || { echo "no server binary at $BIN (cargo build --release --bins)" >&2; exit 1; }
[ -d "$MODEL" ] || { echo "no model dir at $MODEL" >&2; exit 1; }

{
  echo "== batch sweep: B=[$BATCHES] max_tokens=$MAX_TOKENS repeats=$REPEATS memmax=$MEMMAX"
  echo "== model=$MODEL"
  echo "== host: $(uname -sr) | $(nproc) cpus | $(free -g | awk '/^Mem:/{print $2"G ram, "$7"G avail"}')"
  echo "== git: $(git -C "$(dirname "$0")/.." rev-parse --short HEAD 2>/dev/null || echo '?')"
  echo "== env: $(env | grep '^COLI_' | sort | tr '\n' ' ')"
} | tee "$OUT/README.txt"

# Sets the global `pid`. Not a command substitution: the server must stay a
# child of this shell so stop_server's `wait` really blocks until the shutdown
# counters have been flushed to the log.
start_server() { # $1=max_batch $2=logfile
    COLI_MODEL="$MODEL" \
    COLI_ROUTE_STATS_PERSIST=0 \
    COLI_DEBUG=1 \
    MALLOC_ARENA_MAX=2 \
    systemd-run --user --scope -q -p MemoryMax="$MEMMAX" -p MemorySwapMax=0 \
        "$BIN" --model "$MODEL" --port "$PORT" --max-batch "$1" >"$2" 2>&1 &
    pid=$!
}

wait_healthy() { # $1=pid $2=logfile
    local waited=0
    until curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
        if ! kill -0 "$1" 2>/dev/null; then
            echo "  !! server exited during load; tail of $2:" >&2
            tail -5 "$2" >&2
            return 1
        fi
        sleep 2
        waited=$((waited + 2))
        if [ "$waited" -ge "$BOOT_TIMEOUT" ]; then
            echo "  !! server not healthy in ${BOOT_TIMEOUT}s" >&2
            kill -INT "$1" 2>/dev/null
            return 1
        fi
    done
    echo "  loaded in ~${waited}s" >&2
}

stop_server() { # $1=pid $2=logfile $3=counters-out
    kill -INT "$1" 2>/dev/null
    wait "$1" 2>/dev/null
    grep -E '^\[(ecache|prefetch|predict-eval|ram|lane|workingset|expertmap)\]' "$2" >"$3" 2>/dev/null
}

if [ "$REFS" = "1" ]; then
    mkdir -p "$OUT/reference-completions"
    log="$OUT/refs.server.log"
    echo "-- reference transcripts (temperature 0, $REF_TOKENS tokens, quiet server)" >&2
    start_server 4 "$log"
    if wait_healthy "$pid" "$log"; then
        i=0
        for p in "${REF_PROMPTS[@]}"; do
            i=$((i + 1))
            "$GEN" --port "$PORT" --temperature 0 --max-tokens "$REF_TOKENS" \
                   --json "$OUT/reference-completions/ref$i.json" --quiet "$p" \
                   2>"$OUT/reference-completions/ref$i.stats.txt" \
                   >"$OUT/reference-completions/ref$i.txt" \
                || echo "  !! ref $i failed" >&2
            echo "  ref $i done" >&2
        done
        stop_server "$pid" "$log" "$OUT/refs.counters.txt"
    else
        echo "  !! reference pass failed to boot" >&2
    fi
fi

run_arm() { # $1=B $2=rep
    local b=$1 rep=$2
    local tag="b${b}-rep${rep}"
    local log="$OUT/$tag.server.log"
    # Uniform cold start per arm, if the narrow sudoers rule is installed
    # (2026-08-09). Logged either way: an arm's warmth is provenance.
    if sudo -n /usr/local/bin/drop-caches 2>/dev/null; then
        echo "  page cache dropped" >&2
    else
        echo "  page cache NOT dropped (no rule); rotating order is the defense" >&2
    fi
    start_server "$b" "$log"
    wait_healthy "$pid" "$log" || { kill -INT "$pid" 2>/dev/null; return 1; }
    python3 scripts/bench-serve-lanes.py \
        --url "http://127.0.0.1:$PORT/v1/chat/completions" \
        --concurrency "$b" --max-tokens "$MAX_TOKENS" --timeout "$CLIENT_TIMEOUT" \
        --distinct-prompts --label "$tag" >"$OUT/$tag.json"
    local rc=$?
    stop_server "$pid" "$log" "$OUT/$tag.counters.txt"
    return $rc
}

for rep in $(seq 1 "$REPEATS"); do
    order=$(echo $BATCHES | tr ' ' '\n' | awk -v r="$rep" '{a[NR]=$0} END{for(i=0;i<NR;i++) print a[((i+r-1)%NR)+1]}')
    for b in $order; do
        echo "-- rep $rep B=$b"
        run_arm "$b" "$rep" || echo "  arm failed (B=$b rep=$rep)" >&2
    done
done

echo
echo "== medians (aggregate tok/s across B streams)"
python3 - "$OUT" $BATCHES <<'PY'
import glob, json, os, statistics, sys
out, batches = sys.argv[1], sys.argv[2:]
print(f"{'B':>4} {'runs':>5} {'median tok/s':>13} {'all':>8}")
for b in batches:
    vals = []
    for f in sorted(glob.glob(os.path.join(out, f"b{b}-rep*.json"))):
        try:
            with open(f) as fh:
                d = json.load(fh)
            if d.get("streams_ok"):
                vals.append(d["tokens_per_s"])
        except (OSError, json.JSONDecodeError, KeyError):
            pass
    if vals:
        print(f"{b:>4} {len(vals):>5} {statistics.median(vals):>13.3f}   {[round(v, 2) for v in vals]}")
    else:
        print(f"{b:>4} {0:>5} {'—':>13}")
print("\nA difference inside +-3% (long runs) is NOT a result. Check .counters.txt:")
print("more tok/s bought by proportionally more disk_reads means the union is")
print("still growing; flat tok/s with flat reads is the knee.")
PY
