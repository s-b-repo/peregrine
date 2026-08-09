#!/usr/bin/env bash
# Sweep COLI_PREFETCH_LANES over the *serving* path and report medians.
#
# Why this exists rather than another arm in bench-arms.sh: that harness drives
# `peregrine bench` → `Model::forward_step_batched`, and the only caller that
# ever picks a prefetch lane other than 0 is `Model::enqueue_seq_prefetch`, which
# lives in peregrine-serve's batch loop. A lane sweep run through `bench` would
# measure nothing and report a clean null result, which is worse than no number.
#
# Methodology notes, because this box constrains the design:
#   * There is no passwordless sudo here, so the page cache CANNOT be dropped
#     between arms and an arm run after another starts warmer. Arms are therefore
#     run in ROTATING order across repeats and reported as MEDIANS, so cache
#     warmth is spread across arms instead of accumulating on the last one.
#   * A background VM on this box has swung identical configs by ±45% on ~50s
#     runs and ±3% on 200s+ runs. Prefer few long arms to many short ones:
#     raise MAX_TOKENS rather than REPEATS if you have the time budget.
#   * Reads are what a prefetch knob actually changes, so the per-arm [ecache]
#     and [prefetch] shutdown lines are captured too. A tok/s win bought by
#     reading more bytes is not a win.
#
#   usage: scripts/bench-serve-lanes.sh <out-dir> [lanes ...]
set -uo pipefail

MODEL=${COLI_MODEL:-/home/cortix/models/GLM-5.2-colibri-int4-with-int8-mtp}
BIN=${BIN_SERVE:-target/release/peregrine-serve}
PORT=${PORT:-8127}
CONCURRENCY=${CONCURRENCY:-8}
MAX_TOKENS=${MAX_TOKENS:-32}
REPEATS=${REPEATS:-3}
ECACHE=${COLI_ECACHE_GB:-4}
MAX_BATCH=${MAX_BATCH:-8}
BOOT_TIMEOUT=${BOOT_TIMEOUT:-1800}

OUT=${1:?usage: bench-serve-lanes.sh <out-dir> [lanes ...]}
shift
LANES=${*:-"1 2 4 8"}
mkdir -p "$OUT"

[ -x "$BIN" ] || { echo "no server binary at $BIN (cargo build --release --bins)" >&2; exit 1; }
[ -d "$MODEL" ] || { echo "no model dir at $MODEL" >&2; exit 1; }

echo "== lane sweep: lanes=[$LANES] concurrency=$CONCURRENCY max_tokens=$MAX_TOKENS repeats=$REPEATS" | tee "$OUT/README.txt"
echo "== model=$MODEL ecache=${ECACHE}G max_batch=$MAX_BATCH" | tee -a "$OUT/README.txt"
echo "== host: $(uname -sr) | $(nproc) cpus | $(free -g | awk '/^Mem:/{print $2"G ram, "$7"G avail"}')" | tee -a "$OUT/README.txt"

run_arm() {
    local lanes=$1 rep=$2
    local tag="lanes${lanes}-rep${rep}"
    local log="$OUT/$tag.server.log"
    local res="$OUT/$tag.json"

    COLI_MODEL="$MODEL" \
    COLI_PREFETCH_LANES="$lanes" \
    COLI_ECACHE_GB="$ECACHE" \
    COLI_ROUTE_STATS_PERSIST=0 \
    COLI_DEBUG=1 \
    MALLOC_ARENA_MAX=2 \
        "$BIN" --model "$MODEL" --port "$PORT" --max-batch "$MAX_BATCH" >"$log" 2>&1 &
    local pid=$!

    # Wait for the model to load. A 358 GB checkpoint takes minutes, and a dead
    # server must not be waited on for the full timeout.
    local waited=0
    until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "  !! server exited during load; see $log" >&2
            tail -5 "$log" >&2
            return 1
        fi
        sleep 2
        waited=$((waited + 2))
        if [ "$waited" -ge "$BOOT_TIMEOUT" ]; then
            echo "  !! server did not become healthy in ${BOOT_TIMEOUT}s" >&2
            kill -INT "$pid" 2>/dev/null
            return 1
        fi
    done
    echo "  loaded in ~${waited}s" >&2

    python3 scripts/bench-serve-lanes.py \
        --url "http://127.0.0.1:$PORT/v1/chat/completions" \
        --concurrency "$CONCURRENCY" --max-tokens "$MAX_TOKENS" \
        --distinct-prompts --label "$tag" >"$res"
    local rc=$?

    # SIGINT, not SIGKILL: graceful shutdown is what prints the [ecache] and
    # [prefetch] counters this sweep is half about.
    kill -INT "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    grep -E '^\[(ecache|prefetch|predict-eval|ram)\]' "$log" >"$OUT/$tag.counters.txt" 2>/dev/null
    return $rc
}

# Rotating order per repeat, so no arm is systematically the one that runs on the
# warmest page cache.
for rep in $(seq 1 "$REPEATS"); do
    order=$(echo $LANES | tr ' ' '\n' | awk -v r="$rep" '{a[NR]=$0} END{for(i=0;i<NR;i++) print a[((i+r-1)%NR)+1]}')
    for l in $order; do
        echo "-- rep $rep lanes $l" >&2
        run_arm "$l" "$rep" || echo "  arm failed (lanes=$l rep=$rep)" >&2
    done
done

echo
echo "== medians (tokens/s aggregate across $CONCURRENCY streams)"
python3 - "$OUT" $LANES <<'PY'
import glob, json, os, statistics, sys
out, lanes = sys.argv[1], sys.argv[2:]
print(f"{'lanes':>6} {'runs':>5} {'median tok/s':>13} {'all':>8}")
for l in lanes:
    vals = []
    for f in sorted(glob.glob(os.path.join(out, f"lanes{l}-rep*.json"))):
        try:
            with open(f) as fh:
                d = json.load(fh)
            if d.get("streams_ok"):
                vals.append(d["tokens_per_s"])
        except (OSError, json.JSONDecodeError, KeyError):
            pass
    if vals:
        print(f"{l:>6} {len(vals):>5} {statistics.median(vals):>13.3f}   {[round(v,2) for v in vals]}")
    else:
        print(f"{l:>6} {0:>5} {'—':>13}")
print("\nA difference inside the box's ±3% (long runs) / ±45% (short runs) noise")
print("band is NOT a result. Check the .counters.txt files: more tok/s bought by")
print("more disk reads is not a win.")
PY
