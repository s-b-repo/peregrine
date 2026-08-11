#!/usr/bin/env bash
# A/B environment profiles over the serving path at a fixed batch depth.
#
# bench-serve-batch.sh sweeps B under one environment; this holds B and sweeps
# environments, for lever A/Bs (M1 of the 2-tok/s campaign). Each arm is a
# shell fragment of `export COLI_...` lines; the arm's basename is its label.
# Arms run in ROTATING order across repeats, medians reported, [ecache]/[lane]
# counters archived — same discipline and the same server lifecycle as
# bench-serve-batch.sh (systemd scope, SIGINT shutdown, no route_stats rewrite).
#
#   usage: scripts/bench-serve-envarms.sh <out-dir> <B> <arm.env> [arm.env ...]
#   env:   MODEL BIN_SERVE PORT MAX_TOKENS REPEATS BOOT_TIMEOUT MEMMAX
set -uo pipefail

MODEL=${COLI_MODEL:-/home/cortix/models/GLM-5.2-colibri-int4-with-int8-mtp}
BIN=${BIN_SERVE:-target/release/peregrine-serve}
PORT=${PORT:-8131}
MAX_TOKENS=${MAX_TOKENS:-32}
REPEATS=${REPEATS:-3}
BOOT_TIMEOUT=${BOOT_TIMEOUT:-1800}
MEMMAX=${MEMMAX:-34G}
# A non-streaming request spans prefill + all decode tokens; unfused deep
# batches measured >60 min end-to-end, so the guillotine must sit well past it.
CLIENT_TIMEOUT=${CLIENT_TIMEOUT:-10800}

OUT=${1:?usage: bench-serve-envarms.sh <out-dir> <B> <arm.env> [arm.env ...]}
B=${2:?need a batch depth}
shift 2
ARMS=("$@")
[ ${#ARMS[@]} -ge 2 ] || { echo "need at least 2 arm files to A/B" >&2; exit 1; }
for a in "${ARMS[@]}"; do [ -f "$a" ] || { echo "no arm file $a" >&2; exit 1; }; done

mkdir -p "$OUT"
[ -x "$BIN" ] || { echo "no server binary at $BIN" >&2; exit 1; }

{
  echo "== env-arm sweep: B=$B max_tokens=$MAX_TOKENS repeats=$REPEATS memmax=$MEMMAX"
  echo "== arms: ${ARMS[*]}"
  echo "== model=$MODEL"
  echo "== git: $(git -C "$(dirname "$0")/.." rev-parse --short HEAD 2>/dev/null || echo '?')"
} | tee "$OUT/README.txt"
for a in "${ARMS[@]}"; do cp "$a" "$OUT/arm-$(basename "$a")"; done

run_arm() { # $1=arm-file $2=rep
    local armfile=$1 rep=$2
    local name tag log
    name=$(basename "$armfile" .env)
    tag="${name}-rep${rep}"
    log="$OUT/$tag.server.log"
    # Uniform cold start per arm, if the narrow sudoers rule is installed
    # (2026-08-09). Logged either way: an arm's warmth is provenance.
    if sudo -n /usr/local/bin/drop-caches 2>/dev/null; then
        echo "  page cache dropped" >&2
    else
        echo "  page cache NOT dropped (no rule); rotating order is the defense" >&2
    fi
    (
        set -a
        # shellcheck disable=SC1090
        . "$armfile"
        set +a
        COLI_MODEL="$MODEL" COLI_ROUTE_STATS_PERSIST=0 COLI_DEBUG=1 MALLOC_ARENA_MAX=2 \
        exec systemd-run --user --scope -q -p MemoryMax="$MEMMAX" -p MemorySwapMax=0 \
            "$BIN" --model "$MODEL" --port "$PORT" --max-batch "$B"
    ) >"$log" 2>&1 &
    local pid=$!
    local waited=0
    until curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "  !! server exited during load; tail of $log:" >&2
            tail -5 "$log" >&2
            return 1
        fi
        sleep 2
        waited=$((waited + 2))
        [ "$waited" -ge "$BOOT_TIMEOUT" ] && { echo "  !! not healthy in ${BOOT_TIMEOUT}s" >&2; kill -INT "$pid"; return 1; }
    done
    echo "  loaded in ~${waited}s" >&2
    python3 scripts/bench-serve-lanes.py \
        --url "http://127.0.0.1:$PORT/v1/chat/completions" \
        --concurrency "$B" --max-tokens "$MAX_TOKENS" --timeout "$CLIENT_TIMEOUT" \
        --distinct-prompts --label "$tag" >"$OUT/$tag.json"
    local rc=$?
    kill -INT "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    grep -E '^\[(ecache|prefetch|predict-eval|ram|lane|workingset|expertmap)\]' "$log" >"$OUT/$tag.counters.txt" 2>/dev/null
    return $rc
}

for rep in $(seq 1 "$REPEATS"); do
    n=${#ARMS[@]}
    for i in $(seq 0 $((n - 1))); do
        arm=${ARMS[$(((i + rep - 1) % n))]}
        echo "-- rep $rep arm $(basename "$arm")"
        run_arm "$arm" "$rep" || echo "  arm failed ($(basename "$arm") rep=$rep)" >&2
    done
done

echo
echo "== medians (aggregate tok/s at B=$B)"
python3 - "$OUT" "${ARMS[@]}" <<'PY'
import glob, json, os, statistics, sys
out, arms = sys.argv[1], sys.argv[2:]
names = [os.path.basename(a).removesuffix(".env") for a in arms]
w = max(len(n) for n in names) + 2
print(f"{'arm':>{w}} {'runs':>5} {'median tok/s':>13} {'all':>8}")
for n in names:
    vals = []
    for f in sorted(glob.glob(os.path.join(out, f"{n}-rep*.json"))):
        try:
            with open(f) as fh:
                d = json.load(fh)
            if d.get("streams_ok"):
                vals.append(d["tokens_per_s"])
        except (OSError, json.JSONDecodeError, KeyError):
            pass
    if vals:
        print(f"{n:>{w}} {len(vals):>5} {statistics.median(vals):>13.3f}   {[round(v, 2) for v in vals]}")
    else:
        print(f"{n:>{w}} {0:>5} {'—':>13}")
print("\nDifferences inside the +-3% long-run band are not results. Check the")
print(".counters.txt reads: tok/s bought by proportionally more disk_reads is")
print("not a lever, it is a cache trade.")
PY
