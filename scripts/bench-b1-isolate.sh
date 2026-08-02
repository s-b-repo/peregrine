#!/usr/bin/env bash
# B=1 knob isolation with repeats.
#
# The 3-arm sweep showed single-sequence decode *regressing* under the improved
# knob bundle (0.045 -> 0.028 tok/s) while batched decode gained, and the GPU
# arm — the same bundle plus COLI_GPU=1 — beating both. Disk-read counts were
# identical across arms, so the cause is I/O-path scheduling, not cache hits.
# This script re-measures B=1 with repeats to separate real effects from
# run-to-run noise on a contended box.
#
#   usage: scripts/bench-b1-isolate.sh <out-dir> [repeats]
set -uo pipefail

MODEL=${COLI_MODEL:-/home/cortix/models/GLM-5.2-colibri-int4-with-int8-mtp}
BIN_CPU=${BIN_CPU:-target/release/peregrine}
BIN_GPU=${BIN_GPU:-target/cuda/release/peregrine}
STEPS=${COLI_BENCH_STEPS:-3}
ECACHE=${COLI_ECACHE_GB:-2}
MEMMAX=${MEMMAX:-24G}
BATCH=${BATCH:-1}
VARIANTS=${VARIANTS:-"baseline direct_only bundle_nodirect improved gpu"}

OUT=${1:?usage: bench-b1-isolate.sh <out-dir> [repeats]}
REPEATS=${2:-2}
mkdir -p "$OUT"

COMMON="COLI_MODEL=$MODEL MALLOC_ARENA_MAX=2 COLI_ECACHE_GB=$ECACHE \
COLI_BENCH_STEPS=$STEPS COLI_ROUTE_STATS_PERSIST=0 COLI_DEBUG=1"

# The improved bundle minus whichever knob a variant is isolating.
# COLI_REGBUF dropped: read by no code (see docs/configuration.md).
BUNDLE="COLI_IO_TUNE=1 COLI_LANE_BALANCE=1 COLI_SHAPE_SPECIALIZE=1 \
COLI_HYPER_SCHED=1 COLI_PREFETCH_TUNE=1 COLI_ENTROPY_ADAPT=1 COLI_REPLICATE_K=8"

variant_env() {
    case "$1" in
    baseline) ;;                                   # plain defaults
    direct_only) echo "COLI_DIRECT=1" ;;           # baseline + O_DIRECT alone
    bundle_nodirect) echo "$BUNDLE" ;;             # improved minus O_DIRECT
    improved) echo "COLI_DIRECT=1 $BUNDLE" ;;      # full bundle (as swept)
    gpu) echo "COLI_DIRECT=1 $BUNDLE COLI_GPU=1" ;;
    *) echo "unknown variant: $1" >&2; return 1 ;;
    esac
}

echo "variant,rep,seconds,agg_tok_s,peak_rss_gb" >"$OUT/results.csv"
for rep in $(seq 1 "$REPEATS"); do
    for v in $VARIANTS; do
        bin=$BIN_CPU
        [ "$v" = gpu ] && bin=$BIN_GPU
        tag="$v.rep$rep"
        vars=$(variant_env "$v") || exit 1

        # shellcheck disable=SC2046,SC2086  # deliberate splitting into env args
        env $COMMON $vars \
            systemd-run --user --scope -p "MemoryMax=$MEMMAX" -p MemorySwapMax=0 --quiet \
            scripts/runstat.py "$OUT/$tag.stat" "$bin" bench "$BATCH" \
            >"$OUT/$tag.out" 2>"$OUT/$tag.err"

        secs=$(awk -v b="$BATCH" '$1==b{print $4}' "$OUT/$tag.out")
        agg=$(awk -v b="$BATCH" '$1==b{print $5}' "$OUT/$tag.out")
        rss=$(awk '$1=="peak_rss_gb"{print $2}' "$OUT/$tag.stat")
        echo "$v,$rep,${secs:-NA},${agg:-NA},${rss:-NA}" | tee -a "$OUT/results.csv"
    done
done

echo "--- medians ---"
awk -F, 'NR>1 && $4!="NA"{a[$1]=a[$1]" "$4} END{for(v in a) print v": "a[v]}' "$OUT/results.csv"
