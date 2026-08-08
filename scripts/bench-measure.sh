#!/usr/bin/env bash
# Instrumented isolated-batch measurement: capture every checkout-neutral
# signal a real optimization decision rests on, in one short run.
#
# What this measures, and why each one is on its own line:
#
# - `[predict-eval] recall=…` — the router look-ahead's recall against the next
#   layer's *actual* routed set, scored by `COLI_PREDICT_EVAL=1`. Recall is the
#   number that decides whether `COLI_ROUTER_LOOKAHEAD_BATCH` (the multi-row
#   look-ahead lift) pays in or out. A recall under ~50 % is the WASTE caution
#   and says keep it off; above ~70 % says leave it on by default.
#   **DECODE-ONLY: absent whenever BATCH > 1.** `model.rs:3517` gates the
#   scoreboard on `s_n == 1`, because a prefill chunk's "actual set" is a union
#   over positions and recall against it would not be the number any predictor
#   is aiming at. That gate cannot distinguish "prefill chunk of s_n positions"
#   from "batch of s_n sequences each decoding one token", where the actual set
#   *is* well defined per row — so it correctly suppresses the first and
#   incorrectly suppresses the second. Until that is split, this line and the
#   `[lookahead]` line below simply do not appear at B > 1.
# - `[union] selections=… distinct=… share=…` — the batch-union sharing factor
#   under speculation. With `--draft γ`, the union is over `(1+γ) × B` rows; the
#   share is what every distinct expert read serves. If the union grows with γ
#   fast enough to cancel the 1+γ amortization, speculation is a net negative
#   here. `COLI_UNION_STATS=1` emits this on the live engine.
# - `[gate] below_0.5%=…` — the per-position gate-mass distribution
#   (`COLI_GATE_STATS=1`). The share of routed experts below each threshold is
#   the share of the read budget that bought almost nothing — the size of the
#   `COLI_ROUTE_MIN_SHARE` lever. The repo's quality gate `prediction_flip_rate`
#   is what bounds a default value; this is what sizes it.
# - `[lookahead] issued=…` — speculative prefetch reads started by the
#   router look-ahead during the inter-layer boundary. Compare against
#   `[ecache] disk_reads=…` to read its hit rate.
#
# The number nobody has measured here is multi-row look-ahead recall on a B > 1
# batch. This script cannot get it — see the decode-only note above — and the
# 2026-08-07 pass is the evidence: its `lookahead_batch` arm came back with
# counters byte-identical to `baseline`.
#
# That null result had a second cause, now fixed. `COLI_ROUTER_LOOKAHEAD_BATCH`
# **defaults to ON** (`model.rs:836-840` reads `!matches!(v, Ok("0")|Ok("false"))`),
# so an arm that set it to 1 against a baseline that left it unset was comparing
# a configuration with itself. The comment that used to sit here said "the
# defaults align with the existing baseline (off)", which is where the wrong
# assumption was written down. `baseline` now disables it explicitly.
#
#   usage: scripts/bench-measure.sh <out-dir> [batch]
set -uo pipefail

MODEL=${COLI_MODEL:-/home/cortix/models/GLM-5.2-colibri-int4-with-int8-mtp}
BIN_CPU=${BIN_CPU:-target/release/peregrine}
BIN_GPU=${BIN_GPU:-target/cuda/release/peregrine}
BATCH=${BATCH:-16}
STEPS=${COLI_BENCH_STEPS:-2}
ECACHE=${COLI_ECACHE_GB:-2}
DRAFT=${COLI_DRAFT:-4}
MEMMAX=${MEMMAX:-24G}

OUT=${1:?usage: bench-measure.sh <out-dir> [batch]}
shift
if [ $# -ge 1 ]; then BATCH=$1; fi
mkdir -p "$OUT"

# Provenance, so a directory of .err files is still interpretable later. The
# 2026-08-07 pass shipped without this and its raw output cannot now be tied to
# a commit. The dirty-tree count is part of it on purpose: that pass ran against
# a tree with thousands of uncommitted lines, so a commit id alone would have
# described something that was never built.
{
    git log -1 --oneline 2>/dev/null || echo "(not a git checkout)"
    echo "uncommitted files: $(git status --porcelain 2>/dev/null | wc -l)"
    echo "built $(date -Is)"
    echo "batch=$BATCH steps=$STEPS ecache_gb=$ECACHE draft=$DRAFT memmax=$MEMMAX"
    echo "model=$MODEL"
} >"$OUT/BUILD.txt"

if [ "$BATCH" -gt 1 ]; then
    echo "note: BATCH=$BATCH > 1, so [predict-eval] and [lookahead] will be absent" \
         "(model.rs:3517 gates the scoreboard on s_n == 1)" | tee -a "$OUT/summary.txt"
fi

COMMON="COLI_MODEL=$MODEL MALLOC_ARENA_MAX=2 COLI_ECACHE_GB=$ECACHE \
COLI_BENCH_STEPS=$STEPS COLI_ROUTE_STATS_PERSIST=0 COLI_DEBUG=1"

# Instrumentation that's on for every arm here. All bit-identical (advisory stats
# only) — the only change that touches model output in this whole script is
# `COLI_DRAFT`, and that is bit-identical under greedy decode too.
INSTRUMENT="COLI_PREDICT_EVAL=1 COLI_PREDICT_EVAL_N=8 COLI_UNION_STATS=1 \
COLI_GATE_STATS=1 COLI_PERF_COUNTERS=1"

for variant in baseline spec lookahead_batch; do
    case "$variant" in
    baseline) vars="COLI_ROUTER_LOOKAHEAD_BATCH=0" ;;
    spec) vars="COLI_ROUTER_LOOKAHEAD_BATCH=0 COLI_DRAFT=$DRAFT" ;;
    lookahead_batch) vars="COLI_ROUTER_LOOKAHEAD_BATCH=1" ;;
    esac

    tag="$variant.$BATCH"
    cap=(systemd-run --user --scope -p "MemoryMax=$MEMMAX" -p MemorySwapMax=0 --quiet)
    extra_args=""
    [ "$variant" = spec ] && extra_args="--draft $DRAFT"

    printf '%s\n' $COMMON $INSTRUMENT $vars >"$OUT/$tag.env"

    echo "== $tag: $BIN_CPU batch=$BATCH steps=$STEPS extra=[$extra_args]" | tee -a "$OUT/summary.txt"
    # shellcheck disable=SC2046
    env $COMMON $INSTRUMENT $vars "${cap[@]}" \
        scripts/runstat.py "$OUT/$tag.stat" "$BIN_CPU" bench $BATCH $extra_args \
        >"$OUT/$tag.out" 2>"$OUT/$tag.err"
    rc=$?
    echo "   rc=$rc" | tee -a "$OUT/summary.txt"
    tee -a "$OUT/summary.txt" <"$OUT/$tag.out"
    grep -E "predict-eval|union|gate|lookahead|ecache|prefetch" "$OUT/$tag.err" \
        | sed 's/^/   /' | head -60 | tee -a "$OUT/summary.txt"
    grep -E "wall_s|peak_rss_gb|major_faults" "$OUT/$tag.stat" 2>/dev/null \
        | sed 's/^/   /' | tee -a "$OUT/summary.txt"
    [ $rc -ne 0 ] && tail -20 "$OUT/$tag.err"
    echo | tee -a "$OUT/summary.txt"
done

# Not "$OUT/$variant.$BATCH.err": `$variant` outlives the loop, so that named
# only the last arm however many ran.
echo "raw instrument output in $OUT/ (one .err per arm, provenance in BUILD.txt)"
