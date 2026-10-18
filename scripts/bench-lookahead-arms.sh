#!/usr/bin/env bash
# bench-lookahead-arms.sh — does the router look-ahead pay for itself on *this*
# container?
#
# The knob's answer is a property of the storage, not of the engine: the
# speculative reads it issues are device reads queued on the same per-device ring
# as the reads the token waits for, so a box with fast NVMe and a box with three
# SATA SSDs can give opposite verdicts for identical code. The one time this was
# measured on four drives (one of them a 5.4 kRPM HDD) the answer was "on, +14 %".
# On three SSDs it is "off, +72 %". Hence a script rather than a number in a doc:
# run it on your box, get a verdict with its spread attached.
#
#   usage: scripts/bench-lookahead-arms.sh [out-dir] [reps]
#   default out-dir: bench-data/$(date +%F)-lookahead
#
# Arms (all correctness-neutral — the authoritative router still decides, so the
# token stream must be identical across arms; the script does not check that,
# because a bench cannot see the stream, and `router_lookahead_cannot_move_a_token`
# is the test that does):
#
#   on    COLI_ROUTER_LOOKAHEAD=1 COLI_ROUTER_LOOKAHEAD_N=<N>   the old default
#   off   COLI_ROUTER_LOOKAHEAD=0                                the new default
#   off+hints   off, plus COLI_PREFETCH_HINT_PATHS=<H>           page-cache fadvise
#                                                            instead of a device read
#
# Discipline, because this box's spread is large enough to invent verdicts:
#   - arms run strictly one at a time (two overlapping arms halve each other's
#     bandwidth and the gap stops being about the knob);
#   - the arm order ROTATES per rep, so drift in the box cannot favour one arm;
#   - the verdict is median-based and REFUSES to resolve a gap smaller than the
#     spread of the losing arm.
set -uo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 2

OUT=${1:-"bench-data/$(date +%F)-lookahead"}
REPS=${2:-4}
MODEL=${COLI_MODEL:-/srv/m-sdd/GLM-5.3-Flash-4ssd}
BIN=${BIN:-target-cuda/release/peregrine}
BATCH=${BATCH:-1}
STEPS=${COLI_BENCH_STEPS:-${STEPS:-6}}
WIDTH=${COLI_ROUTER_LOOKAHEAD_N:-64}
HINTS=${COLI_PREFETCH_HINT_PATHS:-64}

mkdir -p "$OUT"
[ -x "$BIN" ] || { echo "missing bench binary: $BIN" >&2; exit 2; }
[ -f "$MODEL/config.json" ] || { echo "missing container: $MODEL" >&2; exit 2; }

# The shipped profile, minus the two knobs under test. Everything else — the GPU
# tier, the fused reduce, the device scheduler, the io batch — is held constant so
# the diff is the knob.
base_env=(
    COLI_MODEL="$MODEL" COLI_STREAM=1
    COLI_GPU=1 COLI_GPU_INT4=1 COLI_GPU_F32_FRAC=0 COLI_GPU_PINNED=1 COLI_GPU_DEVICES=0
    COLI_GPU_EXPERT_HEADROOM_MB=1792 COLI_CUDA_GRAPH=1 COLI_CUDA_FUSED_REDUCE=1
    COLI_GPU_DENSE=1 COLI_GPU_DENSE_HEADROOM_MB=256 COLI_GPU_DENSE_MAX_BATCH=1
    COLI_MOE_ENGINE=concurrent COLI_IO_BATCH=8
    COLI_IO_DEVICE_SCHED=1 COLI_SSD_AWARE_SCHED=1 COLI_ECACHE_GB=6 COLI_DRAFT=0
    COLI_PREFETCH_STALE_DROP=1 COLI_FUSE_PREFILL=1 COLI_KV_BUDGET_MB=6000
    COLI_PAR_THREADS=12 MALLOC_ARENA_MAX=2
    COLI_BENCH_STEPS="$STEPS" COLI_UNION_STATS=1 COLI_RSS_GUARD_GB=20
)

arms=(on off off+hints)
declare -A arm_env=(
    [on]="COLI_ROUTER_LOOKAHEAD=1 COLI_ROUTER_LOOKAHEAD_N=$WIDTH"
    [off]="COLI_ROUTER_LOOKAHEAD=0"
    [off+hints]="COLI_ROUTER_LOOKAHEAD=0 COLI_PREFETCH_HINT_PATHS=$HINTS"
)

echo "== look-ahead arms =="
printf 'model=%s bin=%s batch=%s steps=%s reps=%s width=%s hints=%s\n' \
    "$MODEL" "$BIN" "$BATCH" "$STEPS" "$REPS" "$WIDTH" "$HINTS"
echo "out=$OUT"
echo

for ((rep = 1; rep <= REPS; rep++)); do
    # Rotate so the first arm of rep N is the last arm of rep N-1.
    for ((k = 0; k < ${#arms[@]}; k++)); do
        arm=${arms[$(( (rep - 1 + k) % ${#arms[@]} ))]}
        tag="$OUT/rep$(printf '%02d' "$rep")-$arm"
        mkdir -p "$tag"
        # shellcheck disable=SC2086 # arm_env is a deliberately word-split list
        timeout 1800 env "${base_env[@]}" ${arm_env[$arm]} \
            "$BIN" bench "$BATCH" >"$tag/bench.out" 2>"$tag/bench.err"
        # A CPU-only binary prints one advisory line and then runs to completion,
        # producing a plausible tok/s number measured on the wrong engine — which
        # is how a mistyped `--features cuda` becomes a benchmark nobody catches.
        # `COLI_GPU=1` is a request, not a guarantee, so the measurement checks.
        if [ "${REQUIRE_GPU:-1}" = 1 ] && ! grep -qE 'gpu=available' "$tag/bench.err"; then
            echo "rep$rep $arm: no CUDA backend in this run — refusing to record a number." >&2
            grep -oE 'gpu=.*' "$tag/bench.err" | head -1 >&2
            echo "  rebuild: cargo build --release --features cuda" >&2
            exit 1
        fi
        tok=$(awk '/^ +[0-9]+ +[0-9]+/{print $6; exit}' "$tag/bench.out")
        printf 'rep%02d %-10s %s tok/s\n' "$rep" "$arm" "${tok:-n/a}"
        # Serial arms are the whole point; a pause keeps the previous process's
        # teardown (and its page-cache writes) from landing inside the next one.
        sleep 2
    done
done

echo
echo "== verdict =="
declare -A vals
for arm in "${arms[@]}"; do
    vals[$arm]=$(
        for f in "$OUT"/rep*-"$arm"/bench.out; do
            awk '/^ +[0-9]+ +[0-9]+/{print $6; exit}' "$f" 2>/dev/null
        done | grep . | sort -n | tr '\n' ' '
    )
    med=$(
        for f in "$OUT"/rep*-"$arm"/bench.out; do
            awk '/^ +[0-9]+ +[0-9]+/{print $6; exit}' "$f" 2>/dev/null
        done | grep . | sort -n | awk '{a[NR]=$1} END{if(!NR)print "n/a"; else print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'
    )
    spread=$(
        for f in "$OUT"/rep*-"$arm"/bench.out; do
            awk '/^ +[0-9]+ +[0-9]+/{print $6; exit}' "$f" 2>/dev/null
        done | grep . | sort -n | awk 'NR==1{mn=$1} {mx=$1} END{if(!NR||mx==0)print "n/a"; else printf "%.1f%%",(mx-mn)/((mx+mn)/2)*100}'
    )
    printf '%-10s median=%-8s spread=%-6s runs=[%s]\n' "$arm" "$med" "$spread" "${vals[$arm]}"
done

on_med=$(for f in "$OUT"/rep*-on/bench.out; do awk '/^ +[0-9]+ +[0-9]+/{print $6; exit}' "$f"; done | sort -n | awk '{a[NR]=$1} END{print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}')
off_med=$(for f in "$OUT"/rep*-off/bench.out; do awk '/^ +[0-9]+ +[0-9]+/{print $6; exit}' "$f"; done | sort -n | awk '{a[NR]=$1} END{print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}')
on_spread=$(for f in "$OUT"/rep*-on/bench.out; do awk '/^ +[0-9]+ +[0-9]+/{print $6; exit}' "$f"; done | grep . | sort -n | awk 'NR==1{mn=$1}{mx=$1}END{printf "%.4f",(mx-mn)/((mx+mn)/2)}')
gap=$(awk -v a="$off_med" -v b="$on_med" 'BEGIN{if(b>0)printf "%.4f",(a-b)/b; else print "n/a"}')

on_n=$(for f in "$OUT"/rep*-on/bench.out; do awk '/^ +[0-9]+ +[0-9]+/{print $6; exit}' "$f"; done | grep -c .)
off_n=$(for f in "$OUT"/rep*-off/bench.out; do awk '/^ +[0-9]+ +[0-9]+/{print $6; exit}' "$f"; done | grep -c .)
if [ "$gap" = "n/a" ] || [ "$on_spread" = "n/a" ]; then
    echo "INCONCLUSIVE: not enough completed runs"
    exit 1
fi
# A spread computed from one run is 0 by definition, which would make any gap
# "larger than the spread". One rep is a smoke test, not a verdict.
if [ "$on_n" -lt 2 ] || [ "$off_n" -lt 2 ]; then
    echo "INCONCLUSIVE: $on_n on-run(s) and $off_n off-run(s); a verdict needs at least 2 of each (spread over one run is 0 by definition)"
    printf 'smoke result: on=%s off=%s — re-run with more reps before quoting\n' "$on_med" "$off_med"
    exit 1
fi
if awk -v g="$gap" -v s="$on_spread" 'BEGIN{exit !(g > s)}'; then
    echo "OFF WINS: median ${off_med} vs ${on_med} tok/s (+$(awk -v g="$gap" 'BEGIN{printf "%.0f", g*100}')%)," \
         "which is larger than the on-arm's ${on_spread} relative spread"
    exit 0
fi
echo "UNRESOLVED: gap ${gap} is inside the on-arm's ${on_spread} spread — more reps needed"
exit 1