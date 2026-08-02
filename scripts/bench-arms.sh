#!/usr/bin/env bash
# Multi-arm decode-throughput benchmark for peregrine.
#
# Each arm is one full `peregrine bench` sweep over the batch sizes in $BATCHES,
# run in a fresh process so no adaptive state leaks between arms. Per arm we
# capture the stdout table, the stderr [ecache] counters, peak RSS and wall
# time. Arms differ only in environment: every knob here is correctness-neutral,
# so all arms must produce the same token stream.
#
#   usage: scripts/bench-arms.sh <out-dir> [arm ...]
set -uo pipefail

MODEL=${COLI_MODEL:-/home/cortix/models/GLM-5.2-colibri-int4-with-int8-mtp}
BIN_CPU=${BIN_CPU:-target/release/peregrine}
BIN_GPU=${BIN_GPU:-target/cuda/release/peregrine}
BATCHES=${BATCHES:-"1 4 16"}
STEPS=${COLI_BENCH_STEPS:-3}
ECACHE=${COLI_ECACHE_GB:-4}

OUT=${1:?usage: bench-arms.sh <out-dir> [arm ...]}
shift
ARMS=${*:-"baseline improved gpu"}
mkdir -p "$OUT"

# Shared by every arm: the study's OOM guard, a cache sized for this box, and
# no cross-session persistence — each arm starts cold so the comparison is
# between the knobs, not between how warm the model dir happened to be.
base_env() {
    echo "COLI_MODEL=$MODEL"
    echo "MALLOC_ARENA_MAX=2"
    echo "COLI_ECACHE_GB=$ECACHE"
    echo "COLI_BENCH_STEPS=$STEPS"
    echo "COLI_ROUTE_STATS_PERSIST=0"
    echo "COLI_DEBUG=1"
}

# Per-arm knobs. `baseline` is deliberately empty: every adaptive knob defaults
# to the historical behavior, so plain defaults *are* the published study's
# configuration.
arm_env() {
    case "$1" in
    baseline) ;;
    improved | gpu)
        echo "COLI_DIRECT=1"          # O_DIRECT lane, bypass page cache
        # COLI_REGBUF deliberately NOT set: the reachability pass showed no code
        # reads it, so this arm published a nine-knob bundle that was really eight.
        echo "COLI_IO_TUNE=1"         # adaptive iowq_max_workers
        echo "COLI_LANE_BALANCE=1"    # LaneBalancer placement override
        echo "COLI_SHAPE_SPECIALIZE=1" # probe-then-memoize matmul dispatch
        echo "COLI_HYPER_SCHED=1"     # co-activation io-claim grouping
        echo "COLI_PREFETCH_TUNE=1"   # EWMA prefetch-distance tuner
        echo "COLI_ENTROPY_ADAPT=1"   # entropy-adaptive prefetch breadth
        echo "COLI_REPLICATE_K=8"     # hot GPU residents also warmed in RAM
        # plain `[ ... ] && echo` would return 1 on the `improved` arm and the
        # caller's `|| exit 1` would kill the whole sweep — keep it an if.
        if [ "$1" = gpu ]; then echo "COLI_GPU=1"; fi
        ;;
    *)
        echo "unknown arm: $1" >&2
        return 1
        ;;
    esac
}

for arm in $ARMS; do
    bin=$BIN_CPU
    [ "$arm" = gpu ] && bin=$BIN_GPU
    if [ ! -x "$bin" ]; then
        echo "== arm $arm SKIPPED: no binary at $bin" | tee -a "$OUT/summary.txt"
        continue
    fi

    env_lines=$( { base_env; arm_env "$arm"; } ) || exit 1
    printf '%s\n' "$env_lines" >"$OUT/$arm.env"

    echo "== arm $arm: $bin  batches=[$BATCHES] steps=$STEPS" | tee -a "$OUT/summary.txt"
    # Run under a memory-capped cgroup so a runaway arm is killed in its own
    # scope instead of letting the global OOM killer pick a victim elsewhere on
    # the box (this machine also hosts a VM). MEMMAX= disables the cap.
    cap=()
    if [ -n "${MEMMAX:-}" ]; then
        cap=(systemd-run --user --scope -p "MemoryMax=$MEMMAX" -p MemorySwapMax=0 --quiet)
    fi
    # shellcheck disable=SC2046  # word splitting is how we turn lines into env args
    env $(printf '%s ' $env_lines) "${cap[@]}" \
        scripts/runstat.py "$OUT/$arm.stat" "$bin" bench $BATCHES \
        >"$OUT/$arm.out" 2>"$OUT/$arm.err"
    rc=$?

    echo "   rc=$rc" | tee -a "$OUT/summary.txt"
    cat "$OUT/$arm.out" | tee -a "$OUT/summary.txt"
    grep -E "wall_s|peak_rss_gb|major_faults" "$OUT/$arm.stat" 2>/dev/null \
        | sed 's/^/   /' | tee -a "$OUT/summary.txt"
    [ $rc -ne 0 ] && tail -20 "$OUT/$arm.err" | tee -a "$OUT/summary.txt"
    echo | tee -a "$OUT/summary.txt"
done

echo "results in $OUT"
