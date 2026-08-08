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
# Draft depth γ for the speculative arms (`spec`, `spec_gpu`). 4 is the floor of
# the production stack's guidance (GLM-5.2 reports 2.76 accepted per round at γ=4
# — already 82 % of that configuration's 1+γ ceiling). The default keeps the eight
# other knobs' published numbers unchanged when an operator chooses to leave
# speculation off.
DRAFT=${COLI_DRAFT:-4}

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
    # The speculative arm: the same knob bundle as `improved`, plus the MTP draft
    # depth. `bench --draft $DRAFT` enables batched verify — every step drafts γ
    # tokens per sequence, then verifies B·(1+γ) rows in one forward, sharing one
    # expert-read union per step. Off when the checkpoint has no MTP head; the
    # bench harness refuses loudly rather than silently decoding non-speculatively.
    # Bit-identical to greedy per-token decode by construction (`accept_run` is the
    # one definition). Compares **directly** to `improved`: |spec| − |improved| is
    # the speculation speedup. `tok/s` in the table is verified tokens, `agg
    # tok/s` is forward-issued tokens (the two diverge with the accept length).
    spec)
        echo "COLI_DIRECT=1"
        echo "COLI_IO_TUNE=1"
        echo "COLI_LANE_BALANCE=1"
        echo "COLI_SHAPE_SPECIALIZE=1"
        echo "COLI_HYPER_SCHED=1"
        echo "COLI_PREFETCH_TUNE=1"
        echo "COLI_ENTROPY_ADAPT=1"
        echo "COLI_REPLICATE_K=8"
        echo "COLI_DRAFT=$DRAFT"       # MTP speculative batched verify
        ;;
    spec_gpu)
        echo "COLI_DIRECT=1"
        echo "COLI_IO_TUNE=1"
        echo "COLI_LANE_BALANCE=1"
        echo "COLI_SHAPE_SPECIALIZE=1"
        echo "COLI_HYPER_SCHED=1"
        echo "COLI_PREFETCH_TUNE=1"
        echo "COLI_ENTROPY_ADAPT=1"
        echo "COLI_REPLICATE_K=8"
        echo "COLI_DRAFT=$DRAFT"
        echo "COLI_GPU=1"
        ;;
    # The batched router look-ahead arm: prefetch the cross-row union of the next
    # layer's top experts (gated by `COLI_ROUTER_LOOKAHEAD_BATCH`). `width` is the
    # per-step IO budget (never `width × s_n`), deduped so the prefetch depth
    # matches disk's idle window. Recall against the next layer's actual routed
    # set is what `COLI_PREDICT_EVAL=1` measures here; bit-identical either way
    # because the authoritative router still decides.
    lookahead_batch)
        echo "COLI_DIRECT=1"
        echo "COLI_IO_TUNE=1"
        echo "COLI_LANE_BALANCE=1"
        echo "COLI_SHAPE_SPECIALIZE=1"
        echo "COLI_HYPER_SCHED=1"
        echo "COLI_PREFETCH_TUNE=1"
        echo "COLI_ENTROPY_ADAPT=1"
        echo "COLI_REPLICATE_K=8"
        echo "COLI_ROUTER_LOOKAHEAD_BATCH=1"   # on by default; explicit for measurement
        echo "COLI_PREDICT_EVAL=1"             # recall/precision scoreboard
        echo "COLI_PREDICT_EVAL_N=8"           # 8 arms × s_n batched recall
        ;;
    *)
        echo "unknown arm: $1" >&2
        return 1
        ;;
    esac
}

# Extra `peregrine bench` positional args per arm. Used by the speculative arms to
# pass `--draft N` (parsed by the bench loop's own `draft_depth`). `--` compounds the
# splitter to word-splitting; otherwise the literal `--draft N` is one identity token
# to bash.
arm_bench_extra() {
    case "$1" in
    spec | spec_gpu) echo "--draft $DRAFT" ;;
    *) echo "" ;;
    esac
}

for arm in $ARMS; do
    bin=$BIN_CPU
    case "$arm" in
        gpu | spec_gpu) bin=$BIN_GPU ;;
    esac
    if [ ! -x "$bin" ]; then
        echo "== arm $arm SKIPPED: no binary at $bin" | tee -a "$OUT/summary.txt"
        continue
    fi

    env_lines=$( { base_env; arm_env "$arm"; } ) || exit 1
    printf '%s\n' "$env_lines" >"$OUT/$arm.env"
    extra=$(arm_bench_extra "$arm")

    echo "== arm $arm: $bin  batches=[$BATCHES] steps=$STEPS extra=[$extra]" | tee -a "$OUT/summary.txt"
    # Run under a memory-capped cgroup so a runaway arm is killed in its own
    # scope instead of letting the global OOM killer pick a victim elsewhere on
    # the box (this machine also hosts a VM). MEMMAX= disables the cap.
    cap=()
    if [ -n "${MEMMAX:-}" ]; then
        cap=(systemd-run --user --scope -p "MemoryMax=$MEMMAX" -p MemorySwapMax=0 --quiet)
    fi
    # shellcheck disable=SC2046  # word splitting is how we turn lines into env args
    env $(printf '%s ' $env_lines) "${cap[@]}" \
        scripts/runstat.py "$OUT/$arm.stat" "$bin" bench $BATCHES $extra \
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
