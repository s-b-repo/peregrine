#!/usr/bin/env bash
set -euo pipefail

MODEL=${COLI_MODEL:-/srv/m-sdd/GLM-5.3-Flash-4ssd}
BIN=${BIN:-target-cuda/release/peregrine}
OUT=${1:-bench-data/glm53-cuda-matrix}
REPEATS=${REPEATS:-1}
mkdir -p "$OUT"

[[ -x "$BIN" ]] || { printf 'missing CUDA benchmark binary: %s\n' "$BIN" >&2; exit 1; }
[[ -f "$MODEL/config.json" ]] || { printf 'missing GLM-5.3 model: %s\n' "$MODEL" >&2; exit 1; }

common=(
    env
    -u COLI_GPU_DENSE
    -u COLI_GPU_DENSE_HEADROOM_MB
    -u COLI_GPU_DENSE_MAX_BATCH
    -u COLI_CUDA_FUSED_REDUCE
    -u COLI_DRAFT
    COLI_MODEL="$MODEL"
    COLI_STREAM=1
    COLI_GPU=1
    COLI_GPU_INT4=1
    COLI_GPU_F32_FRAC=0
    COLI_GPU_PINNED=1
    COLI_GPU_DEVICES=0
    COLI_GPU_EXPERT_HEADROOM_MB=1792
    COLI_CUDA_GRAPH=1
    COLI_MOE_ENGINE=concurrent
    COLI_IO_BATCH=8
    COLI_IO_DEVICE_SCHED=1
    COLI_SSD_AWARE_SCHED=1
    COLI_PREFETCH_STALE_DROP=1
    COLI_FUSE_PREFILL=1
    COLI_KV_BUDGET_MB=6000
    COLI_PAR_THREADS=12
    MALLOC_ARENA_MAX=2
)

gpu_dense=(
    COLI_GPU_DENSE=1
    COLI_GPU_DENSE_HEADROOM_MB=256
    COLI_GPU_DENSE_MAX_BATCH=1
)

gpu_dense_b16=(
    COLI_GPU_DENSE=1
    COLI_GPU_DENSE_HEADROOM_MB=256
    COLI_GPU_DENSE_MAX_BATCH=16
)

run_arm() {
    local name=$1
    local batch=$2
    local steps=$3
    local cache_gb=$4
    local draft=$5
    shift 5
    local rep
    for ((rep = 1; rep <= REPEATS; rep++)); do
        local tag
        tag=$(printf '%s/rep%02d' "$name" "$rep")
        mkdir -p "$OUT/$tag"
        {
            printf 'arm=%s\nbatch=%s\nsteps=%s\necache_gb=%s\ndraft=%s\n' "$name" "$batch" "$steps" "$cache_gb" "$draft"
            printf 'env=%q ' "$@"
            printf '\n'
        } >"$OUT/$tag/arm.env"
        "${common[@]}" \
            COLI_BENCH_STEPS="$steps" \
            COLI_ECACHE_GB="$cache_gb" \
            COLI_DRAFT="$draft" \
            "$@" \
            "$BIN" bench "$batch" \
            >"$OUT/$tag/bench.out" \
            2>"$OUT/$tag/bench.err"
    done
}

run_arm original_b1_step1 1 1 6 0
run_arm fused_b1_step1 1 1 6 0 COLI_CUDA_FUSED_REDUCE=1
run_arm new_b1_step1 1 1 6 0 COLI_CUDA_FUSED_REDUCE=1 "${gpu_dense[@]}"
run_arm new_b1_sustained 1 3 6 0 COLI_CUDA_FUSED_REDUCE=1 "${gpu_dense[@]}"
run_arm no_shared_dense_b1_sustained 1 3 6 0 COLI_CUDA_FUSED_REDUCE=1
run_arm original_b16 16 1 6 0
run_arm new_b16 16 1 6 0 COLI_CUDA_FUSED_REDUCE=1 "${gpu_dense_b16[@]}"
run_arm adaptive_b16 16 1 6 0 COLI_CUDA_FUSED_REDUCE=1 "${gpu_dense[@]}"
run_arm mtp_depth1_b1 1 1 6 1 COLI_CUDA_FUSED_REDUCE=1 "${gpu_dense[@]}"
run_arm ecache12_b1_sustained 1 3 12 0 COLI_CUDA_FUSED_REDUCE=1 "${gpu_dense[@]}"

printf 'results=%s\n' "$OUT"
