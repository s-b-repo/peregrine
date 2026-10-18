#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_ID="${HF_ID:-ornith-ai/Ornith-1.5-35B-A3B}"
HF_DIR="${HF_DIR:-/srv/m-sdd/Ornith-1.5-35B-A3B-hf}"
IMPORT_DIR="${IMPORT_DIR:-/srv/m-sdd/Ornith-1.5-35B-A3B-peregrine}"
STAGE_DIR="${STAGE_DIR:-/srv/m-sdb/ornith35-reshard}"
MODEL_DIR="${MODEL_DIR:-/srv/m-sdd/Ornith-1.5-35B-A3B-4drive}"
BIN_DIR="${BIN_DIR:-$REPO_DIR/target-cuda/release}"
GROUP_SPEC="sda:470,sdb:540,sdc:85,sdd:570"
SHARD_BYTES="${SHARD_BYTES:-2147483648}"
LAYOUT_ID="${LAYOUT_ID:-$(date '+%Y%m%d%H%M%S')-$$}"
DRIVES=(sda sdb sdc sdd)

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
die() { printf '[%s] ERROR: %s\n' "$(date '+%F %T')" "$*" >&2; exit 1; }

require_build_tools() {
    command -v cargo >/dev/null || die "cargo is required"
    command -v nvidia-smi >/dev/null || die "nvidia-smi is required"
}

require_download_tools() {
    command -v hf >/dev/null || die "hf is required"
}

check_mounts() {
    command -v findmnt >/dev/null || die "findmnt is required"
    local drive target source
    declare -A seen=()
    for drive in "${DRIVES[@]}"; do
        local path="/srv/m-$drive"
        [[ -d "$path" ]] || die "$path is missing"
        target="$(findmnt -n -o TARGET --target "$path")"
        [[ "$target" == "$path" ]] || die "$path is not a mount point (resolved to $target)"
        [[ -w "$path" ]] || die "$path is not writable"
        source="$(findmnt -n -o SOURCE --target "$path")"
        [[ -z "${seen[$source]+x}" ]] || die "$path and ${seen[$source]:-another drive} resolve to the same device $source"
        seen[$source]="$path"
    done
}

build() {
    require_build_tools
    log "building CUDA release binaries"
    CARGO_TARGET_DIR="$REPO_DIR/target-cuda" cargo build --release --features cuda \
        -p peregrine-engine -p peregrine-serve -p peregrine-tools
}

download() {
    require_build_tools
    require_download_tools
    check_mounts
    mkdir -p "$HF_DIR"
    log "downloading $HF_ID to $HF_DIR"
    hf download "$HF_ID" --local-dir "$HF_DIR"
    [[ -f "$HF_DIR/config.json" ]] || die "download finished without config.json"
    [[ -f "$HF_DIR/model.safetensors.index.json" ]] || die "download finished without model.safetensors.index.json"
}

import_model() {
    require_build_tools
    check_mounts
    [[ -f "$HF_DIR/config.json" ]] || die "missing $HF_DIR/config.json; run download first"
    [[ -x "$BIN_DIR/peregrine-import-hf" ]] || die "missing importer; run build first"
    if [[ -f "$IMPORT_DIR/config.json" ]]; then
        log "imported model already ready at $IMPORT_DIR"
        return 0
    fi
    [[ ! -e "$IMPORT_DIR" ]] || die "$IMPORT_DIR exists but is incomplete; move it aside before retrying"
    local incoming="$IMPORT_DIR.incoming.$LAYOUT_ID"
    [[ ! -e "$incoming" ]] || die "$incoming already exists"
    mkdir -p "$incoming"
    log "importing NVIDIA BF16 checkpoint into Peregrine int4/int8 format"
    if ! "$BIN_DIR/peregrine-import-hf" "$HF_DIR" "$incoming" --shard-bytes "$SHARD_BYTES"; then
        die "import failed; partial output remains at $incoming"
    fi
    [[ -f "$incoming/config.json" ]] || die "import produced no config.json"
    mv "$incoming" "$IMPORT_DIR"
    log "imported model ready at $IMPORT_DIR"
}

layout_paths() {
    local -a paths
    [[ -f "$MODEL_DIR/config.json" && -f "$MODEL_DIR/model_paths.json" && -f "$MODEL_DIR/manifest.json" ]] || return 1
    [[ "$(jq -r '.model_type // empty' "$MODEL_DIR/config.json")" == qwen3_5_moe ]] || return 1
    jq -e '.paths | type == "array" and length == 4 and all(type == "string")' "$MODEL_DIR/model_paths.json" >/dev/null || return 1
    mapfile -t paths < <(jq -r '.paths[]' "$MODEL_DIR/model_paths.json")
    [[ ${#paths[@]} -eq 4 ]] || return 1
    local path
    declare -A seen=()
    for path in "${paths[@]}"; do
        [[ "$path" == /srv/m-*/ornith35-* ]] || return 1
        [[ -f "$path/.peregrine-ornith35-layout" ]] || return 1
        [[ -z "${seen[$path]+x}" ]] || return 1
        seen[$path]=1
    done
    return 0
}

layout_ready() {
    layout_paths >/dev/null 2>&1
}

reshard_model() {
    require_build_tools
    check_mounts
    [[ -f "$IMPORT_DIR/config.json" ]] || die "missing imported model; run import first"
    [[ -x "$BIN_DIR/peregrine-reshard" ]] || die "missing resharder; run build first"
    [[ "$LAYOUT_ID" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid LAYOUT_ID"
    [[ ! -e "$MODEL_DIR" ]] || die "$MODEL_DIR already exists; use clean-layout or set MODEL_DIR to a new path"
    local incoming="$STAGE_DIR.incoming.$LAYOUT_ID"
    [[ ! -e "$incoming" ]] || die "$incoming already exists"
    local drive destination
    for drive in "${DRIVES[@]}"; do
        destination="/srv/m-$drive/ornith35-$LAYOUT_ID"
        [[ ! -e "$destination" ]] || die "$destination already exists"
    done
    mkdir -p "$incoming"
    log "resharding experts across four storage devices: $GROUP_SPEC"
    if ! "$BIN_DIR/peregrine-reshard" --model "$IMPORT_DIR" --out "$incoming" --groups "$GROUP_SPEC" --verify; then
        die "reshard failed; verified staging remains at $incoming"
    fi
    for drive in "${DRIVES[@]}"; do
        destination="/srv/m-$drive/ornith35-$LAYOUT_ID"
        mkdir -p "$destination"
        mv "$incoming"/experts-l*-"$drive".safetensors "$destination/"
        if [[ "$drive" == sda ]]; then
            mv "$incoming"/trunk-sda-*.safetensors "$destination/"
        fi
        printf 'layout=%s\ndrive=%s\nmodel=%s\n' "$LAYOUT_ID" "$drive" "$MODEL_DIR" > "$destination/.peregrine-ornith35-layout"
    done
    local model_incoming="$MODEL_DIR.incoming.$LAYOUT_ID"
    [[ ! -e "$model_incoming" ]] || die "$model_incoming already exists"
    mkdir -p "$model_incoming"
    local side
    for side in config.json tokenizer.json generation_config.json tokenizer_config.json chat_template.jinja; do
        [[ -f "$IMPORT_DIR/$side" ]] && cp "$IMPORT_DIR/$side" "$model_incoming/$side"
    done
    mv "$incoming/manifest.json" "$model_incoming/manifest.json"
    printf '{"paths":["/srv/m-sda/ornith35-%s","/srv/m-sdb/ornith35-%s","/srv/m-sdc/ornith35-%s","/srv/m-sdd/ornith35-%s"]}\n' \
        "$LAYOUT_ID" "$LAYOUT_ID" "$LAYOUT_ID" "$LAYOUT_ID" > "$model_incoming/model_paths.json"
    mv "$model_incoming" "$MODEL_DIR"
    rmdir "$incoming" 2>/dev/null || true
    layout_ready || die "new layout failed readiness validation"
    log "four-drive model ready at $MODEL_DIR"
}

clean_layout() {
    [[ "${CONFIRM_CLEAN:-0}" == 1 ]] || die "set CONFIRM_CLEAN=1"
    check_mounts
    layout_paths || die "$MODEL_DIR is not a complete Ornith four-drive layout"
    local stamp path quarantine
    local -a paths
    stamp="$(date '+%Y%m%d%H%M%S')-$$"
    mapfile -t paths < <(jq -r '.paths[]' "$MODEL_DIR/model_paths.json")
    quarantine="$MODEL_DIR.disabled.$stamp"
    [[ ! -e "$quarantine" ]] || die "$quarantine already exists"
    mv "$MODEL_DIR" "$quarantine"
    for path in "${paths[@]}"; do
        quarantine="$path.disabled.$stamp"
        [[ ! -e "$quarantine" ]] || die "$quarantine already exists"
        mv "$path" "$quarantine"
    done
    log "quarantined live model at $MODEL_DIR.disabled.$stamp"
}

status() {
    check_mounts
    printf 'HF source: %s\n' "$([[ -f "$HF_DIR/config.json" ]] && echo ready || echo missing)"
    printf 'Imported: %s\n' "$([[ -f "$IMPORT_DIR/config.json" ]] && echo ready || echo missing)"
    if layout_ready; then
        printf '4-drive layout: ready (%s)\n' "$MODEL_DIR"
    else
        printf '4-drive layout: missing or incomplete\n'
    fi
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
    df -h /srv/m-sda /srv/m-sdb /srv/m-sdc /srv/m-sdd
}

serve() {
    exec "$REPO_DIR/scripts/serve-ornith35-cuda.sh" "$@"
}

parity_gate() {
    [[ -x "$BIN_DIR/peregrine" ]] || die "missing engine; run build first"
    [[ -n "${REFERENCE_JSON:-}" ]] || die "REFERENCE_JSON must point to an HF teacher-forcing reference dump"
    [[ -f "$REFERENCE_JSON" ]] || die "reference dump not found: $REFERENCE_JSON"
    local -a args=(flip-rate "$MODEL_DIR" --reference-json "$REFERENCE_JSON")
    if [[ -n "${PARITY_TEXT:-}" ]]; then
        [[ -f "$PARITY_TEXT" ]] || die "parity corpus not found: $PARITY_TEXT"
        args+=(--text "$PARITY_TEXT")
    fi
    log "running HF teacher-forcing parity gate"
    "$BIN_DIR/peregrine" "${args[@]}" || die "HF teacher-forcing parity gate failed"
}

case "${1:-all}" in
    build) build ;;
    download) download ;;
    import) import_model ;;
    reshard) reshard_model ;;
    clean-layout) clean_layout ;;
    serve) serve ;;
    status) status ;;
    parity) parity_gate ;;
    all)
        build
        download
        import_model
        reshard_model
        parity_gate
        status
        ;;
    *)
        printf 'usage: %s {build|download|import|reshard|parity|clean-layout|serve|status|all}\n' "$0" >&2
        exit 2
        ;;
esac
