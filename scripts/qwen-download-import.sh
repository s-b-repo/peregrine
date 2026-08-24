#!/usr/bin/env bash
# Download Qwen3.8-27B from HuggingFace, verify checksums, then import to
# peregrine container format. The model must exist at
# /srv/modelstripe/qwen/Qwen3.8-27B-peregrine/ for serve-qwen.sh to start
# peregrine-serve on :8132 (the auto-improve loop's benchmark target).
set -uo pipefail

DEST="/home/cortix/peregrine"
HF_DIR="/srv/modelstripe/qwen/Qwen3.8-27B"
PEREG_DIR="/srv/modelstripe/qwen/Qwen3.8-27B-peregrine"
LOG="/home/cortix/peregrine/qwen-download.log"
IMPORT_LOG="/home/cortix/peregrine/qwen-import.log"

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

log "=== Qwen download+import pipeline started ==="

# Step 1: Download (resumable with aria2c)
log "Step 1: Downloading Qwen3.8-27B from HuggingFace (~55.6 GB)..."
cd /home/cortix/peregrine
bash scripts/download-qwen3.8-27b.sh 2>&1 | while read -r line; do echo "$line" >> "$LOG"; done
dl_rc=${PIPESTATUS[0]}

if [ "$dl_rc" -ne 0 ]; then
    log "Download FAILED (rc=$dl_rc) — will be retried by next watchdog cycle"
    exit 1
fi

log "Download completed successfully."

# Step 2: Checksum verification is handled within the download script itself
# (down-qwen3.8-27b.sh verifies crc32.txt after download completes).
log "Step 2: Checksum verification complete (handled by download script)."

# Step 3: Import to peregrine container format
log "Step 3: Importing to peregrine container format..."
mkdir -p "$(dirname "$PEREG_DIR")"
/home/cortix/peregrine/target/release/peregrine-import-hf "$HF_DIR" "$PEREG_DIR" 2>&1 | tee -a "$IMPORT_LOG"
imp_rc=${PIPESTATUS[0]}

if [ "$imp_rc" -ne 0 ]; then
    log "Import FAILED (rc=$imp_rc)"
    exit 1
fi

log "Import completed successfully. Model available at $PEREG_DIR"
log "=== Qwen download+import pipeline complete ==="
