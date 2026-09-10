#!/usr/bin/env bash
# Setup script for GLM-5.3-Flash-DFlash2 with peregrine
# 
# This script:
# 1. Downloads the model from Hugging Face
# 2. Imports it to peregrine container format
# 3. Optionally reshard across multiple drives
# 4. Runs inference/serve

set -uo pipefail

# Configuration - MODIFY THESE FOR YOUR SETUP
HF_MODEL_ID="incoai/GLM-5.3-Flash-DFlash2"
HF_CACHE_DIR="/tmp/hf-cache"
DOWNLOAD_DIR="/tmp/glm53-flash-dflash2-hf"
PEREGRINE_MODEL_DIR="/tmp/glm53-flash-dflash2-peregrine"
MODEL_ID="glm-5.3-flash-dflash2"

# Multi-drive configuration (uncomment and modify for your drives)
# REShard_GROUPS="sda:500,sdb:500,sdc:500,sdd:500"  # drive:weight in MB/s
# RESHARD_OUT="/srv/glm53-dflash2-resharded"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() { echo -e "${GREEN}[$(date '+%F %T')]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date '+%F %T')] WARNING:${NC} $*"; }
error() { echo -e "${RED}[$(date '+%F %T')] ERROR:${NC} $*"; exit 1; }

# Check if running in the right directory
if [[ ! -f "Cargo.toml" ]] || [[ ! -d "crates/peregrine-tools" ]]; then
    error "Run this script from the peregrine repository root"
fi

# Build peregrine tools if needed
build_tools() {
    log "Building peregrine-import-hf..."
    cargo build --release -p peregrine-tools --bin peregrine-import-hf || error "Failed to build peregrine-import-hf"
    
    log "Building peregrine-reshard..."
    cargo build --release -p peregrine-tools --bin peregrine-reshard || error "Failed to build peregrine-reshard"
    
    log "Building peregrine-engine..."
    cargo build --release -p peregrine-engine --bin peregrine || error "Failed to build peregrine-engine"
    
    log "Building peregrine-serve..."
    cargo build --release -p peregrine-serve --bin peregrine-serve || error "Failed to build peregrine-serve"
}

# Download model from Hugging Face
download_model() {
    log "Downloading $HF_MODEL_ID from Hugging Face..."
    mkdir -p "$DOWNLOAD_DIR"
    
    # Use hf (huggingface_hub) if available, otherwise use curl
    if command -v hf &> /dev/null; then
        log "Using hf to download..."
        hf download "$HF_MODEL_ID" \
            --local-dir "$DOWNLOAD_DIR" || error "Download failed"
    else
        log "Using curl to download (this may take a while)..."
        # Download config.json
        curl -L "https://huggingface.co/$HF_MODEL_ID/resolve/main/config.json" \
            -o "$DOWNLOAD_DIR/config.json" || error "Failed to download config.json"
        
        # Download model.safetensors
        curl -L "https://huggingface.co/$HF_MODEL_ID/resolve/main/model.safetensors" \
            -o "$DOWNLOAD_DIR/model.safetensors" || error "Failed to download model.safetensors"
        
        # Download tokenizer files if they exist
        for file in tokenizer.json tokenizer_config.json generation_config.json chat_template.jinja; do
            curl -L "https://huggingface.co/$HF_MODEL_ID/resolve/main/$file" \
                -o "$DOWNLOAD_DIR/$file" 2>/dev/null || true
        done
    fi
    
    log "Download complete. Model at $DOWNLOAD_DIR"
    ls -lh "$DOWNLOAD_DIR"
}

# Import to peregrine container format
import_model() {
    log "Importing model to peregrine format at $PEREGRINE_MODEL_DIR..."
    mkdir -p "$(dirname "$PEREGRINE_MODEL_DIR")"
    
    # Shard size: 2GB per shard (adjust based on your needs)
    SHARD_BYTES=$((2 * 1024 * 1024 * 1024))
    
    target/release/peregrine-import-hf "$DOWNLOAD_DIR" "$PEREGRINE_MODEL_DIR" --shard-bytes "$SHARD_BYTES" \
        2>&1 | tee /tmp/peregrine-import.log
    
    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        error "Import failed. Check /tmp/peregrine-import.log"
    fi
    
    log "Import complete. Model at $PEREGRINE_MODEL_DIR"
    ls -lh "$PEREGRINE_MODEL_DIR"
}

# Reshard across multiple drives (optional)
reshard_model() {
    if [[ -z "${RESHARD_GROUPS:-}" ]] || [[ -z "${RESHARD_OUT:-}" ]]; then
        warn "RESHARD_GROUPS and RESHARD_OUT not set. Skipping reshard."
        warn "To enable multi-drive, set REShard_GROUPS and RESHARD_OUT in the script config."
        return 0
    fi
    
    log "Resharding model across drives..."
    log "Groups: $RESHARD_GROUPS"
    log "Output: $RESHARD_OUT"
    
    mkdir -p "$(dirname "$RESHARD_OUT")"
    
    target/release/peregrine-reshard \
        --model "$PEREGRINE_MODEL_DIR" \
        --out "$RESHARD_OUT" \
        --groups "$RESHARD_GROUPS" \
        --verify \
        2>&1 | tee /tmp/peregrine-reshard.log
    
    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        error "Reshard failed. Check /tmp/peregrine-reshard.log"
    fi
    
    log "Reshard complete. Model at $RESHARD_OUT"
    ls -lh "$RESHARD_OUT"
    
    # Update MODEL_DIR to point to resharded model
    PEREGRINE_MODEL_DIR="$RESHARD_OUT"
}

# Run inference demo
run_demo() {
    log "Running inference demo..."
    COLI_MODEL="$PEREGRINE_MODEL_DIR" \
    target/release/peregrine demo
}

# Run benchmark
run_bench() {
    log "Running benchmark..."
    COLI_MODEL="$PEREGRINE_MODEL_DIR" \
    target/release/peregrine bench 1 4 16
}

# Start serve mode (OpenAI-compatible API)
run_serve() {
    log "Starting peregrine-serve on port 8080..."
    log "Model ID: $MODEL_ID"
    log "Model dir: $PEREGRINE_MODEL_DIR"
    
    COLI_MODEL="$PEREGRINE_MODEL_DIR" \
    target/release/peregrine-serve \
        --model-id "$MODEL_ID" \
        --host 0.0.0.0 \
        --port 8080 \
        2>&1 | tee /tmp/peregrine-serve.log
}

# Build automaton for better routing (optional, run once)
build_automaton() {
    log "Building automaton for better expert routing..."
    COLI_MODEL="$PEREGRINE_MODEL_DIR" \
    target/release/peregrine build-automaton "$PEREGRINE_MODEL_DIR" 256 \
        2>&1 | tee /tmp/peregrine-automaton.log
    
    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        log "Automaton built successfully"
    else
        warn "Automaton build failed (non-critical). Check /tmp/peregrine-automaton.log"
    fi
}

# Main execution
main() {
    log "=== GLM-5.3-Flash-DFlash2 peregrine Setup ==="
    log "Model: $HF_MODEL_ID"
    log "Peregrine model dir: $PEREGRINE_MODEL_DIR"
    
    # Step 1: Build tools
    build_tools
    
    # Step 2: Download model
    download_model
    
    # Step 3: Import to peregrine format
    import_model
    
    # Step 4: Optionally reshard across drives
    reshard_model
    
    # Step 5: Build automaton (optional but recommended)
    build_automaton
    
    log "=== Setup Complete ==="
    log "Model ready at: $PEREGRINE_MODEL_DIR"
    log ""
    log "To run inference demo:"
    log "  COLI_MODEL=\"$PEREGRINE_MODEL_DIR\" target/release/peregrine demo"
    log ""
    log "To run benchmark:"
    log "  COLI_MODEL=\"$PEREGRINE_MODEL_DIR\" target/release/peregrine bench 1 4 16"
    log ""
    log "To start OpenAI-compatible server:"
    log "  COLI_MODEL=\"$PEREGRINE_MODEL_DIR\" target/release/peregrine-serve --model-id \"$MODEL_ID\""
    log ""
    log "Environment variables for tuning:"
    log "  COLI_STREAM=1          # Enable expert streaming (for large models)"
    log "  COLI_ECACHE_GB=4       # Warm expert cache size in GB"
    log "  COLI_GPU=1             # Enable GPU expert tier (requires CUDA)"
    log "  COLI_GPU_DENSE=1       # Enable dense MLP on GPU"
    log "  COLI_IO_DEVICE_SCHED=1 # Device-aware I/O scheduling"
    log "  COLI_SSD_AWARE_SCHED=1 # SSD-aware scheduling"
}

# Parse arguments
case "${1:-all}" in
    build)
        build_tools
        ;;
    download)
        download_model
        ;;
    import)
        import_model
        ;;
    reshard)
        reshard_model
        ;;
    demo)
        run_demo
        ;;
    bench)
        run_bench
        ;;
    serve)
        run_serve
        ;;
    automaton)
        build_automaton
        ;;
    all)
        main
        ;;
    *)
        echo "Usage: $0 {build|download|import|reshard|demo|bench|serve|automaton|all}"
        echo ""
        echo "Commands:"
        echo "  build      - Build peregrine tools"
        echo "  download   - Download model from Hugging Face"
        echo "  import     - Import model to peregrine format"
        echo "  reshard    - Reshard model across drives (requires REShard_GROUPS and RESHARD_OUT)"
        echo "  demo       - Run inference demo"
        echo "  bench      - Run benchmark"
        echo "  serve      - Start OpenAI-compatible server"
        echo "  automaton  - Build automaton for expert routing"
        echo "  all        - Run all steps (default)"
        exit 1
        ;;
esac