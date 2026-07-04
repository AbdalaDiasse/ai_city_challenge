#!/bin/bash
# Phase 1 — Download TAR annotations and videos
# Usage: bash track3/download_data.sh
# Run from the repo root on a login node (no GPU needed).

set -euo pipefail

WORK=/leonardo_work/AIH4A_syrate
DATA_DIR=$WORK/data/track3
HF_CACHE=$WORK/hf_cache

mkdir -p "$DATA_DIR/annotations"
mkdir -p "$DATA_DIR/videos"
mkdir -p "$HF_CACHE"

export HF_HOME=$HF_CACHE

echo "==> Installing download dependencies..."
pip install -q huggingface_hub

# hf_transfer is a Rust wheel that only builds on Python 3.8+.
# It's a speed-only optimisation — skip silently if unavailable.
if pip install -q hf_transfer 2>/dev/null; then
    export HF_XET_HIGH_PERFORMANCE=1   # replaces deprecated HF_HUB_ENABLE_HF_TRANSFER
    echo "    hf_transfer/xet enabled (faster downloads)"
else
    echo "    hf_transfer not available — using standard download"
fi

echo "==> Downloading TAR annotations from HuggingFace..."
hf download nvidia/PhysicalAI-Traffic-Anomaly-Reasoning \
    --repo-type dataset \
    --local-dir "$DATA_DIR/annotations"

echo "==> Annotations saved to $DATA_DIR/annotations"
ls "$DATA_DIR/annotations"

# The HF dataset ships a download_videos.py script.
# It pulls ~150 GB from the original source repos.
DOWNLOAD_SCRIPT="$DATA_DIR/annotations/download_videos.py"
if [ -f "$DOWNLOAD_SCRIPT" ]; then
    echo "==> Starting video download (~150 GB, this will take a while)..."
    python "$DOWNLOAD_SCRIPT" --out "$DATA_DIR/videos"
    echo "==> Videos saved to $DATA_DIR/videos"
else
    echo "WARNING: $DOWNLOAD_SCRIPT not found."
    echo "Check the dataset repo for the video download instructions."
fi

echo "==> Done. Verify with:"
echo "    ls $DATA_DIR/annotations"
echo "    du -sh $DATA_DIR/videos"
