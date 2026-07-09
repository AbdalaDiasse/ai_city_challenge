#!/bin/bash
# Download Qwen3-VL-8B model weights to $WORK/hf_cache (login node only — needs internet).
#
# Run once before submitting train_sft.slurm:
#   bash track3/download_model.sh
#
# What this downloads:
#   unsloth/Qwen3-VL-8B-Instruct               — base model config / tokenizer
#   unsloth/qwen3-vl-8b-instruct-unsloth-bnb-4bit — Unsloth's pre-quantized 4-bit weights
#   (FastVisionModel.from_pretrained redirects to the bnb-4bit repo when load_in_4bit=True)

set -euo pipefail

WORK="${WORK:-/leonardo_work/AIH4A_syrate}"
export HF_HOME="$WORK/hf_cache"

echo "HF_HOME = $HF_HOME"
echo

module load profile/deeplrn
module load cineca-ai/4.3.0
source /leonardo/home/userexternal/adiasse0/venvs/unsloth/bin/activate
unset PYTHONPATH

echo "=== Downloading unsloth/Qwen3-VL-8B-Instruct ==="
hf download unsloth/Qwen3-VL-8B-Instruct

echo
echo "=== Downloading unsloth/qwen3-vl-8b-instruct-unsloth-bnb-4bit ==="
hf download unsloth/qwen3-vl-8b-instruct-unsloth-bnb-4bit

echo
echo "Done. Both models cached under $HF_HOME"
echo "You can now submit: sbatch track3/train_sft.slurm"
