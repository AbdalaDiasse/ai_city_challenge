#!/bin/bash
#SBATCH --job-name=track3_sft_fps2
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=AIH4A_syrate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00
#SBATCH --output=logs/track3_sft_fps2/%x_%j.out
#SBATCH --error=logs/track3_sft_fps2/%x_%j.err

# ── environment ──────────────────────────────────────────────────────────────
module load cuda/12.6
module load profile/deeplrn
module load cineca-ai/4.3.0

unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"

source /leonardo/home/userexternal/adiasse0/venvs/unsloth/bin/activate
unset PYTHONPATH

export WORK=/leonardo_work/AIH4A_syrate
export HF_HOME=$WORK/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

export FPS=2.0
export RUN=sft_bf16_fps2_v1

# ── step 1: build fps=2 dataset (separate files, existing fps=1 untouched) ───
cd /leonardo/home/userexternal/adiasse0/ai/ai_city_challenge

mkdir -p logs/track3_sft_fps2

echo "=========================================="
echo "STEP 1 — Build dataset at fps=${FPS}"
echo "  outputs: train_fps2.jsonl / val_fps2.jsonl"
echo "=========================================="

python track3/prepare_dataset.py \
    --work_dir  $WORK \
    --fps       $FPS \
    --val_ratio 0.1 \
    --seed      42

if [ $? -ne 0 ]; then
    echo "[ERROR] prepare_dataset.py failed — aborting"
    exit 1
fi

echo ""
echo "Dataset ready:"
ls -lh $WORK/data/track3/train_fps2.jsonl $WORK/data/track3/val_fps2.jsonl

# ── step 2: train on the new dataset ─────────────────────────────────────────
echo ""
echo "=========================================="
echo "STEP 2 — Train SFT BF16 on fps=${FPS} dataset"
echo "  run: $RUN"
echo "=========================================="

rm -rf unsloth_compiled_cache/

torchrun \
    --nproc_per_node=4 \
    --master_port=29503 \
    track3/train_sft_bf16.py \
    --work_dir    $WORK \
    --run_name    $RUN \
    --train_jsonl $WORK/data/track3/train_fps2.jsonl \
    --val_jsonl   $WORK/data/track3/val_fps2.jsonl \
    --epochs      3 \
    --lr          1e-4 \
    --batch_size  1 \
    --grad_accum  4
