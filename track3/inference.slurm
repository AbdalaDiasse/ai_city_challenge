#!/bin/bash
#SBATCH --job-name=track3_infer_thinking
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=AIH4A_syrate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=logs/track3_infer_thinking/%x_%j.out
#SBATCH --error=logs/track3_infer_thinking/%x_%j.err

# ── environment ──────────────────────────────────────────────────────────────
module load cuda/12.6
module load profile/deeplrn
module load cineca-ai/4.3.0

source /leonardo/home/userexternal/adiasse0/venvs/unsloth/bin/activate
unset PYTHONPATH

export WORK=/leonardo_work/AIH4A_syrate
export HF_HOME=$WORK/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

# ── config ───────────────────────────────────────────────────────────────────
CHECKPOINT=$WORK/checkpoints/track3_sft/sft_thinking_v1/lora_adapter
RUN_NAME=sft_thinking_v1_lora_adapter_v2
NUM_GPUS=4

# BF16 adapter — no quantization flag needed
LOAD_FLAG=""

# ── launch 4 parallel inference workers (one per GPU) ────────────────────────
cd /leonardo/home/userexternal/adiasse0/ai/ai_city_challenge

mkdir -p logs/track3_infer_thinking $WORK/predictions/$RUN_NAME

for SHARD_ID in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$SHARD_ID python track3/inference.py \
        --checkpoint   $CHECKPOINT \
        --test_json    $WORK/data/track3/annotations/test/test.json \
        --video_root   $WORK/data/track3/videos/tar_test \
        --output_dir   $WORK/predictions/$RUN_NAME \
        --fps          1.0 \
        --max_pixels   $((360 * 420)) \
        --num_shards   $NUM_GPUS \
        --shard_id     $SHARD_ID \
        --resume \
        $LOAD_FLAG &
done

wait
echo "All shards done. Merging..."

# ── merge shard outputs ───────────────────────────────────────────────────────
cat $WORK/predictions/$RUN_NAME/raw_predictions_shard*.jsonl \
    > $WORK/predictions/$RUN_NAME/raw_predictions.jsonl

echo "Merge done. Running postprocess..."

python track3/postprocess.py \
    --input    $WORK/predictions/$RUN_NAME/raw_predictions.jsonl \
    --output   $WORK/predictions/$RUN_NAME/submission.csv \
    --test_json $WORK/data/track3/annotations/test/test.json

echo "Submission CSV → $WORK/predictions/$RUN_NAME/submission.csv"
