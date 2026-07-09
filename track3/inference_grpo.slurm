#!/bin/bash
#SBATCH --job-name=track3_infer_grpo
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=AIH4A_syrate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=logs/track3_infer_grpo/%x_%j.out
#SBATCH --error=logs/track3_infer_grpo/%x_%j.err

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
# Change CKPT_STEP to whichever GRPO checkpoint you want to evaluate.
CKPT_STEP=3800
CHECKPOINT=$WORK/checkpoints/track3_grpo/grpo_v1/checkpoint-${CKPT_STEP}
RUN_NAME=grpo_v1_ckp${CKPT_STEP}
NUM_GPUS=4

# GRPO was trained on the 4-bit base model (unsloth/qwen3-vl-8b-instruct-unsloth-bnb-4bit)
# so we must load in 4-bit here to match the adapter's base model.
LOAD_FLAG="--load_in_4bit"

# ── launch 4 parallel inference workers (one per GPU) ────────────────────────
cd /leonardo/home/userexternal/adiasse0/ai/ai_city_challenge

mkdir -p logs/track3_infer_grpo "$WORK/predictions/$RUN_NAME"

echo "Checkpoint : $CHECKPOINT"
echo "Run name   : $RUN_NAME"
echo "Load flag  : $LOAD_FLAG"

for SHARD_ID in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$SHARD_ID python track3/inference.py \
        --checkpoint   "$CHECKPOINT" \
        --test_json    "$WORK/data/track3/annotations/test/test.json" \
        --video_root   "$WORK/data/track3/videos/tar_test" \
        --output_dir   "$WORK/predictions/$RUN_NAME" \
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
cat "$WORK/predictions/$RUN_NAME/raw_predictions_shard"*.jsonl \
    > "$WORK/predictions/$RUN_NAME/raw_predictions.jsonl"

echo "Merge done. Running postprocess..."

python track3/postprocess.py \
    --input    "$WORK/predictions/$RUN_NAME/raw_predictions.jsonl" \
    --output   "$WORK/predictions/$RUN_NAME/submission.csv" \
    --test_json "$WORK/data/track3/annotations/test/test.json"

echo "Submission CSV → $WORK/predictions/$RUN_NAME/submission.csv"
