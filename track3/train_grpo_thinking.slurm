#!/bin/bash
#SBATCH --job-name=track3_grpo_think
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=AIH4A_syrate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=logs/track3_grpo_thinking/%x_%j.out
#SBATCH --error=logs/track3_grpo_thinking/%x_%j.err

# ── environment ───────────────────────────────────────────────────────────────
module load cuda/12.6
module load profile/deeplrn
module load cineca-ai/4.3.0

unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"

source /leonardo/home/userexternal/adiasse0/venvs/unsloth/bin/activate
unset PYTHONPATH

export WORK=/leonardo_work/AIH4A_syrate
export HF_HOME=$WORK/hf_cache
export TORCHDYNAMO_DISABLE=1   # required for 4-bit (AlignDevicesHook conflict)
export PYTHONUNBUFFERED=1

export RUN=grpo_thinking_v1

# ── download 4-bit Thinking model if not cached ───────────────────────────────
# Must happen BEFORE setting HF_HUB_OFFLINE=1 — offline mode blocks downloads.
# Must use standard HF cache format (no --local-dir) so HF_HUB_OFFLINE finds it.
MODEL_CACHE_4BIT="$HF_HOME/hub/models--unsloth--Qwen3-VL-8B-Thinking-unsloth-bnb-4bit"
if [ ! -d "$MODEL_CACHE_4BIT/snapshots" ]; then
    echo "Downloading unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit ..."
    HF_HOME=$HF_HOME huggingface-cli download \
        unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit
    echo "Download complete."
else
    echo "4-bit Thinking model already in HF cache."
fi

# Lock to local cache now that the model is present
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ── config ────────────────────────────────────────────────────────────────────
# Point to the Thinking SFT checkpoint produced by train_sft_thinking.slurm.
# Change this to the latest checkpoint if training was interrupted.
SFT_CKPT=$WORK/checkpoints/track3_sft/sft_thinking_v1/lora_adapter

# ── launch ────────────────────────────────────────────────────────────────────
cd /leonardo/home/userexternal/adiasse0/ai/ai_city_challenge

mkdir -p logs/track3_grpo_thinking
rm -rf unsloth_compiled_cache/

torchrun \
    --nproc_per_node=4 \
    --master_port=29504 \
    track3/train_grpo_thinking.py \
    --work_dir           $WORK \
    --run_name           $RUN \
    --sft_ckpt           $SFT_CKPT \
    --epochs             1 \
    --lr                 5e-6 \
    --batch_size         1 \
    --num_generations    4 \
    --max_prompt_len     2048 \
    --max_completion_len 1024
