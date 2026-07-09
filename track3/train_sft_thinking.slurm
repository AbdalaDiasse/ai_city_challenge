#!/bin/bash
#SBATCH --job-name=track3_sft_think
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=AIH4A_syrate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00
#SBATCH --output=logs/track3_sft_thinking/%x_%j.out
#SBATCH --error=logs/track3_sft_thinking/%x_%j.err

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
export PYTHONUNBUFFERED=1

export RUN=sft_thinking_v1

# ── download model if not cached ──────────────────────────────────────────────
# FastVisionModel.from_pretrained uses the standard HF cache format
# (blobs/snapshots/refs/trees).  Do NOT use --local-dir — it saves files in a
# flat layout that HF_HUB_OFFLINE=1 cannot find.
MODEL_CACHE="$HF_HOME/hub/models--unsloth--Qwen3-VL-8B-Thinking"
if [ ! -d "$MODEL_CACHE/snapshots" ]; then
    echo "Downloading unsloth/Qwen3-VL-8B-Thinking into HF cache ..."
    HF_HOME=$HF_HOME huggingface-cli download unsloth/Qwen3-VL-8B-Thinking
    echo "Download complete."
else
    echo "Model already in HF cache: $MODEL_CACHE"
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ── launch ────────────────────────────────────────────────────────────────────
cd /leonardo/home/userexternal/adiasse0/ai/ai_city_challenge

mkdir -p logs/track3_sft_thinking
rm -rf unsloth_compiled_cache/

torchrun \
    --nproc_per_node=4 \
    --master_port=29503 \
    track3/train_sft_thinking.py \
    --work_dir   $WORK \
    --run_name   $RUN \
    --epochs     3 \
    --lr         5e-5 \
    --batch_size 1 \
    --grad_accum 4
