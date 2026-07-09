#!/bin/bash
#SBATCH --job-name=track3_grpo
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=AIH4A_syrate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=logs/track3_grpo/%x_%j.out
#SBATCH --error=logs/track3_grpo/%x_%j.err

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
export TORCHDYNAMO_DISABLE=1   # required for 4-bit model (AlignDevicesHook conflict)

export RUN=grpo_v1

# ── launch ───────────────────────────────────────────────────────────────────
cd /leonardo/home/userexternal/adiasse0/ai/ai_city_challenge

mkdir -p logs/track3_grpo
rm -rf unsloth_compiled_cache/

torchrun \
    --nproc_per_node=4 \
    --master_port=29501 \
    track3/train_grpo.py \
    --work_dir         $WORK \
    --run_name         $RUN \
    --sft_ckpt         $WORK/checkpoints/track3_sft/sft_v1/checkpoint-5600 \
    --epochs           1 \
    --lr               5e-6 \
    --batch_size       1 \
    --num_generations  4 \
    --max_prompt_len   2048 \
    --max_completion_len 512
