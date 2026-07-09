#!/bin/bash
#SBATCH --job-name=track3_sft_bf16
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=AIH4A_syrate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00
#SBATCH --output=logs/track3_sft_bf16/%x_%j.out
#SBATCH --error=logs/track3_sft_bf16/%x_%j.err

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
# no TORCHDYNAMO_DISABLE needed — BF16 loading has no AlignDevicesHook conflict

export RUN=sft_bf16_v1

# ── launch ───────────────────────────────────────────────────────────────────
cd /leonardo/home/userexternal/adiasse0/ai/ai_city_challenge

rm -rf unsloth_compiled_cache/   # clear stale compiled cache from previous runs

torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    track3/train_sft_bf16.py \
    --work_dir   $WORK \
    --run_name   $RUN \
    --epochs     3 \
    --lr         1e-4 \
    --batch_size 1 \
    --grad_accum 4
