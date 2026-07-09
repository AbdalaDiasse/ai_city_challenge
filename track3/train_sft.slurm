#!/bin/bash
#SBATCH --job-name=track3_sft
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=AIH4A_syrate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=logs/track3_sft/%x_%j.out
#SBATCH --error=logs/track3_sft/%x_%j.err

# ── environment ──────────────────────────────────────────────────────────────
module load cuda/12.6
module load profile/deeplrn
module load cineca-ai/4.3.0

# Prevent CUDA memory fragmentation
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"

# Activate venv — absolute path required on compute nodes
source /leonardo/home/userexternal/adiasse0/venvs/unsloth/bin/activate
unset PYTHONPATH

# Storage
export WORK=/leonardo_work/AIH4A_syrate
export HF_HOME=$WORK/hf_cache
export HF_HUB_OFFLINE=1        # compute nodes have no internet; use cached weights
export TRANSFORMERS_OFFLINE=1
export TORCHDYNAMO_DISABLE=1   # torch.compile conflicts with accelerate AlignDevicesHook on 4-bit models in torch 2.12

# Experiment name — change this for new runs
export RUN=sft_v1

# ── launch ───────────────────────────────────────────────────────────────────
cd /leonardo/home/userexternal/adiasse0/ai/ai_city_challenge

torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    track3/train_sft.py \
    --work_dir   $WORK \
    --run_name   $RUN \
    --epochs     3 \
    --lr         2e-4 \
    --batch_size 1 \
    --grad_accum 4 \
    --resume 
