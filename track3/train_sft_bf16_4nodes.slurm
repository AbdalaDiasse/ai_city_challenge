#!/bin/bash
#SBATCH --job-name=track3_sft_bf16_4n
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=AIH4A_syrate
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/track3_sft_bf16_4nodes/%x_%j.out
#SBATCH --error=logs/track3_sft_bf16_4nodes/%x_%j.err

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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# no TORCHDYNAMO_DISABLE needed — BF16 has no AlignDevicesHook conflict
export PYTHONUNBUFFERED=1

export RUN=sft_bf16_v1_4n

# ── multi-node setup ──────────────────────────────────────────────────────────
# scontrol runs on the head node before srun; MASTER_ADDR is visible to all.
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29501

echo "Nodes     : $SLURM_NNODES  ($SLURM_JOB_NODELIST)"
echo "Master    : $MASTER_ADDR:$MASTER_PORT"
echo "Run       : $RUN"
echo "GPUs/node : $(nvidia-smi --list-gpus | wc -l)"
echo "Total GPUs: $((SLURM_NNODES * 4))"

# ── launch ────────────────────────────────────────────────────────────────────
cd /leonardo/home/userexternal/adiasse0/ai/ai_city_challenge

mkdir -p logs/track3_sft_bf16_4nodes
rm -rf unsloth_compiled_cache/

# Effective batch = nodes × gpus/node × batch_size × grad_accum
#                 = 4     × 4         × 1           × 1           = 16
# This matches the single-node run (1 × 4 × 1 × 4 = 16), keeping
# the same gradient update scale while running ~4× faster.
#
# srun dispatches torchrun to ALL 4 nodes (one task per node).
# c10d rendezvous auto-assigns node ranks via $SLURM_PROCID — no
# explicit --node_rank needed.
srun torchrun \
    --nproc_per_node=4 \
    --nnodes=$SLURM_NNODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    track3/train_sft_bf16.py \
    --work_dir   $WORK \
    --run_name   $RUN \
    --epochs     3 \
    --lr         1e-4 \
    --batch_size 1 \
    --grad_accum 1
