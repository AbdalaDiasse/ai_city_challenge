#!/bin/bash
#SBATCH --job-name=track3_grpo_4n
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=AIH4A_syrate
#SBATCH --nodes=4
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
export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1

export RUN=grpo_v1_4n

# ── multi-node setup ─────────────────────────────────────────────────────────
# scontrol runs on the head node before srun; MASTER_ADDR is visible to all
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29502

echo "Nodes     : $SLURM_NNODES  ($SLURM_JOB_NODELIST)"
echo "Master    : $MASTER_ADDR:$MASTER_PORT"
echo "GPUs/node : $(nvidia-smi --list-gpus | wc -l)"

# ── launch ───────────────────────────────────────────────────────────────────
cd /leonardo/home/userexternal/adiasse0/ai/ai_city_challenge

mkdir -p logs/track3_grpo
rm -rf unsloth_compiled_cache/

# srun dispatches torchrun to ALL allocated nodes (one task per node).
# c10d rendezvous removes the need for explicit --node_rank: each task's
# $SLURM_PROCID (0–3) is passed automatically by the rdzv backend.
srun torchrun \
    --nproc_per_node=4 \
    --nnodes=$SLURM_NNODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    track3/train_grpo.py \
    --work_dir          $WORK \
    --run_name          $RUN \
    --sft_ckpt          $WORK/checkpoints/track3_sft/sft_v1/checkpoint-5600 \
    --epochs            1 \
    --lr                5e-6 \
    --batch_size        1 \
    --num_generations   4 \
    --max_prompt_len    2048 \
    --max_completion_len 512
