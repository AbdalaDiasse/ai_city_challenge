# AI City Challenge — Track 3: Step-by-Step Runbook

Complete sequence of commands to go from a fresh clone to a submitted result.
Run every step in order. Each step says **where** (login node vs GPU node vs SLURM).

---

## Prerequisites

- Leonardo HPC account with access to `AIH4A_syrate` project
- Repository cloned to `/leonardo/home/userexternal/adiasse0/ai/ai_city_challenge`
- Unsloth venv exists at `/leonardo/home/userexternal/adiasse0/venvs/unsloth`

If the venv does not exist or is broken, see **Appendix A** to rebuild it from scratch.

---

## Step 1 — Verify the environment (login node)

Run this every time you start a new session to confirm everything is wired correctly.

```bash
# Load modules
module load cuda/12.6
module load profile/deeplrn
module load cineca-ai/4.3.0

# Activate venv
source /leonardo/home/userexternal/adiasse0/venvs/unsloth/bin/activate
unset PYTHONPATH

# Quick sanity check (no GPU needed for imports)
python -c "import torch; import unsloth; print('torch:', torch.__version__)"
# Expected: torch: 2.12.1+cu126
```

---

## Step 2 — Download annotations (login node)

Annotations are ~500 MB and download fast. Videos come later.

```bash
export WORK=/leonardo_work/AIH4A_syrate
export HF_HOME=$WORK/hf_cache

# Activate venv first so huggingface-cli is in PATH
module load cuda/12.6
module load profile/deeplrn
module load cineca-ai/4.3.0
source /leonardo/home/userexternal/adiasse0/venvs/unsloth/bin/activate
unset PYTHONPATH

huggingface-cli download nvidia/PhysicalAI-Traffic-Anomaly-Reasoning \
    --repo-type dataset \
    --local-dir $WORK/data/track3/annotations

# Confirm the 10 task files are present
ls $WORK/data/track3/annotations/train/*.json | wc -l
# Expected: 10
```

---

## Step 3 — Download videos (~150 GB, login node, inside screen)

This runs for several hours. Use `screen` so it survives a disconnect.

```bash
# Start a persistent screen session
screen -S video_dl

export WORK=/leonardo_work/AIH4A_syrate

module load cuda/12.6
module load profile/deeplrn
module load cineca-ai/4.3.0
source /leonardo/home/userexternal/adiasse0/venvs/unsloth/bin/activate
unset PYTHONPATH

python $WORK/data/track3/annotations/download_videos.py \
    --out $WORK/data/track3/videos \
    --install-deps
```

**Detach** from screen: `Ctrl+A` then `D`
**Reattach** later: `screen -r video_dl`
**Check progress**: `du -sh $WORK/data/track3/videos`

When the download completes, verify:
```bash
ls $WORK/data/track3/videos/
# Should show subdirectories: TAD/, UCF_Crime/, etc.
du -sh $WORK/data/track3/videos/
# Should be ~150 GB
```

---

## Step 4 — Convert annotations to training format (login node)

Once videos are downloaded, convert all 10 task JSONs to Unsloth's JSONL format.

```bash
cd /leonardo/home/userexternal/adiasse0/ai/ai_city_challenge

export WORK=/leonardo_work/AIH4A_syrate

module load cuda/12.6
module load profile/deeplrn
module load cineca-ai/4.3.0
source /leonardo/home/userexternal/adiasse0/venvs/unsloth/bin/activate
unset PYTHONPATH

python track3/prepare_dataset.py --val_ratio 0.1 --seed 42
```

Expected output:
```
Total conversations loaded: ~44040
Split → train: ~39636, val: ~4404
Saved 39636 records → /leonardo_work/AIH4A_syrate/data/track3/train_all.jsonl
Saved 4404 records → /leonardo_work/AIH4A_syrate/data/track3/val_all.jsonl
```

Check the stats file for per-task counts:
```bash
cat $WORK/data/track3/dataset_stats.json
```

---

## Step 5 — Write the SFT training script

This file does not exist yet. Create `track3/train_sft.py`:

```bash
# From the repo root, ask Claude Code to write it, or create it manually.
# It must use FastVisionModel (not FastLanguageModel) and SFTTrainer with dataset_num_proc=1.
# See track3/PLAN.md Phase 2 for the template.
```

Key requirements for `train_sft.py`:
- `FastVisionModel.from_pretrained("unsloth/Qwen3-VL-8B-Instruct", load_in_4bit=True)`
- `FastVisionModel.get_peft_model(...)` with `r=16, lora_alpha=16`
- `SFTConfig(dataset_num_proc=1, ...)` — this is mandatory, not optional
- Reads `$WORK/data/track3/train_all.jsonl`
- Saves checkpoints to `$WORK/checkpoints/track3_sft/`

---

## Step 6 — Write the SFT SLURM script

Create `track3/train_sft.slurm`. The activation order is critical — wrong order causes CUDA failures.

```bash
#!/bin/bash
#SBATCH --job-name=track3_sft
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=AIH4A_syrate
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00
#SBATCH --output=logs/track3/%A_sft.out
#SBATCH --error=logs/track3/%A_sft.err

# ORDER MATTERS: modules → unset alloc conf → venv → unset PYTHONPATH → exports
module load cuda/12.6
module load profile/deeplrn
module load cineca-ai/4.3.0

unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"

source /leonardo/home/userexternal/adiasse0/venvs/unsloth/bin/activate
unset PYTHONPATH

export WORK=/leonardo_work/AIH4A_syrate
export HF_HOME=$WORK/hf_cache

mkdir -p logs/track3

torchrun --nproc_per_node=4 \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$(hostname -s) \
    --master_port=29500 \
    track3/train_sft.py
```

---

## Step 7 — Run SFT training (SLURM)

```bash
cd /leonardo/home/userexternal/adiasse0/ai/ai_city_challenge
mkdir -p logs/track3

sbatch track3/train_sft.slurm
```

Monitor:
```bash
squeue -u $USER                              # check job status
tail -f logs/track3/<job_id>_sft.out         # stream training logs
```

Cancel if needed:
```bash
scancel <job_id>
```

Training is complete when you see the final checkpoint saved to `$WORK/checkpoints/track3_sft/`.

---

## Step 8 — Evaluate SFT checkpoint (GPU node or SLURM)

Get a quick GPU node for interactive evaluation:
```bash
srun --partition=boost_usr_prod \
     --account=AIH4A_syrate \
     --gres=gpu:1 \
     --time=01:00:00 \
     --pty bash
```

Then inside the GPU node:
```bash
cd /leonardo/home/userexternal/adiasse0/ai/ai_city_challenge

module load cuda/12.6
module load profile/deeplrn
module load cineca-ai/4.3.0
source /leonardo/home/userexternal/adiasse0/venvs/unsloth/bin/activate
unset PYTHONPATH

export WORK=/leonardo_work/AIH4A_syrate

python track3/evaluate.py \
    --checkpoint $WORK/checkpoints/track3_sft/checkpoint-<N> \
    --data $WORK/data/track3/val_all.jsonl \
    --output $WORK/results/val_sft.json
```

Look for per-task BERTScore F1 in the output. If open-ended tasks score below 0.70 F1, consider running GRPO (Step 9). If binary/MCQ accuracy is below 0.85, increase training epochs.

---

## Step 9 — GRPO fine-tuning (optional but high-impact, SLURM)

Only run this after a good SFT checkpoint exists. GRPO directly optimizes the evaluation metrics.

Write `track3/train_grpo.py`:
- Initialize from the SFT checkpoint: `FastVisionModel.from_pretrained("$WORK/checkpoints/track3_sft/checkpoint-<best>")`
- Use `GRPOTrainer` from TRL
- Reward function routes per task type:
  - Binary / MCQ → exact-match reward (0.0 or 1.0)
  - All open-ended types → BERTScore F1 as continuous reward

Write `track3/train_grpo.slurm`:
- Same header as SFT but `--time=48:00:00`
- Same module + venv activation order

Submit:
```bash
sbatch track3/train_grpo.slurm
```

---

## Step 10 — Run inference on test sets (SLURM)

Three test sets, three separate inference runs. Write `track3/inference.py` then:

```bash
# TAR test set
python track3/inference.py \
    --checkpoint $WORK/checkpoints/track3_grpo/final \
    --test_json  $WORK/data/track3/annotations/test_tar.json \
    --video_dir  $WORK/data/track3/videos \
    --output     $WORK/submissions/track3_tar_raw.json

# FETV test set
python track3/inference.py \
    --checkpoint $WORK/checkpoints/track3_grpo/final \
    --test_json  $WORK/data/track3/annotations/test_fetv.json \
    --video_dir  $WORK/data/track3/videos \
    --output     $WORK/submissions/track3_fetv_raw.json

# PSI VQA test set
python track3/inference.py \
    --checkpoint $WORK/checkpoints/track3_grpo/final \
    --test_json  $WORK/data/track3/annotations/test_psi_vqa.json \
    --video_dir  $WORK/data/track3/videos \
    --output     $WORK/submissions/track3_psi_vqa_raw.json
```

---

## Step 11 — Post-process and generate submission CSVs

Write `track3/postprocess.py` then:

```bash
python track3/postprocess.py \
    --raw  $WORK/submissions/track3_tar_raw.json \
    --out  $WORK/submissions/track3_tar.csv

python track3/postprocess.py \
    --raw  $WORK/submissions/track3_fetv_raw.json \
    --out  $WORK/submissions/track3_fetv.csv

python track3/postprocess.py \
    --raw  $WORK/submissions/track3_psi_vqa_raw.json \
    --out  $WORK/submissions/track3_psi_vqa.csv
```

Post-processing rules per task type:
| Task | Rule |
|---|---|
| Binary (Yes/No) | Regex: first occurrence of `Yes` or `No` in output |
| MCQ (A/B/C/D) | Regex: first standalone letter A–D |
| Open-ended | Strip `<think>…</think>` block; return remaining text |
| Temporal | Parse JSON `{"start": "MM:SS", "end": "MM:SS"}` from output |

Verify the CSV format before uploading:
```bash
head -5 $WORK/submissions/track3_tar.csv
# Expected:
# item_index,prediction
# 0,Yes
# 1,B
# 2,"The vehicle failed to yield..."
```

---

## Step 12 — Submit to the leaderboard

Upload the three CSVs at: https://www.aicitychallenge.org/2026-track3/

- `track3_tar.csv` → TAR leaderboard (primary)
- `track3_fetv.csv` → FETV leaderboard
- `track3_psi_vqa.csv` → PSI VQA leaderboard

> TAR submission is required before FETV and PSI VQA count for prize eligibility.

---

## Current Status

| Step | Status |
|---|---|
| 1 — Verify environment | Done (2026-07-04) — Unsloth OK on lrdn2492 |
| 2 — Download annotations | Done — 10 task JSONs in `$WORK/data/track3/annotations/` |
| 3 — Download videos | **Pending** — run in screen session (~150 GB) |
| 4 — Convert to JSONL | Pending (needs videos first) |
| 5 — Write `train_sft.py` | Pending |
| 6 — Write `train_sft.slurm` | Pending |
| 7 — Run SFT | Pending |
| 8 — Evaluate SFT | Pending |
| 9 — Run GRPO | Pending |
| 10 — Run inference | Pending |
| 11 — Post-process | Pending |
| 12 — Submit | Pending |

---

## Appendix A — Rebuild the Unsloth venv from scratch

Use this if the venv is broken or missing.

```bash
# On a login node:
module load profile/deeplrn
module load cineca-ai/4.3.0

# Remove old venv if it exists
rm -rf ~/venvs/unsloth

# Create fresh venv using Python 3.11 from the cineca-ai module
python3 -m venv ~/venvs/unsloth
source ~/venvs/unsloth/bin/activate
unset PYTHONPATH

# Install torch FIRST with the correct CUDA 12.6 build
pip install torch==2.12.1+cu126 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

# Then install Unsloth and all dependencies
pip install "unsloth @ git+https://github.com/unslothai/unsloth.git" \
    unsloth_zoo \
    bitsandbytes \
    trl \
    peft \
    accelerate \
    datasets \
    huggingface_hub \
    bert-score \
    numpy

# Verify
python -c "from unsloth import FastVisionModel; print('Unsloth OK')"
# Expected: 🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning. Unsloth OK
```

**Why this order matters**:
- `cineca-ai/4.3.0` provides Python 3.11.6 — required for all modern ML packages
- `torch` must be installed before `unsloth_zoo` so that `unsloth_zoo` resolves against the correct torch version
- `bitsandbytes` is required by Unsloth kernels even when not using 4-bit quantization
- `unset PYTHONPATH` prevents the cineca-ai conda environment from shadowing venv packages

---

## Appendix B — Interactive GPU session

For debugging or evaluation without submitting a full job:

```bash
srun --partition=boost_usr_prod \
     --account=AIH4A_syrate \
     --gres=gpu:1 \
     --cpus-per-task=10 \
     --time=01:00:00 \
     --pty bash
```

Inside the interactive session:
```bash
module load cuda/12.6
module load profile/deeplrn
module load cineca-ai/4.3.0

unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"

source /leonardo/home/userexternal/adiasse0/venvs/unsloth/bin/activate
unset PYTHONPATH

export WORK=/leonardo_work/AIH4A_syrate
export HF_HOME=$WORK/hf_cache
```

---

## Appendix C — Storage map

| Location | What lives there |
|---|---|
| `/leonardo_work/AIH4A_syrate/data/track3/annotations/` | HF dataset annotation JSONs |
| `/leonardo_work/AIH4A_syrate/data/track3/videos/` | ~150 GB raw video files |
| `/leonardo_work/AIH4A_syrate/data/track3/train_all.jsonl` | Converted training data |
| `/leonardo_work/AIH4A_syrate/data/track3/val_all.jsonl` | Converted validation data |
| `/leonardo_work/AIH4A_syrate/checkpoints/track3_sft/` | SFT checkpoints |
| `/leonardo_work/AIH4A_syrate/checkpoints/track3_grpo/` | GRPO checkpoints |
| `/leonardo_work/AIH4A_syrate/submissions/` | Final submission CSVs |
| `/leonardo_work/AIH4A_syrate/hf_cache/` | HuggingFace model cache |
| `/leonardo/home/userexternal/adiasse0/venvs/unsloth/` | Python venv |
| `/leonardo/home/userexternal/adiasse0/ai/ai_city_challenge/` | This repo |

> Never store large files under `$HOME` — the home directory has a tight quota on Leonardo.
