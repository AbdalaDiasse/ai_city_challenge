# AI City Challenge 2026 — Track 3 Pipeline Steps

End-to-end guide for training, GRPO, inference, and submission.
All large files live under `$WORK = /leonardo_work/AIH4A_syrate`.

---

## Quick status (as of 2026-07-06)

| Phase | Status | Location |
|---|---|---|
| Phase 1 — Data | Done | `$WORK/data/track3/` |
| Phase 2 — SFT (4-bit) | Running — step 2200/7251, epoch 0.91 | `$WORK/checkpoints/track3_sft/sft_v1/` |
| Phase 2 — SFT (BF16) | In progress (OOM fixes being applied) | `$WORK/checkpoints/track3_sft/sft_bf16_v1/` |
| Phase 3 — GRPO | Not started — depends on Phase 2 completing | `$WORK/checkpoints/track3_grpo/grpo_v1/` |
| Phase 4 — Inference | Ready to run once test videos are downloaded | `$WORK/predictions/` |

---

## Phase 1 — Data setup (already done)

### What exists on disk

```
$WORK/data/track3/
├── annotations/
│   ├── train/          ← 10 task JSON files (44,040 items)
│   └── test/           ← test.json (960 items), evaluate.py, clip_manifest.csv
├── videos/             ← all 8 training video sources (~150 GB)
│   ├── Accident-Bench/
│   ├── TAD/
│   ├── so-tad/
│   └── ...
├── train_all.jsonl     ← 38,662 training conversations (Unsloth format)
└── val_all.jsonl       ← 4,286 validation conversations
```

Coverage: **42,948 / 44,040 (98%)** — 1,092 missing from UCF_Crimes and Vad-R1 Normal
(source files removed upstream, not recoverable).

### If you need to re-run data preparation

```bash
# 1. Download HF annotations (login node, needs internet)
bash track3/download_data.sh

# 2. Convert to Unsloth JSONL format
source /leonardo/home/userexternal/adiasse0/venvs/unsloth/bin/activate
unset PYTHONPATH
python track3/prepare_dataset.py \
    --annotation_dir $WORK/data/track3/annotations/train \
    --video_root     $WORK/data/track3/videos \
    --output_dir     $WORK/data/track3
```

---

## Phase 2 — SFT fine-tuning

Two scripts exist: 4-bit QLoRA (stable, currently running) and BF16 full-precision
(higher accuracy, OOM issue being fixed).

### 2a — 4-bit QLoRA (train_sft.py) — currently running as `sft_v1`

**Config**: Qwen3-VL-8B · 4-bit NF4 · LoRA r=16 · batch=1 · grad_accum=4 · lr=2e-4 · 3 epochs

```bash
mkdir -p logs/track3_sft
sbatch track3/train_sft.slurm
```

**Monitor**:
```bash
squeue -u $USER
tail -f logs/track3_sft/track3_sft_<JOBID>.out
```

**Checkpoints** saved every 200 steps to `$WORK/checkpoints/track3_sft/sft_v1/`.
Only the 3 most recent are kept (`save_total_limit=3`).

**Resume** if the job hits the 24h wall time:
```bash
# Edit SLURM script to add --resume, then resubmit
sbatch track3/train_sft.slurm   # trainer auto-detects latest checkpoint
```

**Check training progress**:
```bash
python3 -c "
import json, glob
base = '$WORK/checkpoints/track3_sft/sft_v1'
ckpts = sorted(glob.glob(f'{base}/checkpoint-*'), key=lambda p: int(p.split('-')[-1]))
s = json.load(open(f'{ckpts[-1]}/trainer_state.json'))
print(f'Latest checkpoint: {ckpts[-1]}')
print(f'Step {s[\"global_step\"]}/{s[\"max_steps\"]}  epoch {s[\"epoch\"]:.2f}')
print(f'Train loss: {s[\"log_history\"][-2][\"loss\"]:.4f}')
print(f'Best eval loss: {s[\"best_metric\"]:.4f}')
"
```

### 2b — BF16 full-precision (train_sft_bf16.py) — `sft_bf16_v1`

**Config**: Qwen3-VL-8B · BF16 · LoRA r=32 · batch=1 · grad_accum=4 · lr=1e-4 · 3 epochs

Produces a higher-accuracy adapter for the final submission. Uses the same data.

```bash
mkdir -p logs/track3_sft_bf16
sbatch track3/train_sft_bf16.slurm
```

Checkpoints land at `$WORK/checkpoints/track3_sft/sft_bf16_v1/`.

---

## Phase 3 — GRPO fine-tuning

Runs AFTER Phase 2. Starts from the best SFT checkpoint and applies
reinforcement learning with task-specific reward functions.

| Task type | Reward function |
|---|---|
| `bcq` | Exact match Yes/No → 0 or 1 |
| `mcq` | Exact match A/B/C/D → 0 or 1 |
| `bcq_openended` | Exact match on yes/no prefix → 0 or 1 |
| `mcq_openended` | Exact match on letter prefix → 0 or 1 |
| `temporal_localization` | Temporal IoU → 0–1 |
| All open-ended | ROUGE-L F1 → 0–1 |

### Setup

Edit `track3/train_grpo.slurm` to point `--sft_ckpt` at the best SFT checkpoint
(use the one with lowest `eval_loss` from `trainer_state.json`):

```bash
# Find best checkpoint
python3 -c "
import json, glob
base = '$WORK/checkpoints/track3_sft/sft_v1'
best_ckpt, best_loss = '', float('inf')
for ckpt in glob.glob(f'{base}/checkpoint-*'):
    s = json.load(open(f'{ckpt}/trainer_state.json'))
    if s.get('best_metric', float('inf')) < best_loss:
        best_loss = s['best_metric']
        best_ckpt = s.get('best_model_checkpoint', ckpt)
print(f'Best checkpoint: {best_ckpt}  eval_loss={best_loss:.4f}')
"
```

Then update `train_grpo.slurm`:
```bash
# Line to change:
--sft_ckpt  $WORK/checkpoints/track3_sft/sft_v1/checkpoint-XXXX
```

### Submit

```bash
mkdir -p logs/track3_grpo
sbatch track3/train_grpo.slurm
```

**Checkpoints** saved every 100 steps to `$WORK/checkpoints/track3_grpo/grpo_v1/`.

---

## Phase 4 — Inference & submission

### Step 4.1 — Download test videos (login node, needs internet)

The test set uses 80 YouTube clips. They must be downloaded before inference.

```bash
cd $WORK/data/track3/annotations/test
source /leonardo/home/userexternal/adiasse0/venvs/unsloth/bin/activate
unset PYTHONPATH
pip install yt-dlp  # if not already installed
python download_test_videos.py
# Videos land at: $WORK/data/track3/videos/tar_test/
```

Verify:
```bash
ls $WORK/data/track3/videos/tar_test/ | wc -l   # should be 80
```

### Step 4.2 — Run inference

Edit `track3/inference.slurm` to choose which checkpoint to use:

```bash
# For 4-bit SFT checkpoint (currently available):
CHECKPOINT=$WORK/checkpoints/track3_sft/sft_v1/checkpoint-2200
RUN_NAME=sft_v1_ckpt2200

# For GRPO checkpoint (after Phase 3):
CHECKPOINT=$WORK/checkpoints/track3_grpo/grpo_v1/checkpoint-XXXX
RUN_NAME=grpo_v1_ckptXXXX
```

Submit:
```bash
mkdir -p logs/track3_infer
sbatch track3/inference.slurm
```

The SLURM script runs inference **and** postprocess in sequence.

**Output**:
```
$WORK/predictions/<RUN_NAME>/
├── raw_predictions.jsonl   ← full model output including <think> blocks
├── submission.csv          ← competition-ready CSV (item_index, prediction)
└── inference.log           ← skipped/OOM items
```

### Step 4.3 — Run inference manually (without SLURM, for testing)

```bash
source /leonardo/home/userexternal/adiasse0/venvs/unsloth/bin/activate
unset PYTHONPATH
export WORK=/leonardo_work/AIH4A_syrate
export HF_HOME=$WORK/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python track3/inference.py \
    --checkpoint $WORK/checkpoints/track3_sft/sft_v1/checkpoint-2200 \
    --test_json  $WORK/data/track3/annotations/test/test.json \
    --video_root $WORK/data/track3/videos \
    --output_dir $WORK/predictions/sft_v1_ckpt2200 \
    --resume
```

### Step 4.4 — Post-process to submission CSV

```bash
python track3/postprocess.py \
    --input    $WORK/predictions/sft_v1_ckpt2200/raw_predictions.jsonl \
    --output   $WORK/predictions/sft_v1_ckpt2200/submission.csv \
    --test_json $WORK/data/track3/annotations/test/test.json
```

### Step 4.5 — Validate submission format

The official `evaluate.py` validates format (column names, all 960 rows present,
bcq starts with Yes/No, temporal has valid JSON, etc.):

```bash
python $WORK/data/track3/annotations/test/evaluate.py \
    --gt         $WORK/data/track3/annotations/test/test.json \
    --submission $WORK/predictions/sft_v1_ckpt2200/submission.csv
```

The released `test.json` has answers **redacted** — this only validates format,
not score. Actual scoring happens on the organizer's private server after upload.

### Step 4.6 — Upload to leaderboard

Upload `submission.csv` to: https://www.aicitychallenge.org (Track 3 submission page).

---

## Reference: checkpoint structure

See `docs/storage_guide.md` for a full explanation of what every file in a
checkpoint does and when you need it.

**TL;DR** — for inference you only need two files:
```
checkpoint-XXXX/
├── adapter_config.json        ← LoRA rank, target modules, base model name
└── adapter_model.safetensors  ← 196 MB trained LoRA weights
```

---

## Reference: script index

| Script | Purpose | How to run |
|---|---|---|
| `track3/download_data.sh` | Download HF annotations + video sources | `bash track3/download_data.sh` |
| `track3/download_model.sh` | Download base model weights to HF cache | `bash track3/download_model.sh` |
| `track3/prepare_dataset.py` | Convert annotations → train/val JSONL | `python track3/prepare_dataset.py` |
| `track3/postprocess_videos.py` | Fix partial video downloads | `python track3/postprocess_videos.py` |
| `track3/train_sft.py` | Phase 2 — 4-bit SFT training | `sbatch track3/train_sft.slurm` |
| `track3/train_sft_bf16.py` | Phase 2 — BF16 SFT training | `sbatch track3/train_sft_bf16.slurm` |
| `track3/train_grpo.py` | Phase 3 — GRPO from SFT checkpoint | `sbatch track3/train_grpo.slurm` |
| `track3/inference.py` | Phase 4 — generate predictions on test set | `sbatch track3/inference.slurm` |
| `track3/postprocess.py` | Phase 4 — format predictions as submission CSV | `python track3/postprocess.py` |

---

## Common commands

```bash
# Check running jobs
squeue -u $USER

# Cancel a job
scancel <JOBID>

# Watch live logs
tail -f logs/track3_sft/track3_sft_<JOBID>.out

# Check GPU memory while a job runs
# (run on the compute node via srun if needed)
nvidia-smi

# Check disk usage
du -sh $WORK/checkpoints/
du -sh $WORK/data/track3/videos/
du -sh $WORK/hf_cache/hub/

# Extend a running job's wall time (only works if QOS allows)
scontrol update JobId=<JOBID> TimeLimit=48:00:00
```
