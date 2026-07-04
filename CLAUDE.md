# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Training pipeline for the [AI City Challenge](https://www.aicitychallenge.org), targeting:
- **Track 3**: Anomalous Events in Transportation (video VQA with chain-of-thought reasoning)
- **Track 6**: TBD (added in stage 2)

Framework: [Unsloth](https://unsloth.ai) for efficient VLM fine-tuning. HPC: Leonardo (CINECA) via SLURM.

**Always read `HANDOFF.md` at the start of a session** — it tracks decisions that changed, failed approaches, and the exact next step.

---

## Track 3 — Anomalous Events in Transportation

**Official page**: https://www.aicitychallenge.org/2026-track3/
**Dataset (HF)**: `nvidia/PhysicalAI-Traffic-Anomaly-Reasoning`
**Model**: `unsloth/Qwen3-VL-8B-Instruct` (or Thinking variant for CoT)

### Task

Given a video clip (CCTV surveillance, fisheye traffic cam, or dashcam), answer questions requiring detection, reasoning, and explanation of anomalous events. All answers are grounded in explicit chain-of-thought reasoning.

There are **10 task types** across 3 groups:

| Group | Tasks |
|---|---|
| Basic | Event verification (Yes/No), MCQ (A/B/C/D), Open-ended QA |
| Scene | Scene description, Video summarization |
| Temporal | Temporal localization, Causal linkage, Event description |

### Dataset (TAR — Traffic Anomaly Reasoning)

- **Train**: 44,040 annotations across 3,670 CCTV videos (~26.1 hours total)
- **Test (TAR)**: 960 human-verified annotations, 80 clips (YouTube)
- **Test (FETV)**: 200 fisheye traffic-violation clips
- **Test (PSI VQA)**: 40 egocentric dashcam clips
- **Video size**: ~150 GB total, downloaded separately via the HF dataset script
- **Video sources**: VAD-R1, TAD, Accident-Bench, SO-TAD, TADBenchmark, UCF Crime, and others

**Data format** (`tao-vl-reason-v1.0`): JSON files, one per task type. Each item has:
```json
{
  "video_id": "TAD/01_Accident_001.mp4",
  "question": "<task-specific prompt>",
  "answer": "<expected output>",
  "reasoning": "<chain-of-thought explanation>"
}
```

Training data is split into 10 JSON files (one per task). Test data is a single combined JSON.

### Evaluation Metrics

**TAR (primary leaderboard)**:
- Binary / MCQ: exact accuracy
- Open-ended (8 types): BERTScore F1
- Final score: unweighted mean across 9 scored task types

**FETV**: `0.25·CIDEr + 0.25·BERTScore + 0.5·MacroF1`

**PSI VQA**: `0.25·T1 + 0.25·T2 + 0.25·T3 + 0.25·T4`

### Submission Format

CSV with two columns: `item_index`, `prediction`.
- Binary: `"Yes"` or `"No"`
- MCQ: single letter `"A"`, `"B"`, `"C"`, or `"D"`
- Open-ended: free-form text
- Temporal: `{"start": "MM:SS", "end": "MM:SS"}` (7-second tolerance)

### Training Code Pattern (Unsloth + FastVisionModel)

```python
from unsloth import FastVisionModel
import torch

model, tokenizer = FastVisionModel.from_pretrained(
    model_name="unsloth/Qwen3-VL-8B-Instruct",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,    # train vision encoder too
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
)
```

Use `SFTTrainer` from TRL with `SFTConfig`. Set `dataset_num_proc=1` to avoid a known crash.

### Track 3 — Execution Plan

**Phase 1 – Data setup**
1. Download dataset: `huggingface-cli download nvidia/PhysicalAI-Traffic-Anomaly-Reasoning`
2. Run video download script (~150 GB). Store videos in `data/track3/videos/`
3. Parse the 10 task-type JSON files; convert to Unsloth's conversation format with `<think>` blocks for CoT

**Phase 2 – Baseline SFT**
1. Fine-tune `Qwen3-VL-8B-Instruct` on all 10 task types jointly using `FastVisionModel` + LoRA
2. Train on Leonardo with multi-GPU SLURM job (see SLURM section below)
3. Validate on held-out examples per task type; track BERTScore F1

**Phase 3 – RL / GRPO (optional but high-impact)**
1. Use GRPO with task-specific reward functions:
   - Binary/MCQ: exact-match reward
   - Open-ended: BERTScore F1 as reward
2. Unsloth supports Vision GRPO for Qwen3-VL-8B natively (`FastVisionModel` + GRPO)

**Phase 4 – Inference & submission**
1. Run inference across all 3 test sets (TAR, FETV, PSI VQA)
2. Post-process outputs to match required formats per task type
3. Generate CSV: `item_index,prediction`

---

## Environment: Leonardo HPC (CINECA)

All GPU jobs must be submitted via SLURM.

**Standard SLURM header for GPU jobs:**
```bash
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=AIH4A_syrate
#SBATCH --gres=gpu:4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00
```

**Module setup for deep learning:**
```bash
module load cuda/12.6
module load profile/deeplrn
module load cineca-ai/4.3.0
```

**Unsloth venv**: `/leonardo/home/userexternal/adiasse0/venvs/unsloth` — Python 3.11, built from `cineca-ai/4.3.0`.

**Confirmed working package versions** (as of 2026-07-04):
```bash
torch==2.12.1+cu126   # cu126 is the correct build — cu121 caps at 2.5.1, too old for unsloth_zoo
bitsandbytes          # required by Unsloth kernels even for non-4bit runs
```

To rebuild the venv from scratch:
```bash
module load profile/deeplrn && module load cineca-ai/4.3.0
python3 -m venv ~/venvs/unsloth
source ~/venvs/unsloth/bin/activate
unset PYTHONPATH
pip install torch==2.12.1+cu126 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126
pip install "unsloth @ git+https://github.com/unslothai/unsloth.git" \
    unsloth_zoo bitsandbytes trl peft accelerate \
    datasets huggingface_hub bert-score numpy
```

**After activating venv**, always unset PYTHONPATH to prevent cineca-ai module packages from shadowing venv installs:
```bash
unset PYTHONPATH
```

**Required env var before training** (prevents CUDA memory fragmentation):
```bash
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
```

Submit: `sbatch <script.slurm>` | Monitor: `squeue -u $USER` | Cancel: `scancel <job_id>`

**Storage**: All large files (datasets, videos, checkpoints, model weights) must be stored under `/leonardo_work/AIH4A_syrate/`. The home directory has very limited quota. Use `export WORK=/leonardo_work/AIH4A_syrate` in all scripts.

HuggingFace model cache: `~/.cache/huggingface/hub/` — override with `export HF_HOME=$WORK/hf_cache` to avoid home quota issues.

---

## Related Sibling Projects

- `../mits/` — Working LoRA fine-tuning of Qwen2.5-VL-7B on traffic data using ms-swift. Reference `train_cinera.slurm` for a proven multi-GPU SLURM template on Leonardo.
- `../nanoVLM/` — Pure PyTorch VLM training with `torchrun`. Reference `slurm/train_job_distributed_cineca.slurm`.
- `../unsloth/` — Local Unsloth source checkout.
