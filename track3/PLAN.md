# Track 3 — Anomalous Events in Transportation: Execution Plan

**Challenge**: AI City Challenge 2026
**Official page**: https://www.aicitychallenge.org/2026-track3/
**Dataset**: `nvidia/PhysicalAI-Traffic-Anomaly-Reasoning`
**Model**: `unsloth/Qwen3-VL-8B-Instruct` (or Thinking variant)
**Framework**: Unsloth (FastVisionModel) + TRL (SFTTrainer / GRPOTrainer)
**HPC**: Leonardo / CINECA via SLURM

---

## Task Overview

Given a video clip from CCTV surveillance, fisheye traffic cameras, or egocentric dashcams, the model must detect, reason about, and explain anomalous traffic events using explicit chain-of-thought reasoning.

### 10 Task Types

| # | Task | Output Format |
|---|---|---|
| 1 | Event Verification | `"Yes"` or `"No"` |
| 2 | Multiple-Choice QA | Single letter: `"A"`, `"B"`, `"C"`, or `"D"` |
| 3 | Open-ended QA | Free-form text |
| 4 | Scene Description | Free-form text |
| 5 | Video Summarization | Free-form text |
| 6 | Temporal Localization | `{"start": "MM:SS", "end": "MM:SS"}` |
| 7 | Causal Linkage | Free-form text |
| 8 | Event Description | Free-form text |
| 9 | (additional open QA variants) | Free-form text |
| 10 | (additional open QA variants) | Free-form text |

### Three Independent Leaderboards

| Leaderboard | Test Set | Scoring |
|---|---|---|
| TAR | 960 annotations, 80 YouTube clips | Mean BERTScore F1 (open-ended) + accuracy (binary/MCQ) |
| FETV | 200 fisheye traffic-violation clips | `0.25·CIDEr + 0.25·BERTScore + 0.5·MacroF1` |
| PSI VQA | 40 egocentric dashcam clips | `0.25·T1 + 0.25·T2 + 0.25·T3 + 0.25·T4` |

Prize eligibility for FETV and PSI VQA requires a valid TAR submission.

---

## Dataset Details

- **Train**: 44,040 annotations across 3,670 CCTV videos (~26.1 hours total)
  - ~9.2 hours anomalous events, ~16.9 hours normal traffic
  - Annotations generated via VLM pipeline (Gemini 2.1 Pro, Gemma-4)
- **Videos**: ~150 GB, not redistributed — downloaded via HF dataset script
- **Video sources**: VAD-R1, TAD, Accident-Bench, SO-TAD, TADBenchmark, UCF Crime, Highway Traffic Videos, Barbados Challenge
- **Format**: `tao-vl-reason-v1.0` — 10 JSON files (one per task type) for train; single JSON for test

Each annotation:
```json
{
  "video_id": "TAD/01_Accident_001.mp4",
  "question": "<task-specific prompt>",
  "answer": "<expected output>",
  "reasoning": "<chain-of-thought explanation>"
}
```

---

## Phase 1 — Data Setup

### 1.1 Download Dataset & Videos

All large files (videos, annotations, checkpoints) must be stored under `/leonardo_work/AIH4A_syrate/` — this is the shared project scratch space on Leonardo, not the home directory.

```bash
export WORK=/leonardo_work/AIH4A_syrate

# Install HF CLI if needed
pip install huggingface_hub hf_transfer

# Download annotations
huggingface-cli download nvidia/PhysicalAI-Traffic-Anomaly-Reasoning \
    --repo-type dataset \
    --local-dir $WORK/data/track3/annotations

# Run the video download script (provided in the dataset repo)
# NOTE: vendor script uses --out, not --output_dir
python $WORK/data/track3/annotations/download_videos.py \
    --out $WORK/data/track3/videos \
    --install-deps
```

Expected structure after download:
```
/leonardo_work/AIH4A_syrate/
└── data/track3/
    ├── annotations/
    │   ├── train_event_verification.json
    │   ├── train_mcq.json
    │   ├── train_open_qa.json
    │   ├── ... (10 files total)
    │   └── test_combined.json
    └── videos/
        ├── TAD/
        ├── UCF_Crime/
        └── ...
```

### 1.2 Convert to Unsloth Conversation Format

Convert each annotation into Qwen3-VL's native conversation format with chain-of-thought in `<think>` blocks:

```python
def build_conversation(item, video_path):
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": item["question"]}
                ]
            },
            {
                "role": "assistant",
                "content": f"<think>{item['reasoning']}</think>\n{item['answer']}"
            }
        ]
    }
```

Save merged dataset as `/leonardo_work/AIH4A_syrate/data/track3/train_all.jsonl` (one item per line).

### 1.3 Validation Split

Hold out ~10% per task type for local BERTScore evaluation before submission.

---

## Phase 2 — Baseline SFT (Supervised Fine-Tuning)

### 2.1 Training Script (`track3/train_sft.py`)

```python
from unsloth import FastVisionModel
from trl import SFTTrainer, SFTConfig

model, tokenizer = FastVisionModel.from_pretrained(
    model_name="unsloth/Qwen3-VL-8B-Instruct",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_steps=500,
        output_dir="checkpoints/track3_sft",
        dataset_num_proc=1,   # must be 1 — avoids a known TRL crash
    ),
)
trainer.train()
```

### 2.2 SLURM Job (`track3/train_sft.slurm`)

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

# 1. Load modules first — must happen before venv activation so CUDA
#    libraries are on PATH before Python picks up its environment.
module load cuda/12.6
module load profile/deeplrn
module load cineca-ai/4.3.0

# 2. CUDA allocator fix — must be unset before training
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"

# 3. Activate Unsloth venv — absolute path, not ~/ (tilde is not
#    guaranteed to expand on all compute nodes under SLURM)
source /leonardo/home/userexternal/adiasse0/venvs/unsloth/bin/activate
unset PYTHONPATH   # prevent cineca-ai module from shadowing venv packages

# Confirmed working: torch==2.12.1+cu126, bitsandbytes, unsloth (git main)
# See CLAUDE.md for full venv rebuild instructions if needed.

export WORK=/leonardo_work/AIH4A_syrate

torchrun --nproc_per_node=4 track3/train_sft.py
```

### 2.3 Validation

After each checkpoint, evaluate on the held-out split:

```bash
python track3/evaluate.py \
    --checkpoint checkpoints/track3_sft/checkpoint-<N> \
    --data data/track3/val_all.jsonl \
    --output results/val_sft.json
```

Metrics reported per task type: accuracy (binary/MCQ), BERTScore F1 (open-ended).

---

## Phase 3 — GRPO / Reinforcement Learning (High-Impact)

This phase is the main differentiator vs. baseline SFT. Unsloth natively supports Vision GRPO for Qwen3-VL-8B.

### 3.1 Reward Functions

```python
from bert_score import score as bert_score

def reward_binary_mcq(predictions, references):
    # Exact match: 1.0 if correct, 0.0 otherwise
    return [1.0 if p.strip() == r.strip() else 0.0
            for p, r in zip(predictions, references)]

def reward_open_ended(predictions, references):
    # BERTScore F1 as continuous reward signal
    _, _, F1 = bert_score(predictions, references, lang="en")
    return F1.tolist()
```

### 3.2 Task-Type Router

Route each sample to the correct reward function based on the task type field in the dataset.

### 3.3 SLURM Job (`track3/train_grpo.slurm`)

Same SLURM header as SFT; increase `--time=48:00:00`. Same module + venv activation order (modules first, then `source /leonardo/home/userexternal/adiasse0/venvs/unsloth/bin/activate`). Use `GRPOTrainer` from TRL initialized from the SFT checkpoint.

---

## Phase 4 — Inference & Submission

### 4.1 Inference Script (`track3/inference.py`)

```bash
python track3/inference.py \
    --checkpoint checkpoints/track3_grpo/final \
    --test_data data/track3/annotations/test_combined.json \
    --video_dir data/track3/videos \
    --output submissions/track3_tar.csv
```

### 4.2 Output Post-Processing

Per task type, enforce the required format:
- **Binary**: extract first `Yes`/`No` from model output (regex)
- **MCQ**: extract first letter A–D
- **Open-ended**: strip `<think>...</think>` block, return remaining text
- **Temporal**: parse and validate `{"start": "MM:SS", "end": "MM:SS"}`

### 4.3 Submission Files

Three separate CSVs, one per leaderboard:

```
submissions/
├── track3_tar.csv       # TAR leaderboard
├── track3_fetv.csv      # FETV leaderboard
└── track3_psi_vqa.csv   # PSI VQA leaderboard
```

Each CSV format:
```
item_index,prediction
0,Yes
1,B
2,"The vehicle ran a red light causing the collision..."
```

---

## Directory Structure (Target)

```
track3/
├── PLAN.md               # this file
├── train_sft.py
├── train_sft.slurm
├── train_grpo.py
├── train_grpo.slurm
├── inference.py
├── evaluate.py
├── data_utils.py         # dataset loading + format conversion
└── postprocess.py        # output formatting per task type

data/track3/
├── annotations/          # HF dataset download
├── videos/               # ~150 GB video files
├── train_all.jsonl       # merged + converted training data
└── val_all.jsonl         # held-out validation set

checkpoints/
├── track3_sft/
└── track3_grpo/

submissions/
├── track3_tar.csv
├── track3_fetv.csv
└── track3_psi_vqa.csv

logs/track3/
```

---

## Key Technical Decisions

| Decision | Choice | Reason |
|---|---|---|
| Model | `Qwen3-VL-8B-Thinking` preferred over Instruct | Native `<think>` CoT matches the annotation format |
| Unsloth API | `FastVisionModel` (not `FastLanguageModel`) | Required for all vision/video models |
| LoRA rank | 16 | Balance between capacity and VRAM; increase to 32 if accuracy plateaus |
| Batch strategy | All 10 task types jointly | Prevents task-specific overfitting; dataset is already balanced |
| GRPO reward | BERTScore F1 for open-ended | Directly optimizes the primary evaluation metric |
| `dataset_num_proc` | Must be set to `1` | Known TRL crash with multiprocessing on video datasets |
