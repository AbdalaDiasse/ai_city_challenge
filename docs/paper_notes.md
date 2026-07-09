# AI City Challenge 2026 — Track 3: Paper Reference Notes

**Purpose**: Living reference for writing the challenge paper. Documents the full pipeline, model choices, training strategy, engineering decisions, and known results. Update after each new experiment.

**Challenge**: [AI City Challenge 2026 — Track 3: Anomalous Events in Transportation](https://www.aicitychallenge.org/2026-track3/)

---

## 1. Task Definition

Given a short video clip from traffic surveillance infrastructure (CCTV, fisheye traffic cam, dashcam), the system must answer questions about anomalous events, producing chain-of-thought reasoning before the final answer.

### 1.1 Task Taxonomy (10 types, 3 groups)

| Group | Task Name | Answer Format |
|---|---|---|
| **Basic** | `bcq` — Binary event verification | `Yes` / `No` |
| | `mcq` — Multiple-choice | `A` / `B` / `C` / `D` |
| | `open_qa` — Open-ended QA | Free-form text |
| **Scene** | `scene_description` | Free-form text |
| | `video_summarization` | Free-form text |
| **Temporal** | `temporal_localization` | `{"start": "MM:SS", "end": "MM:SS"}` |
| | `causal_linkage` | Free-form text |
| | `temporal_description` | Free-form text |
| | `bcq_openended` — Binary with explanation | `Yes`/`No` + explanation |
| | `mcq_openended` — MCQ with explanation | `A`–`D` + explanation |

---

## 2. Dataset — Traffic Anomaly Reasoning (TAR)

**HuggingFace**: `nvidia/PhysicalAI-Traffic-Anomaly-Reasoning`
**Format name**: `tao-vl-reason-v1.0`

### 2.1 Training Data

| Split | Items | Videos | Total Duration |
|---|---|---|---|
| Train | 44,040 annotations | 3,670 clips | ~26.1 hours |

Source datasets: VAD-R1, TAD, Accident-Bench, SO-TAD, TADBenchmark, UCF Crime, and others.

**Data format** (one JSON file per task type):
```json
{
  "video_id": "TAD/01_Accident_001.mp4",
  "question": "<task-specific prompt>",
  "answer": "<expected output>",
  "reasoning": "<chain-of-thought explanation>"
}
```

**Actual coverage after download**: 42,948 / 44,040 **(98%)** — 1,092 items missing from UCF_Crimes and Vad-R1 Normal splits (removed upstream, not recoverable).

**Processed files on disk** (`$WORK = /leonardo_work/AIH4A_syrate`):

| File | Count | Notes |
|---|---|---|
| `data/track3/train_all.jsonl` | 38,662 | 90% stratified split per task type |
| `data/track3/val_all.jsonl` | 4,286 | 10% per task type |
| `data/track3/dataset_stats.json` | — | Coverage stats |

**Stratified split**: `stratified_split()` in `track3/data_utils.py` — 90/10 split per task type ensures all 10 task categories appear in validation.

### 2.2 Test Data

| Test Set | Items | Source | Evaluation Metric |
|---|---|---|---|
| **TAR** | 960 items, 80 clips | YouTube | Primary leaderboard |
| **FETV** | 200 clips | Fisheye traffic-violation cameras | Secondary |
| **PSI VQA** | 40 clips | Egocentric dashcam | Secondary |

### 2.3 Data Preprocessing

The `build_conversation()` function in `track3/data_utils.py` converts each TAR item to Qwen3-VL multi-modal conversation format:

```
System: You are an expert traffic surveillance analyst...
User: [video frames] + [question]
Assistant: <think>[reasoning field from dataset]</think> [answer field]
```

The `<think>` block wraps the dataset's `reasoning` field, teaching the model to produce chain-of-thought reasoning before the final answer.

---

## 3. Evaluation Metrics

### 3.1 TAR (Primary Leaderboard)

| Task Type | Metric |
|---|---|
| `bcq`, `mcq` | Exact accuracy (0/1) |
| All other 8 types | BERTScore F1 |
| **Final score** | Unweighted mean across 9 scored task types |

### 3.2 FETV

`0.25 × CIDEr + 0.25 × BERTScore + 0.5 × MacroF1`

### 3.3 PSI VQA

`0.25 × T1 + 0.25 × T2 + 0.25 × T3 + 0.25 × T4`

### 3.4 Submission Format

CSV with columns: `item_index`, `prediction`.
- Binary: `"Yes"` or `"No"`
- MCQ: single letter `"A"`, `"B"`, `"C"`, `"D"`
- Open-ended: free-form text
- Temporal: `{"start": "MM:SS", "end": "MM:SS"}` (7-second tolerance)

---

## 4. Models

### 4.1 Instruct Pipeline (Primary)

| Property | Value |
|---|---|
| **Base model** | `unsloth/Qwen3-VL-8B-Instruct` |
| **Architecture** | Qwen3-VL (model type `qwen3_vl`); ~8B parameters |
| **HF 4-bit quantized** | `unsloth/qwen3-vl-8b-instruct-unsloth-bnb-4bit` |
| **Training framework** | Unsloth `FastVisionModel` + LoRA + TRL |
| **SFT precision** | BF16 (no quantization — accuracy-first for competition) |
| **GRPO precision** | 4-bit (memory constraint for GRPO's multi-completion sampling) |
| **System prompt** | "You are an expert traffic surveillance analyst. Watch the provided video carefully, then reason step-by-step about any anomalous events before giving your final answer." |

### 4.2 Thinking Pipeline (Secondary / Ablation)

| Property | Value |
|---|---|
| **Base model** | `unsloth/Qwen3-VL-8B-Thinking` |
| **HF 4-bit quantized** | `unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit` |
| **Key difference** | Pre-trained for CoT reasoning; `<think>` blocks are native, not taught |
| **System prompt** | No "reason step-by-step" instruction — model does this natively |
| **Training difference** | Lower LR (`5e-5` vs `1e-4`), `modules_to_save=["lm_head", "embed_tokens"]` |
| **GRPO completion budget** | 1024 tokens (vs 512 for Instruct — Thinking chains are longer) |

### 4.3 Why Not Qwen3.5-VL?

`qwen3_5_vl` is not in transformers 5.5.0's `CONFIG_MAPPING` and has not been released as of 2026-07-07. Unsloth has stub code for it but no working implementation exists yet. Decision: use Qwen3-VL-8B only.

---

## 5. Training Strategy

### 5.1 Overview

Two parallel pipelines, both following the same SFT → GRPO pattern:

```
Instruct pipeline:
  Qwen3-VL-8B-Instruct (BF16) → SFT (train_sft_bf16.py) → GRPO (train_grpo.py)

Thinking pipeline:
  Qwen3-VL-8B-Thinking (BF16) → SFT (train_sft_thinking.py) → GRPO (train_grpo_thinking.py)
```

### 5.2 Phase 1 — Supervised Fine-Tuning (SFT)

**Instruct SFT** (`train_sft_bf16.py` / `train_sft_bf16.slurm`):

| Hyperparameter | Value |
|---|---|
| Model | `unsloth/Qwen3-VL-8B-Instruct` BF16 |
| LoRA rank `r` | 32 |
| LoRA alpha | 32 |
| LoRA dropout | 0 |
| Target modules | `"all-linear"` |
| `finetune_vision_layers` | True |
| `finetune_language_layers` | True |
| `finetune_attention_modules` | True |
| `finetune_mlp_modules` | True |
| Max sequence length | 4096 |
| Epochs | 3 |
| Learning rate | `1e-4` |
| Batch size (per device) | 1 |
| Gradient accumulation | 4 |
| Optimizer | `adamw_8bit` |
| LR scheduler | cosine |
| Warmup ratio | 0.05 |
| `dataset_num_proc` | **1** (required — crashes with >1 for VLMs) |
| Random seed | 3407 |
| Checkpoint path | `$WORK/checkpoints/track3_sft/sft_v1_bf16/` |

**Thinking SFT** (`train_sft_thinking.py`): Same config with `lr=5e-5` (lower — Thinking model needs gentler updates), `modules_to_save=["lm_head", "embed_tokens"]`, and the Thinking-model system prompt.

**Key data insight**: Assistant turns in training data include `<think>...</think>` blocks (from the dataset's `reasoning` field). The model learns to produce chain-of-thought reasoning followed by a clean answer within the same output.

### 5.3 Phase 2 — GRPO (Reinforcement Learning)

**Instruct GRPO** (`train_grpo.py` / `train_grpo_4nodes.slurm`):

| Hyperparameter | Value |
|---|---|
| Base model | `unsloth/qwen3-vl-8b-instruct-unsloth-bnb-4bit` (4-bit) |
| Starting policy | SFT LoRA adapter |
| Learning rate | `5e-6` |
| Batch size (per device) | 1 |
| `num_generations` | 4 (completions per prompt for GRPO gradient) |
| Gradient accumulation | 4 |
| Max prompt length | 2048 |
| Max completion length | 512 (Instruct) / 1024 (Thinking) |
| Optimizer | `adamw_8bit` |
| LR scheduler | cosine |
| Seed | 3407 |

**Reward functions** (task-specific, mapped in `compute_reward()`):

| Task Type | Reward | Range |
|---|---|---|
| `bcq`, `bcq_openended` | Exact match on Yes/No | {0, 1} |
| `mcq`, `mcq_openended` | Exact match on A–D letter | {0, 1} |
| `temporal_localization` | Temporal IoU of predicted vs. GT interval | [0, 1] |
| All open-ended tasks | ROUGE-L F1 | [0, 1] |
| Unknown | 0.5 if non-empty, else 0 | — |

ROUGE-L was chosen over BERTScore as the GRPO reward because it is computed in pure Python (no GPU) and runs much faster per step, avoiding reward computation becoming the bottleneck.

**TRL requirement**: Dataset columns used as reward function kwargs must be named **exactly** `"answers"` and `"task_types"` — TRL passes dataset columns to `reward_fn` as kwargs with those exact names.

### 5.4 Multi-Node GRPO (4 nodes × 4 GPUs = 16 GPUs)

SLURM + PyTorch `torchrun` multi-node pattern:

```bash
srun torchrun \
    --nproc_per_node=4 \
    --nnodes=$SLURM_NNODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    track3/train_grpo.py ...
```

**Critical**: `srun` is required. Without `srun`, SLURM executes `torchrun` only on the head node — the other 3 nodes are idle and the rendezvous times out with "1/4 clients joined."

The `c10d` backend automatically handles `--node_rank` via the rendezvous protocol, eliminating the need for explicit `$SLURM_NODEID` injection.

---

## 6. Inference Pipeline

**Script**: `track3/inference.py`
**SLURM**: `track3/inference.slurm` (SFT BF16 adapter), `track3/inference_grpo.slurm` (GRPO 4-bit adapter)

### 6.1 Sharded Parallel Inference

4 GPU workers run in parallel, each handling 1/4 of test items:

```bash
for SHARD_ID in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$SHARD_ID python track3/inference.py \
        --num_shards 4 --shard_id $SHARD_ID &
done
wait
cat raw_predictions_shard*.jsonl > raw_predictions.jsonl
```

Items are sorted by `video_id` before sharding so each shard processes full videos consecutively (frame cache is effective).

### 6.2 Video Loading

- **Library**: `decord` (faster than OpenCV for video)
- **Frame rate**: 1 fps (default; configurable via `--fps`)
- **Max pixels**: `360 × 420 = 151,200` per frame (controls GPU memory vs. quality tradeoff)
- **Frame cache**: consecutive items from the same video reuse cached frames — video is read from disk only once per clip

### 6.3 Token Budget

| Task group | `max_new_tokens` | Reason |
|---|---|---|
| `bcq`, `mcq` (SHORT_TASKS) | **256** | Enough to close `</think>` then produce Yes/No or A–D |
| All open-ended + temporal | **1024** | Full CoT + detailed answer |

**Critical fix**: Original value for `bcq`/`mcq` was 32 — the model would start `<think>` reasoning but run out of tokens before producing the final answer. Raw and clean predictions were identical truncated `<think>` blocks, causing BCQ=0.5 (random-guess level) and MCQ=0.2.

### 6.4 Think-Block Stripping

```python
def strip_think(text: str) -> str:
    # Pass 1: remove complete <think>...</think> blocks
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Pass 2: remove unclosed <think> to end of string
    clean = re.sub(r"<think>.*", "", clean, flags=re.DOTALL).strip()
    return clean
```

Two-pass approach handles both complete blocks and truncated (unclosed) blocks that occur when the model runs out of tokens mid-reasoning.

### 6.5 Base Model Matching (Critical)

| Adapter | Base model for inference |
|---|---|
| SFT BF16 adapter | `unsloth/Qwen3-VL-8B-Instruct` (BF16, `LOAD_FLAG=""`) |
| GRPO adapter | `unsloth/qwen3-vl-8b-instruct-unsloth-bnb-4bit` (4-bit, `LOAD_FLAG="--load_in_4bit"`) |
| Thinking SFT adapter | `unsloth/Qwen3-VL-8B-Thinking` (BF16) |
| Thinking GRPO adapter | `unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit` (4-bit) |

Mismatching base model to adapter corrupts activations silently — no crash, just wrong outputs.

### 6.6 Postprocessing

`track3/postprocess.py` maps `clean_prediction` → submission-format string per task type:
- Binary: extract `Yes`/`No` prefix
- MCQ: extract `A`–`D` letter
- Temporal: extract and reformat JSON timestamp
- Open-ended: pass through cleaned text

---

## 7. Results (Preliminary)

**Status as of 2026-07-07**: Results from broken inference (max_new_tokens_short=32). Correct runs queued.

| Run | BCQ | MCQ | Notes |
|---|---|---|---|
| `sft_v1_bf16` (broken inference) | 0.5 | 0.2 | Token budget too small — truncated CoT, no answer |
| `sft_v1_bf16_lora_v2` (fixed) | [PENDING] | [PENDING] | max_new_tokens=256; sbatch queued |
| `grpo_v1_ckp1100` | [PENDING] | [PENDING] | GRPO checkpoint-1100 inference queued |
| `sft_thinking_v1` | [PENDING] | [PENDING] | Thinking SFT not yet run |
| `grpo_thinking_v1` | [PENDING] | [PENDING] | Thinking GRPO not yet run |

**Root cause of initial poor scores**: BCQ score 0.5 (random-guess level) and MCQ score 0.2 confirmed to be caused by `max_new_tokens_short=32`. The model starts `<think>` reasoning but the token budget is exhausted before `</think>` + final answer. Both `raw_prediction` and `clean_prediction` contained only truncated `<think>` blocks with no answer text.

[CLAIM NEEDS EVIDENCE] — Results from corrected runs pending. Update this table once inference completes.

---

## 8. Key Engineering Decisions

### 8.1 Why BF16 for SFT

4-bit quantization (QLoRA) reduces memory but introduces quantization noise. For competition where every BERTScore point matters, BF16 preserves full model precision during SFT. 4-bit is used only for GRPO where the memory constraint from generating `num_generations=4` completions per step forces it.

### 8.2 Why LoRA r=32

4-bit baseline recommendations use r=16. BF16 with full A100 memory (63 GiB per GPU) allows doubling to r=32 without OOM. Higher rank ≈ more expressive adapter for the 10-task multi-modal fine-tuning objective.

### 8.3 Why ROUGE-L for GRPO Reward (Not BERTScore)

BERTScore requires running a separate BERT encoder on every sampled completion. With `num_generations=4`, this means 4× reward model calls per training step on CPU/GPU, becoming a major bottleneck. ROUGE-L is pure Python LCS — zero additional model calls, computed in microseconds. Accepted accuracy cost justified by training speed gain.

### 8.4 Why `dataset_num_proc=1`

TRL's `SFTTrainer` with vision datasets uses multiprocessing to preprocess/tokenize. With `num_proc > 1`, the video/image content in message lists causes serialization failures (fork safety issues with PIL Images and decord buffers). Setting `dataset_num_proc=1` forces single-process preprocessing — required for all VLM SFT runs.

### 8.5 Why `torch.utils.data.Dataset` Not HF `Dataset`

Qwen3-VL conversation format has `content` as a string in system/assistant turns, but as a list of dicts in user turns (with video and text items). HuggingFace's `Dataset.from_list()` fails with a PyArrow schema error because the schema is heterogeneous. Wrapping records in a plain `torch.utils.data.Dataset` subclass bypasses PyArrow entirely.

### 8.6 Why `import unsloth` Must Be First

Unsloth patches `transformers`, `trl`, and `peft` at import time to inject its optimized kernels. If any of those libraries loads before `unsloth`, patching is incomplete and training silently falls back to unoptimized paths with no error or warning.

### 8.7 HF Cache Format vs `--local-dir`

`huggingface-cli download --local-dir <path>` saves files in a flat layout, which is incompatible with `FastVisionModel.from_pretrained(hub_id)` when `HF_HUB_OFFLINE=1`. The offline resolver looks for the standard `blobs/snapshots/refs/trees/` structure. Always download without `--local-dir` so the HF Hub caches to the standard directory. Check existence with:

```bash
MODEL_CACHE="$HF_HOME/hub/models--unsloth--Qwen3-VL-8B-Thinking"
if [ ! -d "$MODEL_CACHE/snapshots" ]; then
    huggingface-cli download unsloth/Qwen3-VL-8B-Thinking
fi
export HF_HUB_OFFLINE=1  # must come AFTER download
```

### 8.8 `PYTORCH_CUDA_ALLOC_CONF` Fragmentation Fix

The Leonardo A100 cluster's default CUDA memory allocator settings cause fragmentation in long training runs. Always override:

```bash
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
```

---

## 9. Infrastructure

| Property | Value |
|---|---|
| **Cluster** | Leonardo (CINECA), Italy |
| **GPU** | NVIDIA A100 SXM4 (~63 GiB VRAM per GPU) |
| **SLURM account** | `AIH4A_syrate` |
| **Partition** | `boost_usr_prod` |
| **QoS** | `boost_qos_lprod` |
| **Venv** | `/leonardo/home/userexternal/adiasse0/venvs/unsloth` (Python 3.11) |
| **Storage** | `/leonardo_work/AIH4A_syrate/` (`$WORK`) — all large files here |
| **HF cache** | `$WORK/hf_cache` |
| **Module load** | `cuda/12.6`, `profile/deeplrn`, `cineca-ai/4.3.0` |

**Confirmed working package versions** (as of 2026-07-04):
```
torch==2.12.1+cu126
unsloth (from git main)
unsloth_zoo
bitsandbytes
trl
peft
accelerate
datasets
huggingface_hub
bert-score
numpy
decord
```

**SLURM module load order** (order matters for CUDA library paths):
```bash
module load cuda/12.6
module load profile/deeplrn
module load cineca-ai/4.3.0
source /leonardo/home/userexternal/adiasse0/venvs/unsloth/bin/activate
unset PYTHONPATH   # prevents cineca-ai from shadowing venv packages
```

---

## 10. File Inventory

### Training

| File | Purpose |
|---|---|
| `track3/train_sft_bf16.py` | SFT — Instruct model, BF16 |
| `track3/train_sft_bf16.slurm` | SLURM job for Instruct SFT |
| `track3/train_grpo.py` | GRPO — Instruct model, 4-bit |
| `track3/train_grpo.slurm` | SLURM job for single-node GRPO |
| `track3/train_grpo_4nodes.slurm` | SLURM job for 4-node multi-node GRPO |
| `track3/train_sft_thinking.py` | SFT — Thinking model, BF16 |
| `track3/train_sft_thinking.slurm` | SLURM job for Thinking SFT |
| `track3/train_grpo_thinking.py` | GRPO — Thinking model, 4-bit |
| `track3/train_grpo_thinking.slurm` | SLURM job for Thinking GRPO |

> **DO NOT MODIFY**: `track3/train_sft.py`, `track3/train_sft.slurm` — original SFT baseline, preserved as reference.

### Data and Inference

| File | Purpose |
|---|---|
| `track3/data_utils.py` | Core data loading and conversation conversion |
| `track3/prepare_dataset.py` | CLI: produces `train_all.jsonl`, `val_all.jsonl` |
| `track3/inference.py` | Inference on test set, sharded multi-GPU |
| `track3/inference.slurm` | SLURM job for SFT BF16 inference |
| `track3/inference_grpo.slurm` | SLURM job for GRPO 4-bit inference |
| `track3/postprocess.py` | Maps raw JSONL predictions → submission CSV |
| `track3/postprocess_videos.py` | Repairs partial video downloads (so-tad, TAD, HTV, barbados) |
| `track3/download_model.sh` | Helper: download model weights to HF cache |

### Checkpoints (on disk, not in repo)

| Path | Contents |
|---|---|
| `$WORK/checkpoints/track3_sft/sft_v1_bf16/` | Instruct SFT checkpoints |
| `$WORK/checkpoints/track3_sft/sft_v1_bf16/lora_adapter/` | Final LoRA adapter |
| `$WORK/checkpoints/track3_grpo/grpo_v1/` | GRPO checkpoints |
| `$WORK/checkpoints/track3_sft/sft_thinking_v1/` | Thinking SFT checkpoints |
| `$WORK/checkpoints/track3_grpo/grpo_thinking_v1/` | Thinking GRPO checkpoints |

---

## 11. Bugs Fixed

| Bug | Symptom | Root Cause | Fix |
|---|---|---|---|
| BCQ=0.5, MCQ=0.2 | Scores at random-guess level | `max_new_tokens_short=32` — CoT truncated mid-think, no answer produced | Raised to 256; added two-pass `strip_think()` |
| Multi-node GRPO timeout | "1/4 clients joined" at rendezvous | `torchrun` without `srun` — only head node executed | `srun torchrun --rdzv_backend=c10d` |
| GRPO inference wrong base | Silent output corruption | GRPO adapter trained on 4-bit base; running with BF16 base mismatches | Created `inference_grpo.slurm` with `LOAD_FLAG="--load_in_4bit"` |
| Thinking model download | `OSError` at `from_pretrained` | `--local-dir` uses flat file layout; HF offline resolver needs `blobs/snapshots/` | Removed `--local-dir` from download command |
| `HF_HUB_OFFLINE=1` before download | Download fails immediately | Offline flag exported before model check; blocks the download it was guarding | Moved offline exports to after the download block |
| `dataset_num_proc` in GRPOConfig | `TypeError: unexpected keyword argument` | `GRPOConfig` does not accept `dataset_num_proc` (different from `SFTConfig`) | Removed from `GRPOConfig` |
| GRPO reward_fn missing kwargs | `TypeError: missing required argument` | TRL passes dataset columns by exact column name | Renamed dataset keys to `"answers"` and `"task_types"` |
| `_strip_think` TypeError | `AttributeError: list has no .replace` | TRL passes completions as list of message dicts, not strings | Added list-unwrapping at start of `_strip_think()` |
| PyArrow schema error | Crash at `Dataset.from_list()` | `content` field is heterogeneous (string vs list) | Replaced HF Dataset with `torch.utils.data.Dataset` subclass |
| Stray bare path in inference.slurm | Job crash at bash execution | Manual edit left `/leonardo_work/.../checkpoint-800` as a standalone line | Removed stray line |

---

## 12. Potential Paper Contributions / Claims

> All items below are hypotheses. Mark with [CONFIRMED] once supported by experimental results.

**[CLAIM NEEDS EVIDENCE] C1**: SFT on the full TAR training set with chain-of-thought supervision improves all 9 scored task types relative to the zero-shot Qwen3-VL-8B-Instruct baseline.

**[CLAIM NEEDS EVIDENCE] C2**: GRPO with task-specific reward functions (exact-match for binary/MCQ, temporal IoU, ROUGE-L) further improves over SFT by reinforcing correct reasoning paths.

**[CLAIM NEEDS EVIDENCE] C3**: The Qwen3-VL-8B-Thinking variant, fine-tuned with a lower learning rate and token-preserving settings, outperforms the Instruct variant on open-ended tasks where longer chain-of-thought reasoning is beneficial.

**[CLAIM NEEDS EVIDENCE] C4**: Sharded multi-GPU inference with a per-video frame cache reduces inference wall-clock time by X% compared to per-item video loading.

---

## 13. Next Steps

### Immediate (before results exist)
- [ ] Run: `sbatch track3/inference.slurm` → `sft_v1_bf16_lora_v2` (fixed token budget)
- [ ] Run: `sbatch track3/inference_grpo.slurm` → `grpo_v1_ckp1100`
- [ ] Run: `sbatch track3/train_sft_thinking.slurm` → `sft_thinking_v1`
- [ ] After Thinking SFT: run `sbatch track3/train_grpo_thinking.slurm`
- [ ] Run evaluator: `python evaluate.py --gt test.json --submission submission.csv`

### After results exist
- [ ] Fill in the Results table (Section 7) with actual numbers
- [ ] Mark claims C1–C4 with [CONFIRMED] or revise
- [ ] Add ablation: SFT vs GRPO delta per task type
- [ ] Add ablation: Instruct vs Thinking per task type
- [ ] Consider ensemble: average softmax-logit or majority vote across Instruct + Thinking predictions

### Paper writing
- [ ] Confirm target venue with team (workshop at CVPR 2026 or standalone challenge report)
- [ ] Draft methods section from Sections 3–5 above
- [ ] Collect training loss curves and reward curves for figures
- [ ] BERTScore breakdown per task type as the main results table
