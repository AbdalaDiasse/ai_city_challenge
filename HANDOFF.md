# Session Handoff Log

This file is the living record of the project. Update it at the end of every working session.
It tracks decisions that changed, things that failed, and the current state so any future session
(human or Claude) can pick up without losing context.

**Format per entry**: date · what changed or failed · why · what to do next.

---

## Current State (as of 2026-07-05)

**Phase**: Phase 2 — SFT training scripts fixed, ready to resubmit.
**Active track**: Track 3 (Track 6 deferred to stage 2).
**Blocker**: None. Resubmit `train_sft.slurm` — two bugs from first run are fixed.

### What exists
- `CLAUDE.md` — project-wide guidance (HPC env, SLURM config, storage path)
- `track3/PLAN.md` — full 4-phase execution plan for Track 3
- `track3/download_data.sh` — downloads HF annotations + triggers video download script
- `track3/data_utils.py` — loads task JSONs, converts to Qwen3-VL conversation format, splits train/val
- `track3/prepare_dataset.py` — CLI entry point for the full data pipeline
- `track3/postprocess_videos.py` — post-processes partially downloaded datasets (so-tad, TAD, HTV, barbados)
- `track3/train_sft.py` — Unsloth SFT fine-tuning script (FastVisionModel + LoRA + SFTTrainer) — **both bugs fixed**
- `track3/train_sft.slurm` — SLURM job for 1 node × 4 GPU training via torchrun
- `requirements.txt` — pip dependencies
- `HANDOFF.md` — this file

### Data on disk ($WORK = /leonardo_work/AIH4A_syrate)
- `data/track3/annotations/train/*.json` — 10 task JSON files (44,040 items)
- `data/track3/videos/` — all 8 source datasets downloaded and post-processed
- `data/track3/train_all.jsonl` — 38,662 training conversations
- `data/track3/val_all.jsonl` — 4,286 validation conversations
- `data/track3/dataset_stats.json` — coverage stats
- **Coverage**: 42,948 / 44,040 (98%) — 1,092 missing from UCF_Crimes and Vad-R1 Normal splits (source files removed upstream, not recoverable)

### What does not exist yet
- GRPO code (`train_grpo.py`, `train_grpo.slurm`)
- Inference / submission code (`inference.py`, `postprocess.py`)
- Any checkpoints

### Immediate next step
```bash
cd /leonardo/home/userexternal/adiasse0/ai/ai_city_challenge
mkdir -p logs/track3_sft
sbatch track3/train_sft.slurm
squeue -u $USER
```
Checkpoints land at `$WORK/checkpoints/track3_sft/sft_v1/`.

---

## Decision Log

### 2026-07-03 — Storage location
**Decision**: All large files (datasets, videos, checkpoints) stored under `/leonardo_work/AIH4A_syrate/`.
**Why**: Home directory on Leonardo has a tight quota. `$WORK` is the shared project scratch space.
**Impact**: All scripts use `export WORK=/leonardo_work/AIH4A_syrate` at the top.

### 2026-07-03 — Model choice: Qwen3-VL-8B
**Decision**: Use `unsloth/Qwen3-VL-8B-Instruct` as base; switch to Thinking variant if CoT quality is poor.
**Why**: 8B fits on 4×A100 with 4-bit quantization; Unsloth has native support and a GRPO notebook for this exact model.
**Open question**: Instruct vs Thinking — Thinking variant natively produces `<think>` blocks which matches the dataset's `reasoning` field. Test both after Phase 2.

### 2026-07-03 — Training framework: Unsloth + FastVisionModel
**Decision**: Use `FastVisionModel` (not `FastLanguageModel`) for all training.
**Why**: Qwen3-VL is a vision model; `FastLanguageModel` does not handle the vision encoder.

### 2026-07-03 — Venv path and SLURM activation order
**Decision**: Activate venv with absolute path `/leonardo/home/userexternal/adiasse0/venvs/unsloth/bin/activate`, always after module loads.
**Why**: `~/` does not reliably expand on SLURM compute nodes. Modules must be loaded before venv activation to avoid CUDA library path conflicts.
**SLURM order**: `module load` → `unset PYTORCH_CUDA_ALLOC_CONF` → `source venv` → `export WORK`

---

## Failed / Abandoned Approaches

### 2026-07-04 — Correct torch version: 2.12.1+cu126
**What**: Long chain of version conflicts: cu121 caps at torch 2.5.1 → too old for unsloth_zoo. torch 2.6.0 → too old for torchao (`register_constant` missing). torch 2.11.0 was installed but still had conflicts.
**Fix**: Use `cuda/12.6` + `torch==2.12.1+cu126` (latest available for cu126). Install torch FIRST, then unsloth+deps after, so pip resolves everything against the locked torch version. SLURM scripts use `module load cuda/12.6`.

### 2026-07-03 — `hf_transfer` install failure (root cause: Python 3.6 venv)
**What**: `pip install hf_transfer` fails — "No matching distribution found". Venv also had no Unsloth, torch, or anything installed.
**Why**: The `~/venvs/unsloth` venv was created with system Python 3.6.8, which is EOL and unsupported by all modern ML packages.
**Fix**:
  - Recreate venv using Python 3.11.6 from `cineca-ai/4.3.0` module (must load module first, then `python3 -m venv ~/venvs/unsloth`)
  - Install `unsloth[cu121-torch260]` + `trl peft accelerate datasets huggingface_hub bert-score`
  - Old venv backed up to `~/venvs/unsloth_py36_backup`
  - `hf_transfer` still made optional in `download_data.sh` as a defensive measure.

---

## Known Gotchas

| Issue | Fix |
|---|---|
| `PYTORCH_CUDA_ALLOC_CONF` causes fragmentation | Always `unset` then re-export with `expandable_segments:False` |
| TRL crashes with multiprocessing on video datasets | Set `dataset_num_proc=1` in `SFTConfig` |
| `~/` may not expand on compute nodes | Use absolute paths in all SLURM scripts |
| HF cache defaults to home dir (quota risk) | Set `export HF_HOME=$WORK/hf_cache` |
| `pip list` shows hundreds of packages inside venv | cineca-ai module sets PYTHONPATH; run `unset PYTHONPATH` after venv activation. Venv packages still take precedence at runtime. |
| `import unsloth` must be FIRST import | Unsloth patches trl/transformers/peft at import time; if those load first, patching is incomplete and training silently uses unoptimized paths. |
| PyArrow fails on heterogeneous `content` field | Qwen3-VL messages have `content` as string (system/assistant) OR list (user with video). HF `Dataset.from_list` crashes. Fix: use `torch.utils.data.Dataset` subclass (`ConversationDataset`) — bypasses PyArrow entirely. |

---

## Session Notes

### 2026-07-03 — Project initialization
- Created repo structure, `CLAUDE.md`, `track3/PLAN.md`
- Added Unsloth skill (`.claude/skills/unsloth/`) and HuggingFace skills marketplace
- Confirmed Unsloth venv at `/leonardo/home/userexternal/adiasse0/venvs/unsloth`
- Reference SLURM template: `../mits/train_cinera.slurm` (uses ms-swift, not Unsloth, but cluster config is proven)

### 2026-07-05 — Phase 2 first job run + two bug fixes (job 48588165)
- First SLURM job crashed at dataset loading, no GPU time wasted
- **Bug 1**: `import unsloth` was not the first import → fixed (now line 1 of train_sft.py)
- **Bug 2**: `Dataset.from_list()` crashed with PyArrow schema conflict on heterogeneous `content` field → fixed by replacing `to_hf_dataset()` with `ConversationDataset(TorchDataset)` class
- `train_sft.py` is now correct; ready to resubmit as `sft_v1`
- **Not yet done**: GRPO script, inference script, submission script

### 2026-07-03–04 — Phase 1: data download + post-processing
- All 8 video datasets downloaded and post-processed to mp4
- so-tad: 51-part PKWARE zip extracted via 7z (conda p7zip)
- HTV: 254 .avi transcoded to .mp4 via imageio-ffmpeg
- TAD: JPG frame folders stitched to mp4 via stitch_tad_frames.py
- barbados: downloaded via download_videos.py
- 98% coverage (42,948 / 44,040): 1,092 missing from UCF_Crimes + Vad-R1 Normal splits (removed upstream, not recoverable)

### 2026-07-03 — Phase 1 data setup implementation
- Wrote `track3/download_data.sh` — downloads HF annotations + triggers vendor video download script
- Wrote `track3/data_utils.py` — core conversion logic:
  - `build_conversation()`: maps TAR item → Qwen3-VL multi-modal format with `<think>` CoT
  - `load_all_tasks()`: loads all 10 task JSONs, skips missing gracefully
  - `stratified_split()`: 90/10 split per task type so validation covers all task categories
  - `save_jsonl()` / `load_jsonl()` helpers
- Wrote `track3/prepare_dataset.py` — CLI orchestrator, produces `train_all.jsonl`, `val_all.jsonl`, `dataset_stats.json`
- Wrote `requirements.txt` at repo root
- **Next session should start with**: running `bash track3/download_data.sh` on a login node, then `python track3/prepare_dataset.py`, then writing `track3/train_sft.py`

---

## How to Update This File

At the end of each session, append to the relevant section:
- **Decision Log** — any choice that was reconsidered or newly made
- **Failed / Abandoned Approaches** — what was tried, what went wrong, what error or result was observed
- **Known Gotchas** — environment or library quirks discovered
- **Session Notes** — brief summary of what was done and the exact next step
