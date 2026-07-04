# Session Handoff Log

This file is the living record of the project. Update it at the end of every working session.
It tracks decisions that changed, things that failed, and the current state so any future session
(human or Claude) can pick up without losing context.

**Format per entry**: date · what changed or failed · why · what to do next.

---

## Current State (as of 2026-07-03)

**Phase**: Phase 1 — Data setup scripts written, download not yet run.
**Active track**: Track 3 (Track 6 deferred to stage 2).
**Blocker**: Dataset not downloaded yet. Must run `bash track3/download_data.sh` on a Leonardo login node. Videos are ~150 GB.

### What exists
- `CLAUDE.md` — project-wide guidance (HPC env, SLURM config, storage path)
- `track3/PLAN.md` — full 4-phase execution plan for Track 3
- `track3/download_data.sh` — downloads HF annotations + triggers video download script
- `track3/data_utils.py` — loads task JSONs, converts to Qwen3-VL conversation format, splits train/val
- `track3/prepare_dataset.py` — CLI entry point for the full data pipeline
- `requirements.txt` — pip dependencies
- `HANDOFF.md` — this file

### What does not exist yet
- Training code (`train_sft.py`, `train_sft.slurm`)
- GRPO code (`train_grpo.py`, `train_grpo.slurm`)
- Inference / submission code (`inference.py`, `postprocess.py`)
- Data on disk (`/leonardo_work/AIH4A_syrate/data/track3/` is empty — download not yet run)
- Any checkpoints

### Immediate next step
```bash
# On a Leonardo login node:
bash track3/download_data.sh

# After videos arrive:
export WORK=/leonardo_work/AIH4A_syrate
python track3/prepare_dataset.py
```
Then move to Phase 2 — write `track3/train_sft.py`.

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

---

## Session Notes

### 2026-07-03 — Project initialization
- Created repo structure, `CLAUDE.md`, `track3/PLAN.md`
- Added Unsloth skill (`.claude/skills/unsloth/`) and HuggingFace skills marketplace
- Confirmed Unsloth venv at `/leonardo/home/userexternal/adiasse0/venvs/unsloth`
- Reference SLURM template: `../mits/train_cinera.slurm` (uses ms-swift, not Unsloth, but cluster config is proven)

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
