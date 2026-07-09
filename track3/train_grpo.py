"""
Phase 3 — GRPO fine-tuning starting from the SFT checkpoint.

Loads the SFT LoRA adapter as the starting policy and trains with
task-specific reward functions:
  - bcq / mcq:           exact-match reward (0 or 1)
  - bcq_openended:       exact-match on yes/no prefix (0 or 1)
  - mcq_openended:       exact-match on letter prefix (0 or 1)
  - temporal_localization: temporal IoU reward (0–1)
  - open-ended tasks:    ROUGE-L reward (0–1, faster than BERTScore)

Usage (multi-GPU via torchrun):
    torchrun --nproc_per_node 4 track3/train_grpo.py

Key env vars:
    WORK   — storage root (default: /leonardo_work/AIH4A_syrate)
    RUN    — experiment name (default: grpo_v1)
"""

import unsloth  # must be first

import argparse
import json
import os
import re
from pathlib import Path

import torch
from peft import PeftModel
from torch.utils.data import Dataset as TorchDataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastVisionModel

WORK       = os.environ.get("WORK", "/leonardo_work/AIH4A_syrate")
SFT_CKPT   = f"{WORK}/checkpoints/track3_sft/sft_v1/checkpoint-2000"
MODEL_NAME = "unsloth/qwen3-vl-8b-instruct-unsloth-bnb-4bit"

SYSTEM_PROMPT = (
    "You are an expert traffic surveillance analyst. "
    "Watch the provided video carefully, then reason step-by-step about any anomalous events "
    "before giving your final answer."
)

BCQ_TASKS      = {"bcq"}
MCQ_TASKS      = {"mcq"}
BCQ_OE_TASKS   = {"bcq_openended"}
MCQ_OE_TASKS   = {"mcq_openended"}
TEMPORAL_TASKS = {"temporal_localization"}
OPEN_TASKS     = {
    "open_qa", "scene_description", "video_summarization",
    "causal_linkage", "temporal_description",
}


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--work_dir",   default=WORK)
    p.add_argument("--sft_ckpt",   default=SFT_CKPT,
                   help="SFT checkpoint to start from")
    p.add_argument("--run_name",   default=os.environ.get("RUN", "grpo_v1"))
    p.add_argument("--epochs",     type=int,   default=1)
    p.add_argument("--lr",         type=float, default=5e-6)
    p.add_argument("--batch_size", type=int,   default=1,
                   help="Per-device batch size (GRPO generates multiple completions)")
    p.add_argument("--num_generations", type=int, default=4,
                   help="Completions per prompt for GRPO (higher = better gradient estimate)")
    p.add_argument("--max_prompt_len",  type=int, default=2048)
    p.add_argument("--max_completion_len", type=int, default=512)
    p.add_argument("--resume",     action="store_true")
    return p.parse_args()


# ── data ─────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class GRPODataset(TorchDataset):
    """
    Wraps train_all.jsonl for GRPO.

    Each record exposes:
      - "prompt":    list of messages WITHOUT the assistant turn
      - "answer":    ground-truth string (for reward computation)
      - "task_type": string (for routing to the right reward function)
    """
    def __init__(self, records: list[dict]):
        self.data = []
        for rec in records:
            msgs = rec["messages"]
            # Drop the last (assistant) turn — GRPO generates it
            prompt_msgs = [m for m in msgs if m["role"] != "assistant"]
            answer = next(
                (m["content"] for m in msgs if m["role"] == "assistant"), ""
            )
            # Strip <think>...</think> from the stored answer to get the
            # clean ground truth for reward computation.
            clean_answer = re.sub(
                r"<think>.*?</think>", "", answer, flags=re.DOTALL
            ).strip()
            self.data.append({
                "prompt":     prompt_msgs,
                "answers":    clean_answer,
                "task_types": rec.get("_meta", {}).get("task", "unknown"),
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ── reward helpers ────────────────────────────────────────────────────────────

def _strip_think(text) -> str:
    # TRL may pass completions as list of message dicts instead of plain strings
    if isinstance(text, list):
        text = " ".join(
            item["content"] if isinstance(item, dict) and "content" in item else str(item)
            for item in text
        )
    text = str(text)
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return clean if clean else text.strip()


def _extract_yesno(text: str) -> str | None:
    s = text.strip().lower()
    m = re.match(r"^(yes|no)\b", s)
    if m:
        return m.group(1)
    m = re.search(r"\b(yes|no)\b", s)
    return m.group(1) if m else None


def _extract_letter(text: str) -> str | None:
    s = text.strip()
    m = re.match(r"^\(?([A-Da-d])\)?[).\s,:]", s)
    if m:
        return m.group(1).upper()
    if re.fullmatch(r"[A-Da-d]", s):
        return s.upper()
    m = re.search(r"\b([A-D])\b", s)
    return m.group(1) if m else None


def _parse_timestamp(ts: str) -> float:
    parts = str(ts).strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return float(ts)


def _temporal_iou(pred_text: str, gt_text: str) -> float:
    """Return IoU of predicted and ground-truth time intervals."""
    def parse_json(text):
        m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        m = re.search(r'\{[^{}]*"start"[^{}]*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return None

    pred = parse_json(pred_text)
    gt   = parse_json(gt_text)
    if pred is None or gt is None:
        return 0.0
    try:
        ps, pe = _parse_timestamp(pred["start"]), _parse_timestamp(pred["end"])
        gs, ge = _parse_timestamp(gt["start"]),   _parse_timestamp(gt["end"])
        inter  = max(0.0, min(ge, pe) - max(gs, ps))
        union  = max(0.0, (ge - gs) + (pe - ps) - inter)
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


def _rouge_l(pred: str, ref: str) -> float:
    """Compute ROUGE-L F1 (no external library needed)."""
    if not pred.strip() or not ref.strip():
        return 0.0
    pred_tokens = pred.lower().split()
    ref_tokens  = ref.lower().split()
    m, n = len(pred_tokens), len(ref_tokens)

    # LCS via DP (capped to avoid quadratic blowup on very long texts)
    m = min(m, 200)
    n = min(n, 200)
    pred_tokens = pred_tokens[:m]
    ref_tokens  = ref_tokens[:n]

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    precision = lcs / m if m else 0.0
    recall    = lcs / n if n else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── reward function ───────────────────────────────────────────────────────────

def compute_reward(completion: str, answer: str, task_type: str) -> float:
    """
    Map one (completion, ground_truth) pair to a scalar reward in [0, 1].
    Called per completion inside the vectorised reward_fn below.
    """
    pred = _strip_think(completion)

    if task_type in BCQ_TASKS:
        pred_yn = _extract_yesno(pred)
        gt_yn   = _extract_yesno(answer)
        return 1.0 if (pred_yn and gt_yn and pred_yn == gt_yn) else 0.0

    if task_type in MCQ_TASKS:
        pred_l = _extract_letter(pred)
        gt_l   = _extract_letter(answer)
        return 1.0 if (pred_l and gt_l and pred_l == gt_l) else 0.0

    if task_type in BCQ_OE_TASKS:
        pred_yn = _extract_yesno(pred)
        gt_yn   = _extract_yesno(answer)
        return 1.0 if (pred_yn and gt_yn and pred_yn == gt_yn) else 0.0

    if task_type in MCQ_OE_TASKS:
        pred_l = _extract_letter(pred)
        gt_l   = _extract_letter(answer)
        return 1.0 if (pred_l and gt_l and pred_l == gt_l) else 0.0

    if task_type in TEMPORAL_TASKS:
        return _temporal_iou(pred, answer)

    if task_type in OPEN_TASKS:
        return _rouge_l(pred, answer)

    # Unknown task — give partial credit for non-empty responses
    return 0.5 if pred.strip() else 0.0


def reward_fn(completions: list[str], answers: list[str], task_types: list[str], **kwargs) -> list[float]:
    """
    TRL GRPOTrainer reward function signature.
    Extra dataset columns (answers, task_types) are passed as kwargs by TRL.
    """
    return [
        compute_reward(c, a, t)
        for c, a, t in zip(completions, answers, task_types)
    ]


# ── model ─────────────────────────────────────────────────────────────────────

def load_model_from_sft(sft_ckpt: str):
    """Load base model and apply the SFT LoRA adapter as the GRPO starting policy."""
    print(f"Loading base model: {MODEL_NAME}")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=MODEL_NAME,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    print(f"Applying SFT adapter: {sft_ckpt}")
    model = PeftModel.from_pretrained(model, sft_ckpt, is_trainable=True)
    print("SFT adapter loaded and set trainable for GRPO.")
    return model, tokenizer


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    data_dir   = Path(args.work_dir) / "data/track3"
    output_dir = Path(args.work_dir) / "checkpoints/track3_grpo" / args.run_name

    print(f"Rank     : {local_rank} / cuda:{local_rank}")
    print(f"Run      : {args.run_name}")
    print(f"SFT ckpt : {args.sft_ckpt}")
    print(f"Out dir  : {output_dir}")

    print("Loading dataset...")
    train_records = load_jsonl(str(data_dir / "train_all.jsonl"))
    train_ds = GRPODataset(train_records)
    print(f"  train: {len(train_ds):,} prompts")

    print("Loading model...")
    model, tokenizer = load_model_from_sft(args.sft_ckpt)

    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        run_name=args.run_name,

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        num_generations=args.num_generations,
        gradient_accumulation_steps=4,

        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",

        bf16=True,
        fp16=False,

        max_prompt_length=args.max_prompt_len,
        max_completion_length=args.max_completion_len,

        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,

        ddp_find_unused_parameters=False,
        remove_unused_columns=False,

        seed=3407,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=train_ds,
    )

    print("Starting GRPO training...")
    trainer.train(resume_from_checkpoint=args.resume or None)

    adapter_dir = output_dir / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"GRPO adapter saved → {adapter_dir}")


if __name__ == "__main__":
    main()
