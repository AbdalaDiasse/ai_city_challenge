"""
Phase 2 (Thinking variant) — SFT fine-tuning of Qwen3-VL-8B-Thinking on the TAR dataset.

Key difference from train_sft_bf16.py:
  - Base model is the Thinking variant (pre-trained for chain-of-thought reasoning).
  - System prompt omits the "reason step-by-step" instruction — the model already
    does this natively via its built-in <think> behaviour.
  - We train on assistant turns that include <think>...</think> blocks, reinforcing
    the model's native reasoning style on domain-specific traffic anomaly content.

Usage (multi-GPU via torchrun):
    torchrun --nproc_per_node 4 track3/train_sft_thinking.py

Key env vars:
    WORK   — storage root (default: /leonardo_work/AIH4A_syrate)
    RUN    — experiment name (default: sft_thinking_v1)
"""

import unsloth  # must be first — patches transformers/trl/peft before they load

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset as TorchDataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator


# ── defaults ─────────────────────────────────────────────────────────────────

WORK        = os.environ.get("WORK", "/leonardo_work/AIH4A_syrate")
MODEL_NAME  = "unsloth/Qwen3-VL-8B-Thinking"  # BF16 Thinking variant
MAX_SEQ_LEN = 4096
LORA_RANK   = 32
LORA_ALPHA  = 32

# The Thinking model reasons natively — no need to instruct it to think step-by-step.
SYSTEM_PROMPT = (
    "You are an expert traffic surveillance analyst. "
    "Watch the provided video carefully and answer questions about anomalous events."
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--work_dir",    default=WORK)
    p.add_argument("--run_name",    default=os.environ.get("RUN", "sft_thinking_v1"))
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--lr",          type=float, default=5e-5,
                   help="Lower LR than Instruct SFT — Thinking model needs gentler updates")
    p.add_argument("--batch_size",  type=int,   default=1)
    p.add_argument("--grad_accum",  type=int,   default=4)
    p.add_argument("--max_seq_len", type=int,   default=MAX_SEQ_LEN)
    p.add_argument("--lora_r",      type=int,   default=LORA_RANK)
    p.add_argument("--resume",      action="store_true")
    return p.parse_args()


# ── data ─────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _replace_system_prompt(messages: list[dict]) -> list[dict]:
    """Swap out the Instruct system prompt for the Thinking-model system prompt."""
    out = []
    for m in messages:
        if m.get("role") == "system":
            out.append({"role": "system", "content": SYSTEM_PROMPT})
        else:
            out.append(m)
    return out


class ConversationDataset(TorchDataset):
    """Wraps JSONL records as a torch Dataset, bypassing PyArrow schema inference.

    The <think>...</think> blocks in assistant turns are kept intact — they are
    the training signal that teaches the Thinking model domain-specific reasoning.
    """
    def __init__(self, records: list[dict]):
        self.data = [
            {"messages": _replace_system_prompt(r["messages"])}
            for r in records
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ── model ─────────────────────────────────────────────────────────────────────

def load_model(args, local_rank: int):
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=args.max_seq_len,
        load_in_4bit=False,                  # full BF16
        use_gradient_checkpointing="unsloth",
        device_map={"": local_rank},
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        target_modules="all-linear",
        modules_to_save=["lm_head", "embed_tokens"],
    )
    return model, tokenizer


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    data_dir   = Path(args.work_dir) / "data/track3"
    output_dir = Path(args.work_dir) / "checkpoints/track3_sft" / args.run_name

    print(f"Rank     : {local_rank} / cuda:{local_rank}  visible_gpus={torch.cuda.device_count()}")
    print(f"Run      : {args.run_name}")
    print(f"Model    : {MODEL_NAME}  (BF16 Thinking)")
    print(f"LoRA     : r={args.lora_r}  alpha={LORA_ALPHA}")
    print(f"Epochs   : {args.epochs}  LR: {args.lr}  BS: {args.batch_size}  Accum: {args.grad_accum}")
    print(f"Out dir  : {output_dir}")
    print()

    print("Loading datasets...")
    train_records = load_jsonl(str(data_dir / "train_all.jsonl"))
    val_records   = load_jsonl(str(data_dir / "val_all.jsonl"))
    train_ds = ConversationDataset(train_records)
    val_ds   = ConversationDataset(val_records)
    print(f"  train: {len(train_ds):,}  val: {len(val_ds):,}")

    print("Loading model...")
    model, tokenizer = load_model(args, local_rank)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(
            output_dir=str(output_dir),
            run_name=args.run_name,

            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,

            learning_rate=args.lr,
            weight_decay=0.01,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",

            fp16=False,
            bf16=True,

            logging_steps=10,
            eval_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",

            ddp_find_unused_parameters=False,
            dataset_num_proc=1,   # required for VLMs — crashes with >1

            seed=3407,
            report_to="none",
            remove_unused_columns=False,
        ),
    )

    print("Starting SFT training (Thinking model)...")
    trainer.train(resume_from_checkpoint=args.resume or None)

    adapter_dir = output_dir / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"LoRA adapter saved → {adapter_dir}")


if __name__ == "__main__":
    main()
