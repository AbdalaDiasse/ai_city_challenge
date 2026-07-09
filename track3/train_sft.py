"""
Phase 2 — SFT fine-tuning of Qwen3-VL-8B on the Track 3 TAR dataset.

Usage (single GPU):
    python track3/train_sft.py

Usage (multi-GPU via torchrun — preferred):
    torchrun --nproc_per_node 4 track3/train_sft.py

Key env vars:
    WORK   — storage root (default: /leonardo_work/AIH4A_syrate)
    RUN    — experiment name (default: sft_v1)
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


# ── defaults ────────────────────────────────────────────────────────────────

WORK          = os.environ.get("WORK", "/leonardo_work/AIH4A_syrate")
MODEL_NAME    = "unsloth/Qwen3-VL-8B-Instruct"
MAX_SEQ_LEN   = 2048   # video + CoT fits comfortably; raise to 4096 if truncation appears
LORA_RANK     = 16
LORA_ALPHA    = 16


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--work_dir",    default=WORK)
    p.add_argument("--run_name",    default=os.environ.get("RUN", "sft_v1"))
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--batch_size",  type=int,   default=1,
                   help="Per-device train batch size (video inputs are large)")
    p.add_argument("--grad_accum",  type=int,   default=4,
                   help="Gradient accumulation steps")
    p.add_argument("--max_seq_len", type=int,   default=MAX_SEQ_LEN)
    p.add_argument("--lora_r",      type=int,   default=LORA_RANK)
    p.add_argument("--resume",      action="store_true",
                   help="Resume from latest checkpoint in output_dir")
    return p.parse_args()


# ── data ────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class ConversationDataset(TorchDataset):
    """Wraps JSONL records as a torch Dataset, bypassing PyArrow schema inference.

    PyArrow's Dataset.from_list fails when the 'content' field is sometimes a
    string (system/assistant turns) and sometimes a list (user turns with video).
    Using a plain torch Dataset avoids that entirely.
    """
    def __init__(self, records: list[dict]):
        self.data = [{"messages": r["messages"]} for r in records]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ── model ────────────────────────────────────────────────────────────────────

def load_model(args):
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
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
    )
    return model, tokenizer


# ── main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    data_dir   = Path(args.work_dir) / "data/track3"
    output_dir = Path(args.work_dir) / "checkpoints/track3_sft" / args.run_name

    print(f"Run      : {args.run_name}")
    print(f"Data dir : {data_dir}")
    print(f"Out dir  : {output_dir}")
    print(f"Model    : {MODEL_NAME}")
    print(f"Epochs   : {args.epochs}  LR: {args.lr}  BS: {args.batch_size}  AccumSteps: {args.grad_accum}")
    print()

    # ── load data ────────────────────────────────────────────────────────────
    print("Loading datasets...")
    train_records = load_jsonl(str(data_dir / "train_all.jsonl"))
    val_records   = load_jsonl(str(data_dir / "val_all.jsonl"))
    train_ds = ConversationDataset(train_records)
    val_ds   = ConversationDataset(val_records)
    print(f"  train: {len(train_ds):,}  val: {len(val_ds):,}")

    # ── model + LoRA ──────────────────────────────────────────────────────────
    print("Loading model...")
    model, tokenizer = load_model(args)

    # ── trainer ───────────────────────────────────────────────────────────────
    effective_batch = args.batch_size * args.grad_accum
    print(f"Effective batch size per GPU: {effective_batch}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(
            output_dir=str(output_dir),
            run_name=args.run_name,

            # epochs / steps
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,

            # optimizer
            learning_rate=args.lr,
            weight_decay=0.01,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",

            # precision
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),

            # logging & checkpointing
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",

            # multi-GPU: must be False to avoid unused-param errors with LoRA
            ddp_find_unused_parameters=False,

            # MANDATORY for video datasets — avoids TRL multiprocessing crash
            dataset_num_proc=1,

            # misc
            seed=3407,
            report_to="none",   # swap to "wandb" if you set WANDB_API_KEY
            remove_unused_columns=False,
        ),
    )

    # ── train ─────────────────────────────────────────────────────────────────
    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume or None)

    # ── save LoRA adapter ─────────────────────────────────────────────────────
    adapter_dir = output_dir / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"LoRA adapter saved → {adapter_dir}")


if __name__ == "__main__":
    main()
