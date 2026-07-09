"""
Phase 2 — SFT fine-tuning of Qwen3-VL-8B on the Track 3 TAR dataset.
Full BF16 — no quantization, higher LoRA rank, accuracy-first for competition.

Usage (multi-GPU via torchrun):
    torchrun --nproc_per_node 4 track3/train_sft_bf16.py

Key env vars:
    WORK   — storage root (default: /leonardo_work/AIH4A_syrate)
    RUN    — experiment name (default: sft_bf16_v1)
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
MAX_SEQ_LEN   = 4096   # BF16 with 1 process per GPU has headroom on A100 (~63 GiB)
LORA_RANK     = 32     # 2× the 4-bit baseline (r=16); r=64 needs seq_len<=2048 if tight
LORA_ALPHA    = 32


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--work_dir",    default=WORK)
    p.add_argument("--run_name",    default=os.environ.get("RUN", "sft_bf16_v1"))
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--batch_size",  type=int,   default=1)
    p.add_argument("--grad_accum",  type=int,   default=4)
    p.add_argument("--max_seq_len", type=int,   default=MAX_SEQ_LEN)
    p.add_argument("--lora_r",      type=int,   default=LORA_RANK)
    p.add_argument("--resume",      action="store_true")
    p.add_argument("--train_jsonl", default=None,
                   help="Override train JSONL path (default: <work_dir>/data/track3/train_all.jsonl)")
    p.add_argument("--val_jsonl",   default=None,
                   help="Override val JSONL path   (default: <work_dir>/data/track3/val_all.jsonl)")
    return p.parse_args()


# ── data ────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class ConversationDataset(TorchDataset):
    """Wraps JSONL records as a torch Dataset, bypassing PyArrow schema inference.

    PyArrow's Dataset.from_list fails when the 'content' field is sometimes a
    string (system/assistant turns) and sometimes a list (user turns with video).
    """
    def __init__(self, records: list[dict]):
        self.data = [{"messages": r["messages"]} for r in records]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ── model ────────────────────────────────────────────────────────────────────

def load_model(args, local_rank: int):
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=args.max_seq_len,
        load_in_4bit=False,          # full BF16 — no quantization
        use_gradient_checkpointing="unsloth",
        device_map={"": local_rank},  # pin this rank's copy to its own GPU
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

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    data_dir   = Path(args.work_dir) / "data/track3"
    output_dir = Path(args.work_dir) / "checkpoints/track3_sft" / args.run_name

    print(f"Rank     : {local_rank} / cuda:{local_rank}  visible_gpus={torch.cuda.device_count()}")
    print(f"Run      : {args.run_name}")
    print(f"Data dir : {data_dir}")
    print(f"Out dir  : {output_dir}")
    print(f"Model    : {MODEL_NAME}  (BF16, no quantization)")
    print(f"LoRA     : r={args.lora_r}  alpha={LORA_ALPHA}")
    print(f"Epochs   : {args.epochs}  LR: {args.lr}  BS: {args.batch_size}  AccumSteps: {args.grad_accum}")
    print()

    train_jsonl = args.train_jsonl or str(data_dir / "train_all.jsonl")
    val_jsonl   = args.val_jsonl   or str(data_dir / "val_all.jsonl")

    print("Loading datasets...")
    print(f"  train JSONL: {train_jsonl}")
    print(f"  val   JSONL: {val_jsonl}")
    train_records = load_jsonl(train_jsonl)
    val_records   = load_jsonl(val_jsonl)
    train_ds = ConversationDataset(train_records)
    val_ds   = ConversationDataset(val_records)
    print(f"  train: {len(train_ds):,}  val: {len(val_ds):,}")

    print("Loading model...")
    model, tokenizer = load_model(args, local_rank)

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

            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,

            learning_rate=args.lr,
            weight_decay=0.01,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",      # 8-bit Adam saves optimizer memory even in BF16

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
            dataset_num_proc=1,

            seed=3407,
            report_to="none",
            remove_unused_columns=False,
        ),
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume or None)

    adapter_dir = output_dir / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"LoRA adapter saved → {adapter_dir}")


if __name__ == "__main__":
    main()
