"""
Phase 4 — Inference on TAR test set using a fine-tuned LoRA checkpoint.

Loads base model + LoRA adapter, runs generation on all items in test.json,
and saves raw predictions to a JSONL file for postprocess.py to format.

Usage (single GPU):
    python track3/inference.py \
        --checkpoint $WORK/checkpoints/track3_sft/sft_v1/checkpoint-2000 \
        --test_json  $WORK/data/track3/annotations/test/test.json \
        --video_root $WORK/data/track3/videos \
        --output_dir $WORK/predictions/sft_v1_ckpt2000

Key env vars:
    WORK   — storage root (default: /leonardo_work/AIH4A_syrate)
    CUDA_VISIBLE_DEVICES — which GPU to use (default: GPU 0)
"""

import unsloth  # must be first

import argparse
import json
import os
import re
import time
from pathlib import Path

import decord
import numpy as np
import torch
from PIL import Image
from peft import PeftModel
from unsloth import FastVisionModel

WORK = os.environ.get("WORK", "/leonardo_work/AIH4A_syrate")

BASE_MODEL_4BIT = "unsloth/qwen3-vl-8b-instruct-unsloth-bnb-4bit"
BASE_MODEL_BF16 = "unsloth/Qwen3-VL-8B-Instruct"

SYSTEM_PROMPT = (
    "You are an expert traffic surveillance analyst. "
    "Watch the provided video carefully, then reason step-by-step about any anomalous events "
    "before giving your final answer."
)

# Tasks that require <200 tokens; open-ended tasks need more room.
SHORT_TASKS = {"bcq", "mcq"}
TEMPORAL_TASKS = {"temporal_localization"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   required=True,
                   help="Path to SFTTrainer checkpoint dir (has adapter_config.json)")
    p.add_argument("--test_json",    default=f"{WORK}/data/track3/annotations/test/test.json")
    p.add_argument("--video_root",   default=f"{WORK}/data/track3/videos/tar_test")
    p.add_argument("--output_dir",   default=f"{WORK}/predictions/run")
    p.add_argument("--load_in_4bit", action="store_true", default=False,
                   help="Load base model in 4-bit (pass flag for 4-bit; omit for BF16)")
    p.add_argument("--max_new_tokens_short",  type=int, default=256,
                   help="Max tokens for bcq/mcq (enough to close </think> then output Yes/No or A-D)")
    p.add_argument("--max_new_tokens_long",   type=int, default=1024,
                   help="Max tokens for open-ended and temporal tasks")
    p.add_argument("--fps",          type=float, default=1.0,
                   help="Frames per second to sample from each video")
    p.add_argument("--max_pixels",   type=int, default=360 * 420,
                   help="Max pixels per frame (controls memory vs quality)")
    p.add_argument("--resume",       action="store_true",
                   help="Skip items already in output JSONL (allows resuming a run)")
    p.add_argument("--num_shards",   type=int, default=1,
                   help="Total number of parallel GPU workers")
    p.add_argument("--shard_id",     type=int, default=0,
                   help="Index of this worker (0-indexed)")
    return p.parse_args()


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks. The answer follows the block.

    Two-pass approach handles both complete and truncated (unclosed) think blocks:
      pass 1 removes complete blocks,
      pass 2 removes any remaining unclosed <think> to end-of-string.
    """
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    clean = re.sub(r"<think>.*", "", clean, flags=re.DOTALL).strip()
    return clean


def load_model(checkpoint: str, load_in_4bit: bool):
    base = BASE_MODEL_4BIT if load_in_4bit else BASE_MODEL_BF16
    print(f"Loading base model: {base}")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=base,
        load_in_4bit=load_in_4bit,
        use_gradient_checkpointing=False,
    )
    print(f"Applying LoRA adapter: {checkpoint}")
    model = PeftModel.from_pretrained(model, checkpoint, is_trainable=False)
    FastVisionModel.for_inference(model)
    model.eval()
    print("Model ready.")
    return model, tokenizer


def _resolve_video_path(video_root: str, video_id: str) -> str | None:
    """Try multiple path combinations to locate the video file.

    Handles the case where video_root already includes a subdirectory that
    video_id also starts with (e.g. video_root='.../videos/tar_test' and
    video_id='tar_test/clip.mp4').
    """
    # Direct join (correct when video_root = .../videos and video_id = tar_test/clip.mp4)
    full = os.path.join(video_root, video_id)
    if os.path.exists(full):
        return full
    # Strip leading directory from video_id (handles double-prefix case)
    basename_only = os.path.join(video_root, os.path.basename(video_id))
    if os.path.exists(basename_only):
        return basename_only
    return None


def _load_video_frames(video_path: str, fps: float, max_pixels: int) -> list:
    """Sample frames from a video file using decord, returning PIL Images."""
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    total = len(vr)
    native_fps = vr.get_avg_fps() or 1.0
    num_frames = max(1, int(total / native_fps * fps))
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames_np = vr.get_batch(indices).asnumpy()  # (T, H, W, 3)

    out = []
    for frame in frames_np:
        img = Image.fromarray(frame)
        w, h = img.size
        if w * h > max_pixels:
            scale = (max_pixels / (w * h)) ** 0.5
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        out.append(img)
    return out


def build_messages(item: dict, video_root: str, fps: float, max_pixels: int) -> list | None:
    """Build messages with video path (used when frame cache is not available)."""
    video_path = _resolve_video_path(video_root, item["video_id"])
    if video_path is None:
        return None
    frames = _load_video_frames(video_path, fps, max_pixels)
    return _build_messages_from_frames(item, video_path, frames, fps, max_pixels)


def _build_messages_from_frames(item: dict, video_path: str, frames: list, fps: float, max_pixels: int) -> list:
    """Build messages with pre-loaded frames attached (avoids re-reading the video)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "_frames": frames,   # pre-loaded; picked up by generate_prediction
                    "fps": fps,
                    "max_pixels": max_pixels,
                },
                {"type": "text", "text": item["question"].strip()},
            ],
        },
    ]


@torch.no_grad()
def generate_prediction(model, tokenizer, messages: list, max_new_tokens: int) -> str:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True ,enable_thinking=True ,
    )

    # Use pre-loaded frames if present (_frames key), otherwise load from path.
    video_frames_list = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if item.get("type") == "video":
                if "_frames" in item:
                    video_frames_list.append(item["_frames"])
                else:
                    frames = _load_video_frames(
                        item["video"],
                        fps=item.get("fps", 1.0),
                        max_pixels=item.get("max_pixels", 360 * 420),
                    )
                    video_frames_list.append(frames)

    inputs = tokenizer(
        text=[text],
        images=None,
        videos=video_frames_list if video_frames_list else None,
        return_tensors="pt",
    ).to("cuda:0")

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        use_cache=True,
    )
    new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens[0], skip_special_tokens=True)


def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Each shard writes its own file; single-GPU runs use the plain name.
    suffix = f"_shard{args.shard_id}" if args.num_shards > 1 else ""
    raw_jsonl = out_dir / f"raw_predictions{suffix}.jsonl"
    log_path  = out_dir / f"inference{suffix}.log"

    # Load existing results if resuming
    done: set[str] = set()
    if args.resume and raw_jsonl.exists():
        with open(raw_jsonl) as f:
            for line in f:
                rec = json.loads(line)
                done.add(rec["item_index"])
        print(f"Resuming: {len(done)} items already done.")

    # Load test items
    with open(args.test_json) as f:
        data = json.load(f)
    items = data.get("items", data) if isinstance(data, dict) else data

    # Sort by video_id so all questions for the same video are consecutive
    # (keeps the frame cache effective within each shard).
    items = sorted(items, key=lambda x: x["video_id"])

    # Slice this shard's contiguous chunk (keeps full videos together)
    if args.num_shards > 1:
        shard_size = (len(items) + args.num_shards - 1) // args.num_shards
        start = args.shard_id * shard_size
        items = items[start: start + shard_size]
        print(f"Shard {args.shard_id}/{args.num_shards}: {len(items)} items  |  video_root: {args.video_root}", flush=True)
    else:
        print(f"Test items: {len(items)}  |  video_root: {args.video_root}")

    model, tokenizer = load_model(args.checkpoint, args.load_in_4bit)

    n_done = len(done)
    n_skipped = 0
    t0 = time.time()
    _cached_video_path: str | None = None
    _cached_frames: list | None = None

    with open(raw_jsonl, "a") as fout, open(log_path, "a") as flog:
        for i, item in enumerate(items):
            idx = item["item_index"]
            if idx in done:
                continue

            task = item.get("task_type", "unknown")
            max_tok = (
                args.max_new_tokens_short if task in SHORT_TASKS
                else args.max_new_tokens_long
            )

            video_path = _resolve_video_path(args.video_root, item["video_id"])
            if video_path is None:
                messages = None
            else:
                # Reuse cached frames if same video as previous item
                if video_path != _cached_video_path:
                    _cached_frames = _load_video_frames(video_path, args.fps, args.max_pixels)
                    _cached_video_path = video_path
                    print(f"[VIDEO] loaded {os.path.basename(video_path)}  ({len(_cached_frames)} frames)", flush=True)
                messages = _build_messages_from_frames(item, video_path, _cached_frames, args.fps, args.max_pixels)
            if messages is None:
                n_skipped += 1
                msg = f"[SKIP] {idx}  video not found: {item['video_id']}"
                print(msg)
                flog.write(msg + "\n")
                # Write a placeholder so postprocess doesn't silently drop it
                rec = {
                    "item_index": idx,
                    "task_type": task,
                    "video_id": item["video_id"],
                    "raw_prediction": "",
                    "skipped": True,
                }
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                continue

            try:
                raw = generate_prediction(model, tokenizer, messages, max_tok)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raw = ""
                msg = f"[OOM]  {idx}  task={task}"
                print(msg)
                flog.write(msg + "\n")

            clean = strip_think(raw)
            n_done += 1

            rec = {
                "item_index": idx,
                "task_type": task,
                "video_id": item["video_id"],
                "raw_prediction": raw,       # full output including <think>
                "clean_prediction": clean,   # with <think> stripped
            }
            fout.write(json.dumps(rec) + "\n")
            fout.flush()

            elapsed = time.time() - t0
            rate = n_done / elapsed
            remaining = (len(items) - n_done - n_skipped) / rate if rate > 0 else 0
            print(
                f"[{n_done:>4}/{len(items)}] {task:<24} eta={remaining/60:.0f}min  "
                f"{idx[:12]}  {clean[:80]!r}"
            )

    print(f"\nDone. {n_done} predicted, {n_skipped} skipped (missing video).")
    print(f"Raw predictions → {raw_jsonl}")
    print(f"Next: python track3/postprocess.py --input {raw_jsonl} --output {out_dir}/submission.csv")


if __name__ == "__main__":
    main()
