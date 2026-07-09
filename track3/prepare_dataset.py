"""
Phase 1 — Dataset preparation for Track 3.

Reads the 10 task JSON files, converts to Unsloth conversation format,
performs a stratified train/val split, and writes JSONL files to $WORK.

Usage:
    # Default (fps=1.0) — writes train_all.jsonl / val_all.jsonl
    python track3/prepare_dataset.py

    # Higher fps — writes train_fps2.jsonl / val_fps2.jsonl, never touches existing files
    python track3/prepare_dataset.py --fps 2.0

Expects:
    $WORK/data/track3/annotations/  — downloaded HF dataset
    $WORK/data/track3/videos/       — downloaded video files (~150 GB)

Produces:
    $WORK/data/track3/train_<suffix>.jsonl
    $WORK/data/track3/val_<suffix>.jsonl
    $WORK/data/track3/dataset_stats_<suffix>.json
  where suffix = "all" for fps=1.0, "fps<N>" otherwise (e.g. "fps2" for 2.0).
"""

import argparse
import json
import os
import time
from pathlib import Path

from data_utils import load_all_tasks, stratified_split, save_jsonl


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--work_dir", default=os.environ.get("WORK", "/leonardo_work/AIH4A_syrate"))
    p.add_argument("--fps",      type=float, default=1.0,
                   help="Frames per second baked into training JSONL (default: 1.0)")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed",     type=int,   default=42)
    return p.parse_args()


def fps_suffix(fps: float) -> str:
    """'all' for 1.0 (legacy name), 'fps<N>' otherwise."""
    if fps == 1.0:
        return "all"
    # Format: fps2, fps0p5, fps1p5, etc.
    label = f"{fps:g}".replace(".", "p")
    return f"fps{label}"


def main():
    args = parse_args()

    annotation_dir = os.path.join(args.work_dir, "data/track3/annotations/train")
    video_root     = os.path.join(args.work_dir, "data/track3/videos")
    out_dir        = os.path.join(args.work_dir, "data/track3")
    suffix         = fps_suffix(args.fps)

    train_path = os.path.join(out_dir, f"train_{suffix}.jsonl")
    val_path   = os.path.join(out_dir, f"val_{suffix}.jsonl")
    stats_path = os.path.join(out_dir, f"dataset_stats_{suffix}.json")

    print("=" * 60)
    print("Track 3 — Dataset Preparation")
    print(f"  annotations : {annotation_dir}")
    print(f"  video root  : {video_root}")
    print(f"  fps         : {args.fps}")
    print(f"  output dir  : {out_dir}")
    print(f"  train JSONL : {train_path}")
    print(f"  val JSONL   : {val_path}")
    print(f"  val ratio   : {args.val_ratio}")
    print("=" * 60)

    if not os.path.isdir(annotation_dir):
        raise FileNotFoundError(
            f"Annotation directory not found: {annotation_dir}\n"
            "Run track3/download_data.sh first. "
            "Expected structure: $WORK/data/track3/annotations/train/*.json"
        )
    if not os.path.isdir(video_root):
        raise FileNotFoundError(
            f"Video directory not found: {video_root}\n"
            "Run track3/download_data.sh first."
        )

    t0 = time.time()

    print("\n[1/3] Loading task files...")
    conversations = load_all_tasks(annotation_dir, video_root, fps=args.fps)

    print("\n[2/3] Stratified split...")
    train_data, val_data = stratified_split(conversations, args.val_ratio, args.seed)

    print("\n[3/3] Saving JSONL files...")
    save_jsonl(train_data, train_path)
    save_jsonl(val_data,   val_path)

    stats = {
        "total": len(conversations),
        "train": len(train_data),
        "val": len(val_data),
        "fps": args.fps,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "elapsed_s": round(time.time() - t0, 1),
        "task_counts": {},
    }
    for conv in conversations:
        task = conv["_meta"]["task"]
        stats["task_counts"][task] = stats["task_counts"].get(task, 0) + 1

    Path(stats_path).write_text(json.dumps(stats, indent=2))

    print(f"\nDone in {stats['elapsed_s']}s")
    print(f"Stats → {stats_path}")
    print("\nTask breakdown:")
    for task, count in sorted(stats["task_counts"].items()):
        print(f"  {task:40s} {count:6d}")


if __name__ == "__main__":
    main()
