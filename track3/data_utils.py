"""
Data utilities for Track 3 — Traffic Anomaly Reasoning.

Responsibilities:
  - Load the 10 per-task JSON annotation files
  - Convert each item to Unsloth/Qwen3-VL conversation format
  - Stratified train/val split (per task type)
  - Save as JSONL
"""

import json
import os
import random
from pathlib import Path
from typing import Optional

# Task file names as shipped by nvidia/PhysicalAI-Traffic-Anomaly-Reasoning
# Located under annotations/train/ (not annotations/ directly)
TASK_FILES = [
    "bcq.json",                  # binary Yes/No questions          (7,340 items)
    "bcq_openended.json",        # binary Yes/No + explanation       (7,340 items)
    "mcq.json",                  # multiple choice                   (3,670 items)
    "mcq_openended.json",        # multiple choice + explanation      (3,670 items)
    "open_qa.json",              # open-ended QA                     (3,670 items)
    "scene_description.json",    # scene description                 (3,670 items)
    "video_summarization.json",  # video summary                     (3,670 items)
    "temporal_localization.json",# when did the anomaly occur        (3,670 items)
    "temporal_description.json", # what happened in the interval     (3,670 items)
    "causal_linkage.json",       # what caused the anomaly           (3,670 items)
]

# System prompt applied to all tasks.
SYSTEM_PROMPT = (
    "You are an expert traffic surveillance analyst. "
    "Watch the provided video carefully, then reason step-by-step about any anomalous events "
    "before giving your final answer."
)


def build_conversation(item: dict, video_root: str, fps: float = 1.0) -> Optional[dict]:
    """
    Convert one TAR annotation into Unsloth's multi-modal conversation format.

    Returns None (silently) if the video file is missing — callers aggregate
    and report per-source miss counts rather than printing per-file warnings.
    """
    video_path = os.path.join(video_root, item["video_id"])
    if not os.path.exists(video_path):
        return None

    question = item["question"].strip()
    answer = item["answer"].strip()
    reasoning = item.get("reasoning", "").strip()

    # Wrap reasoning in <think> only when the field is non-empty
    if reasoning:
        assistant_text = f"<think>\n{reasoning}\n</think>\n{answer}"
    else:
        assistant_text = answer

    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        # Keep frame count low to manage memory; the model
                        # samples uniformly across the clip.
                        "max_pixels": 360 * 420,
                        "fps": fps,
                    },
                    {
                        "type": "text",
                        "text": question,
                    },
                ],
            },
            {
                "role": "assistant",
                "content": assistant_text,
            },
        ],
        # Metadata kept for validation/analysis — not fed to the model
        "_meta": {
            "video_id": item["video_id"],
            "task": item.get("task_type", "unknown"),
        },
    }


def load_task_file(json_path: str, video_root: str, task_name: str, fps: float = 1.0) -> list[dict]:
    """Load one per-task JSON and convert all items.

    Files use the tao-vl-reason-v1.0 wrapper:
      {"format": "...", "metadata": {...}, "media_root": null, "items": [...]}
    """
    with open(json_path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        items = data.get("items", [])
    else:
        items = data  # plain list fallback

    conversations = []
    miss_by_source: dict[str, int] = {}
    for item in items:
        item.setdefault("task_type", task_name)
        conv = build_conversation(item, video_root, fps=fps)
        if conv is not None:
            conversations.append(conv)
        else:
            src = item["video_id"].split("/")[0]
            miss_by_source[src] = miss_by_source.get(src, 0) + 1

    loaded = len(conversations)
    total = len(items)
    if miss_by_source:
        miss_summary = ", ".join(f"{s}:{n}" for s, n in sorted(miss_by_source.items()))
        print(f"  {task_name}: {loaded}/{total} loaded  (missing: {miss_summary})")
    else:
        print(f"  {task_name}: {loaded}/{total} loaded")
    return conversations


def load_all_tasks(annotation_dir: str, video_root: str, fps: float = 1.0) -> list[dict]:
    """Load and merge all 10 task files. Skips missing files with a warning."""
    all_conversations = []
    total_items = 0
    source_hits: dict[str, int] = {}
    source_miss: dict[str, int] = {}

    for fname in TASK_FILES:
        fpath = os.path.join(annotation_dir, fname)
        if not os.path.exists(fpath):
            print(f"[WARN] Task file not found, skipping: {fpath}")
            continue
        task_name = fname.replace(".json", "")
        convs = load_task_file(fpath, video_root, task_name, fps=fps)
        all_conversations.extend(convs)

        # Accumulate per-source stats from loaded conversations
        for c in convs:
            src = c["_meta"]["video_id"].split("/")[0]
            source_hits[src] = source_hits.get(src, 0) + 1

    total_items = sum(source_hits.values())

    print(f"\n{'Source':<32} {'loaded':>7}")
    print("-" * 42)
    for src in sorted(source_hits, key=lambda s: -source_hits[s]):
        print(f"  {src:<30} {source_hits[src]:>7}")
    print(f"\nTotal loaded: {len(all_conversations)}")
    pct = 100 * len(all_conversations) / 44040 if all_conversations else 0
    print(f"Coverage: {pct:.0f}% of 44,040 annotations (run postprocess_videos.py for missing sources)")
    return all_conversations


def stratified_split(
    conversations: list[dict],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Hold out val_ratio of each task type for local evaluation.
    This keeps metric estimates per-task comparable.
    """
    random.seed(seed)

    by_task: dict[str, list[dict]] = {}
    for conv in conversations:
        task = conv["_meta"]["task"]
        by_task.setdefault(task, []).append(conv)

    train, val = [], []
    for task, items in by_task.items():
        random.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
        print(f"  {task}: {len(items) - n_val} train / {n_val} val")

    random.shuffle(train)
    random.shuffle(val)
    print(f"\nSplit → train: {len(train)}, val: {len(val)}")
    return train, val


def save_jsonl(records: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} records → {path}")


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
