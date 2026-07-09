"""
Phase 4 — Post-process raw inference outputs into a submission CSV.

Reads the JSONL written by inference.py and applies task-specific extraction
rules that mirror the official evaluate.py scoring code, then writes the
submission CSV required by the leaderboard.

Usage:
    python track3/postprocess.py \
        --input  $WORK/predictions/sft_v1_ckpt2000/raw_predictions.jsonl \
        --output $WORK/predictions/sft_v1_ckpt2000/submission.csv

    # Validate against the official evaluate.py (needs pandas + bert_score installed)
    python $WORK/data/track3/annotations/test/evaluate.py \
        --gt   $WORK/data/track3/annotations/test/test.json \
        --submission $WORK/predictions/sft_v1_ckpt2000/submission.csv
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path


# ── extraction helpers (mirror evaluate.py exactly so we know what scores) ──

def extract_yesno(text: str) -> str:
    """Return 'Yes' or 'No'. Falls back to 'Yes' if nothing found."""
    if not text or not text.strip():
        return "Yes"
    s = text.strip().lower()
    # Prefer the first word at start of string
    m = re.match(r"^(yes|no)\b", s)
    if m:
        return m.group(1).capitalize()
    # Search anywhere
    m = re.search(r"\b(yes|no)\b", s)
    if m:
        return m.group(1).capitalize()
    return "Yes"  # safe fallback; evaluator counts it wrong, not crash


def extract_letter(text: str) -> str:
    """Return 'A', 'B', 'C', or 'D'. Falls back to 'A' if nothing found."""
    if not text or not text.strip():
        return "A"
    s = text.strip()
    # "A)" or "(A)" or "A." or "A," at start
    m = re.match(r"^\(?([A-Da-d])\)?[).\s,:]", s)
    if m:
        return m.group(1).upper()
    # Bare single letter
    if re.fullmatch(r"[A-Da-d]", s):
        return s.upper()
    # Letter word-boundary anywhere
    m = re.search(r"\b([A-D])\b", s)
    if m:
        return m.group(1).upper()
    return "A"


def extract_temporal_json(text: str) -> str:
    """
    Return a JSON string like {"start": "MM:SS", "end": "MM:SS"}.
    Tries to parse from the model output; falls back to a dummy interval.
    """
    if not text:
        return '{"start": "00:00", "end": "00:01"}'

    # Strip <think> blocks first
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Try ```json ... ``` block
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            if "start" in obj and "end" in obj:
                return json.dumps({"start": str(obj["start"]), "end": str(obj["end"])})
        except json.JSONDecodeError:
            pass

    # Try bare JSON object
    m = re.search(r'\{[^{}]*"start"[^{}]*"end"[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if "start" in obj and "end" in obj:
                return json.dumps({"start": str(obj["start"]), "end": str(obj["end"])})
        except json.JSONDecodeError:
            pass

    # Try timestamps like "00:04 - 00:07" or "from 00:04 to 00:07"
    m = re.search(
        r"(\d{1,2}:\d{2}(?:\.\d+)?)\s*(?:to|–|-|→)\s*(\d{1,2}:\d{2}(?:\.\d+)?)",
        text,
    )
    if m:
        return json.dumps({"start": m.group(1), "end": m.group(2)})

    return '{"start": "00:00", "end": "00:01"}'  # fallback


def clean_openended(text: str) -> str:
    """Strip <think> block and trim whitespace for open-ended tasks."""
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return clean if clean else text.strip()


# ── task routing ──────────────────────────────────────────────────────────────

# Tasks scored by exact match on the structured part
BCQ_TASKS  = {"bcq"}
MCQ_TASKS  = {"mcq"}
# Tasks where we extract structured answer AND keep explanation
BCQ_OE_TASKS = {"bcq_openended"}
MCQ_OE_TASKS = {"mcq_openended"}
# Tasks scored by temporal IoU
TEMPORAL_TASKS = {"temporal_localization"}
# Tasks scored by BERTScore (free text, just clean and pass through)
OPEN_TASKS = {
    "open_qa", "scene_description", "video_summarization",
    "temporal_description", "causal_linkage",
}


def format_prediction(task_type: str, clean_prediction: str, raw_prediction: str) -> str:
    """
    Apply task-specific formatting to produce the submission prediction string.

    We use `clean_prediction` (with <think> stripped) for content, and
    `raw_prediction` only as a fallback.
    """
    text = clean_prediction if clean_prediction.strip() else raw_prediction

    if task_type in BCQ_TASKS:
        return extract_yesno(text)

    if task_type in MCQ_TASKS:
        return extract_letter(text)

    if task_type in BCQ_OE_TASKS:
        # Evaluator reads BERTScore on full text; keep the explanation but ensure
        # yes/no leads so the "bcq_openended" partial-match scoring also works.
        yn = extract_yesno(text)
        rest = text.strip()
        # If model already starts with Yes/No, return as-is
        if re.match(r"^(yes|no)\b", rest.lower()):
            return rest
        return f"{yn}. {rest}"

    if task_type in MCQ_OE_TASKS:
        letter = extract_letter(text)
        rest = text.strip()
        if re.match(r"^\(?[A-Da-d]\)?[).\s,:]", rest):
            return rest
        return f"{letter}) {rest}"

    if task_type in TEMPORAL_TASKS:
        return extract_temporal_json(text)

    if task_type in OPEN_TASKS:
        return text.strip()

    # Unknown task — pass through
    return text.strip()


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True,
                   help="JSONL from inference.py (raw_predictions.jsonl)")
    p.add_argument("--output", required=True,
                   help="Output CSV path (submission.csv)")
    p.add_argument("--test_json",
                   default="/leonardo_work/AIH4A_syrate/data/track3/annotations/test/test.json",
                   help="test.json — used to fill in any missing item_index with empty predictions")
    return p.parse_args()


def main():
    args = parse_args()

    # Load raw predictions
    raw: dict[str, dict] = {}
    with open(args.input) as f:
        for line in f:
            rec = json.loads(line)
            raw[rec["item_index"]] = rec
    print(f"Loaded {len(raw)} raw predictions from {args.input}")

    # Load full test item list (to catch any missing items)
    with open(args.test_json) as f:
        data = json.load(f)
    test_items = data.get("items", data) if isinstance(data, dict) else data

    # Build submission rows
    rows: list[tuple[str, str]] = []
    task_counts: dict[str, dict] = {}

    for item in test_items:
        idx = item["item_index"]
        task = item.get("task_type", "unknown")

        if idx not in raw:
            # Missing prediction — use a safe default per task type
            print(f"[WARN] No prediction for {idx} (task={task}) — using default")
            prediction = "Yes" if task in BCQ_TASKS else \
                         "A"   if task in MCQ_TASKS else \
                         '{"start": "00:00", "end": "00:01"}' if task in TEMPORAL_TASKS else \
                         "No anomalous events detected."
        elif raw[idx].get("skipped"):
            # Video was missing at inference time
            print(f"[WARN] Skipped item {idx} (task={task}) — using default")
            prediction = "Yes" if task in BCQ_TASKS else \
                         "A"   if task in MCQ_TASKS else \
                         '{"start": "00:00", "end": "00:01"}' if task in TEMPORAL_TASKS else \
                         "No anomalous events detected."
        else:
            rec = raw[idx]
            prediction = format_prediction(
                task,
                rec.get("clean_prediction", ""),
                rec.get("raw_prediction", ""),
            )

        rows.append((idx, prediction))

        # Track counts per task for summary
        tc = task_counts.setdefault(task, {"total": 0, "has_pred": 0})
        tc["total"] += 1
        if idx in raw and not raw[idx].get("skipped"):
            tc["has_pred"] += 1

    # Write CSV
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["item_index", "prediction"])
        writer.writerows(rows)

    print(f"\nSubmission CSV → {out_path}  ({len(rows)} rows)")
    print(f"\n{'Task':<28} {'predicted':>9} {'total':>6}")
    print("-" * 46)
    for task in sorted(task_counts):
        tc = task_counts[task]
        print(f"  {task:<26} {tc['has_pred']:>9} {tc['total']:>6}")

    print(f"\nValidate with:")
    test_json = args.test_json
    print(f"  python {Path(test_json).parent}/evaluate.py \\")
    print(f"    --gt {test_json} \\")
    print(f"    --submission {out_path}")


if __name__ == "__main__":
    main()
