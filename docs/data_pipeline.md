# Data Pipeline — Deep Explanation

Full walkthrough of how raw TAR data flows from HuggingFace to what the model actually sees during training and inference.

---

## Stage 0: Raw Data on Disk

After `download_data.sh` runs, two separate things exist on disk:

### A) 10 Annotation JSON Files (text only, small)

```
$WORK/data/track3/annotations/train/
    bcq.json                   # 7,340 items  — binary Yes/No
    bcq_openended.json         # 7,340 items  — binary Yes/No + explanation
    mcq.json                   # 3,670 items  — multiple choice A/B/C/D
    mcq_openended.json         # 3,670 items  — multiple choice + explanation
    open_qa.json               # 3,670 items  — open-ended QA
    scene_description.json     # 3,670 items  — describe what is in the scene
    video_summarization.json   # 3,670 items  — summarize the full video
    temporal_localization.json # 3,670 items  — when did the anomaly occur
    temporal_description.json  # 3,670 items  — what happened in a time interval
    causal_linkage.json        # 3,670 items  — what caused the anomaly
```

Each JSON uses the `tao-vl-reason-v1.0` format:

```json
{
  "format": "tao-vl-reason-v1.0",
  "metadata": { "..." : "..." },
  "items": [
    {
      "video_id": "TAD/01_Accident_001.mp4",
      "question": "Does the video show a vehicle collision?",
      "answer": "Yes",
      "reasoning": "At 00:12, two vehicles enter the intersection simultaneously. The red sedan fails to yield ...",
      "task_type": "bcq"
    }
  ]
}
```

The `reasoning` field is the human-written chain-of-thought. This is the most valuable part of the dataset — it is the primary training signal for CoT reasoning.

### B) ~150 GB of Raw Video Files

```
$WORK/data/track3/videos/
    TAD/            # mp4s stitched from JPG frames
    Accident-Bench/
    SO-TAD/         # extracted from 51-part PKWARE zip
    UCF_Crimes/
    Vad-R1/
    HTV/            # transcoded from .avi via imageio-ffmpeg
    barbados/
    TADBenchmark/
```

The `video_id` in each annotation (e.g. `"TAD/01_Accident_001.mp4"`) is a relative path inside this folder. Code joins it with `video_root` to get the absolute path on disk.

---

## Stage 1: `prepare_dataset.py` — Converting Annotations to JSONL

**No video frames are loaded at this stage.** This is pure text processing.

`build_conversation()` in `track3/data_utils.py:40–94` converts one raw annotation item:

### Input (one item from a task JSON):

```python
{
  "video_id": "TAD/01_Accident_001.mp4",
  "question": "Does the video show a vehicle collision?",
  "answer": "Yes",
  "reasoning": "At 00:12, two vehicles enter the intersection simultaneously ...",
  "task_type": "bcq"
}
```

### What the function does, step by step:

1. Builds absolute video path:
   `/leonardo_work/AIH4A_syrate/data/track3/videos/TAD/01_Accident_001.mp4`
2. Checks if the file exists on disk — returns `None` if missing (this is how the
   1,092 unavailable items are silently dropped; callers count misses per source).
3. Wraps the `reasoning` field in `<think>` tags:
   ```
   <think>
   At 00:12, two vehicles enter the intersection simultaneously ...
   </think>
   Yes
   ```
4. Returns a structured conversation dict.

### Output (one conversation record):

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert traffic surveillance analyst. Watch the provided video carefully, then reason step-by-step about any anomalous events before giving your final answer."
    },
    {
      "role": "user",
      "content": [
        {
          "type": "video",
          "video": "/leonardo_work/AIH4A_syrate/data/track3/videos/TAD/01_Accident_001.mp4",
          "max_pixels": 151200,
          "fps": 1.0
        },
        {
          "type": "text",
          "text": "Does the video show a vehicle collision?"
        }
      ]
    },
    {
      "role": "assistant",
      "content": "<think>\nAt 00:12, two vehicles enter the intersection simultaneously. The red sedan fails to yield ...\n</think>\nYes"
    }
  ],
  "_meta": {
    "video_id": "TAD/01_Accident_001.mp4",
    "task": "bcq"
  }
}
```

### Key observations:

- The **video is stored as a path + parameters** (`fps=1.0`, `max_pixels=151200`).
  No pixels exist in the JSONL — only the path on disk.
- `fps=1.0` is **hardcoded** at `data_utils.py:77`. This is where training fps is permanently fixed.
  Changing it requires re-running `prepare_dataset.py`.
- `max_pixels=151200` = 360 × 420. Each frame will be downscaled to fit within this pixel budget.
- The `_meta` field is for your analysis only — the model never sees it.
  `ConversationDataset` strips it: `{"messages": r["messages"]}`.
- The assistant turn contains reasoning AND answer in one string.
  This teaches the model to always reason before answering.

### What `stratified_split()` does:

Groups conversations by task type, shuffles each group independently, and takes 10% of each group as validation. This ensures all 10 task types appear in val, not just the most common ones.

### Output files:

```
$WORK/data/track3/train_all.jsonl   # 38,662 conversation records
$WORK/data/track3/val_all.jsonl     #  4,286 conversation records
$WORK/data/track3/dataset_stats.json
```

Each line of the JSONL is one conversation record. Total file size: ~2–4 GB (paths + text, zero pixels).

---

## Stage 2: Training — What the Model Actually Sees

This is where video frames first enter the pipeline. Frames are loaded **on-the-fly per batch** during training, not pre-cached.

### `ConversationDataset` (train_sft_bf16.py):

```python
self.data = [{"messages": r["messages"]} for r in records]
```

Strips `_meta`. Only `messages` reaches the trainer.

### `UnslothVisionDataCollator` — per-batch processing:

When the SFTTrainer requests a batch, the collator processes each conversation:

**Step 1 — Apply chat template**

The tokenizer's chat template converts the message list into a formatted string with Qwen3-VL special tokens:

```
<|im_start|>system
You are an expert traffic surveillance analyst...<|im_end|>
<|im_start|>user
<video>
Does the video show a vehicle collision?<|im_end|>
<|im_start|>assistant
<think>
At 00:12, two vehicles enter the intersection simultaneously...
</think>
Yes<|im_end|>
```

The `<video>` placeholder marks where visual tokens will be inserted.

**Step 2 — Load video frames from disk**

The collator reads `fps=1.0` and `max_pixels=151200` from the video dict in the message,
opens the `.mp4` using decord, samples frames uniformly at 1 fps, resizes each frame:

```
30-second clip  →  30 frames sampled  (indices spread uniformly)
10-second clip  →  10 frames sampled
Each frame      →  resized so width × height ≤ 151,200 px
                   (e.g., 360 × 420 is the maximum; shorter clips → fewer frames)
```

**Step 3 — Vision encoder**

Each frame passes through Qwen3-VL's vision encoder (ViT-style).
Each frame tile produces a fixed number of visual tokens — typically 64–256 tokens
depending on the resolution of that frame.

Example: 30 frames × ~100 tokens/frame = ~3,000 visual tokens for a 30-second clip.

**Step 4 — Full token sequence into the language model**

```
[system tokens]      "You are an expert traffic surveillance analyst..."
[visual tokens ×N]   [V₁][V₂]...[V₃₀₀₀]   ← one set per frame, N = frames × tokens/frame
[user text tokens]   "Does the video show a vehicle collision?"
[assistant tokens]   "<think>\nAt 00:12...\n</think>\nYes"
```

The visual tokens are inserted at the `<video>` placeholder position.
The sequence is padded/truncated to `max_seq_len=4096` tokens.

**Step 5 — Loss computation**

Loss is computed **only on assistant tokens**. All other tokens (system, video, user question)
are masked — their gradients are zero. The model is trained to predict:

```
<think>
At 00:12, two vehicles enter the intersection simultaneously. The red sedan fails to yield ...
</think>
Yes
```

Every token of the reasoning chain contributes to the loss equally.
This is why supervising on `reasoning` fields teaches the model to produce long,
coherent chain-of-thought before committing to a final answer.

---

## Stage 3: Inference — Same Flow, Different Entry Point

At inference time, `inference.py` performs the same video loading manually
(no collator — we need precise control per item).

```
test.json item
    │
    ├── _resolve_video_path()  →  locate .mp4 on disk
    ├── _load_video_frames()   →  decord at --fps 1.0, resize to max_pixels
    │
    ├── _build_messages_from_frames()  →  same message format as training,
    │                                      but frames pre-attached as _frames key
    │
    ├── tokenizer.apply_chat_template()  →  format string with <video> placeholder
    ├── tokenizer()  →  tokenize + insert visual tokens
    │
    └── model.generate()  →  produces assistant turn token by token:
                              "<think>...reasoning...</think>answer"
                                       │
                              strip_think()  →  extract answer only
                                       │
                              postprocess.py  →  format per task type
                                       │
                              submission.csv  →  item_index, prediction
```

### Frame cache (inference only):

Items are sorted by `video_id` before splitting into shards.
When consecutive items share the same clip, frames are loaded once and reused:

```python
if video_path != _cached_video_path:
    _cached_frames = _load_video_frames(video_path, fps, max_pixels)
    _cached_video_path = video_path
```

This avoids re-reading the same ~50 MB video file for every one of the 10–15 questions that can target the same clip.

### Token budget at inference:

| Task group | max_new_tokens | Why |
|---|---|---|
| `bcq`, `mcq` | 256 | Must close `</think>` then emit Yes/No or A–D |
| Everything else | 1024 | Full CoT + detailed free-form answer |

Using fewer than ~200 tokens for `bcq`/`mcq` causes the model to exhaust the budget inside `<think>`, producing no answer — BCQ score collapses to 0.5 (random guess).

---

## Complete End-to-End Diagram

```
HuggingFace Hub
      │
      ├── 10 × JSON annotation files  (text: question, answer, reasoning)
      └── ~150 GB mp4 video files
                │
                ▼
      prepare_dataset.py  (data_utils.build_conversation)
                │
                │  Per item:
                │  ① Check video file exists on disk
                │  ② Format: system + [video_path, fps=1.0, max_pixels=151200] + question
                │  ③ Wrap reasoning → <think>...</think> + answer
                │  ④ Store video_id and task in _meta (analysis only)
                │
                ▼
      train_all.jsonl  (38,662 lines, paths + text, ~3 GB)
      val_all.jsonl    ( 4,286 lines)
      [NO pixels on disk — only paths]
                │
                ▼
      SFTTrainer + UnslothVisionDataCollator
                │
                │  Per batch at training time:
                │  ① Strip _meta → only messages go to model
                │  ② Apply Qwen3-VL chat template
                │  ③ Load mp4 → sample at fps=1.0 → resize to ≤151200 px
                │  ④ Vision encoder → visual tokens
                │  ⑤ Assemble: [system][visual×N][question][assistant]
                │  ⑥ Loss only on assistant tokens (reasoning + answer)
                │
                ▼
      LoRA adapter checkpoint
                │
                ▼
      inference.py
                │
                │  Per test item:
                │  ① Load mp4 → sample at --fps → resize
                │  ② Frame cache: reuse if same clip as previous item
                │  ③ Apply chat template (no assistant turn — model generates it)
                │  ④ model.generate() → <think>CoT</think> + answer
                │  ⑤ strip_think() → extract answer text only
                │  ⑥ postprocess.py → format per task type
                │
                ▼
      submission.csv  (item_index, prediction)
```

---

## The fps Rule (Critical)

`fps=1.0` is hardcoded at `track3/data_utils.py:77` inside `build_conversation()`.

| Where | fps value | How to change |
|---|---|---|
| Training (JSONL) | **1.0** — baked in at `prepare_dataset.py` run time | Re-run `prepare_dataset.py` with different value in `data_utils.py:77` |
| Inference | **1.0** — runtime arg `--fps 1.0` in `inference.slurm` | Change `--fps` flag, no retraining needed |

Training and inference fps should be kept in sync for best results.
Using a higher fps at inference than training works as a quick experiment
but may shift the visual token distribution relative to what the model learned.
