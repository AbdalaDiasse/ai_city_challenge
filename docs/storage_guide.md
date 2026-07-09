# Storage Guide — Checkpoints & HuggingFace Cache

Everything large lives under `$WORK = /leonardo_work/AIH4A_syrate`.
Home directory (`~/`) has a tight quota — never store models, videos, or checkpoints there.

---

## 1. Training Checkpoints

### Where they land

```
$WORK/checkpoints/track3_sft/
├── sft_v1/          ← 4-bit QLoRA run (train_sft.py, r=16)
│   ├── checkpoint-1600/
│   ├── checkpoint-1800/
│   └── checkpoint-2000/   ← latest saved (step 2000/7251, epoch 0.83)
└── sft_bf16_v1/     ← BF16 run (train_sft_bf16.py, r=32) — empty until OOM is fixed
```

The run name (`sft_v1`, `sft_bf16_v1`) comes from the `RUN=` env var in the SLURM script.
`save_steps=200` means a checkpoint is written every 200 gradient steps.
`save_total_limit=3` means only the 3 most recent checkpoints are kept on disk.

---

### What is inside a checkpoint

```
checkpoint-2000/
│
│  ── INFERENCE (these two files are all you need to run predictions) ──
│
├── adapter_config.json          (4 KB)
│     LoRA configuration. Key fields:
│       - r: 16                  ← rank of LoRA matrices
│       - lora_alpha: 16         ← scaling factor (effective lr scale = alpha/r = 1.0)
│       - lora_dropout: 0
│       - base_model_name_or_path: "unsloth/qwen3-vl-8b-instruct-unsloth-bnb-4bit"
│       - target_modules: regex matching q_proj, k_proj, v_proj, o_proj,
│                         gate_proj, up_proj, down_proj in both vision
│                         encoder and language model layers
│       - peft_type: "LORA"
│       - task_type: "CAUSAL_LM"
│
├── adapter_model.safetensors    (196 MB)
│     The actual trained LoRA weights. Contains only the A and B delta
│     matrices for each targeted module — NOT the full 8B base model.
│     Formula: W_out = W_frozen + (B · A) * (alpha/r)
│     Loading this on top of the base model gives you the fine-tuned model.
│
│  ── TRAINING RESUME (needed only to continue training from this step) ──
│
├── optimizer.pt                 (101 MB)
│     Saved state of the 8-bit Adam optimizer. Contains momentum and
│     variance estimates for every LoRA parameter. Large because it stores
│     two floats per trainable parameter (m and v in Adam).
│     Without this you can still fine-tune further, but from a cold start
│     (optimizer resets, loss may spike briefly before re-converging).
│
├── scheduler.pt                 (4 KB)
│     Learning rate scheduler state — current step, last lr value.
│     Needed so the cosine decay resumes at the right point.
│
├── trainer_state.json           (40 KB)
│     Full log of training: every loss value, eval metric, learning rate,
│     grad norm at each logging step. Also tracks best_metric and
│     best_model_checkpoint so load_best_model_at_end works.
│     Useful for plotting loss curves.
│
├── training_args.bin            (8 KB)
│     Serialised SFTConfig (hyperparameters). Loaded by Trainer.resume_from_checkpoint
│     to verify the config matches.
│
├── rng_state_0.pth              (16 KB each, one per GPU rank)
├── rng_state_1.pth
├── rng_state_2.pth
└── rng_state_3.pth
│     Random number generator states for each of the 4 GPU processes.
│     Saved so that if training is resumed, the data shuffling and dropout
│     masks are reproducible from exactly this point.
│     Not needed for inference.
│
│  ── TOKENIZER (copied from base model, needed at inference time) ──
│
├── tokenizer.json               (11 MB)   ← vocabulary + BPE merge rules
├── tokenizer_config.json        (8 KB)    ← tokenizer class, special tokens
├── chat_template.jinja          (8 KB)    ← Jinja2 template for message formatting
└── processor_config.json        (4 KB)    ← vision processor settings (image/video)
```

### How to use a checkpoint

**For inference** (load base model + apply adapter):
```python
from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/qwen3-vl-8b-instruct-unsloth-bnb-4bit",
    load_in_4bit=True,
)
model = FastVisionModel.for_inference(model)
model.load_adapter("/leonardo_work/AIH4A_syrate/checkpoints/track3_sft/sft_v1/checkpoint-2000")
```

**For resuming training** (pass to SFTTrainer):
```python
# In train_sft.slurm, add --resume flag:
torchrun ... track3/train_sft.py --resume

# In train_sft.py, the trainer picks up from the latest checkpoint automatically:
trainer.train(resume_from_checkpoint=True)
# or point to a specific one:
trainer.train(resume_from_checkpoint=".../checkpoint-2000")
```

---

## 2. HuggingFace Cache

### Location
```
$WORK/hf_cache/hub/
├── models--unsloth--Qwen3-VL-8B-Instruct/          ← BF16 base model (~17 GB)
├── models--unsloth--qwen3-vl-8b-instruct-unsloth-bnb-4bit/  ← 4-bit model (~5 GB)
├── datasets--nvidia--PhysicalAI-Traffic-Anomaly-Reasoning/
├── datasets--cccccxy--so-tad/
├── datasets--Open-Space-Reasoning--AccidentBench/
└── datasets--wbfwonderful--Vad-R1/
```

The naming convention is `{type}--{org}--{repo}` with `/` replaced by `--`.

---

### Internal structure of a model entry

```
models--unsloth--Qwen3-VL-8B-Instruct/
│
├── blobs/                        ← actual file content, stored by SHA-256 hash
│   ├── 8be88fb5...               (4.6 GB)  model shard 1 of 4
│   ├── 83de00ea...               (4.7 GB)  model shard 2 of 4
│   ├── 0a88b98e...               (2.6 GB)  model shard 3 of 4
│   ├── d5d0aef0...               (4.6 GB)  model shard 4 of 4
│   ├── aeb13307...               (11 MB)   tokenizer.json
│   ├── 31349551...               (1.6 MB)  merges.txt
│   ├── 4783fe10...               (2.7 MB)  tokenizer vocab
│   ├── 1081bacf...               (8 KB)    config.json
│   ├── 52373fe2...               (4 KB)    generation_config.json
│   ├── ...                       (other small config files)
│   └── d5d0aef0....incomplete    ← PARTIAL DOWNLOAD — can be safely deleted
│                                    (a download was interrupted; HF will
│                                     re-download if needed)
│
├── refs/
│   └── main                      ← contains one line: "11d38e30f7b6dec7..."
│                                    This is the git commit hash of the HF repo
│                                    at the time it was downloaded. Maps branch
│                                    name → commit hash, just like git refs.
│
└── snapshots/
    └── 11d38e30f7b6dec7.../      ← one directory per commit hash
        ├── config.json           → symlink → ../../blobs/1081bacf...
        ├── model-00001-of-00004.safetensors  → symlink → ../../blobs/8be88fb5...
        ├── model-00002-of-00004.safetensors  → symlink → ../../blobs/83de00ea...
        ├── model-00003-of-00004.safetensors  → symlink → ../../blobs/0a88b98e...
        ├── model-00004-of-00004.safetensors  → symlink → ../../blobs/d5d0aef0...
        ├── tokenizer.json        → symlink → ../../blobs/aeb13307...
        └── ...                   ← ALL files are symlinks, no actual data here
```

### Why this design

**`blobs/`** — content-addressable storage (the same idea as git objects).
Each file is named by its SHA-256 hash. This means:
- Two different model versions that share the same tokenizer store it only once.
- A blob is immutable: if the hash matches, the content is guaranteed correct.
- You can safely deduplicate storage if the same file appears in multiple repos.

**`refs/`** — branch/tag pointers (same concept as `.git/refs/`).
`refs/main` = "the `main` branch of this repo currently points to commit `11d38e30`".
If you `hf download` again after an upstream update, HF will check if `refs/main` changed
and only download new/changed blobs.

**`snapshots/`** — one directory per git commit, containing only symlinks.
When your code calls `from_pretrained("unsloth/Qwen3-VL-8B-Instruct")`, HuggingFace:
1. Reads `refs/main` → gets commit hash `11d38e30...`
2. Opens `snapshots/11d38e30.../`
3. Follows the symlinks to the actual blobs
4. Loads the model weights

The symlink layer means you can have multiple versions of a model cached simultaneously
(e.g., commit A and commit B), each pointing to their own blobs, without duplicating files
that are identical between versions.

---

### The `.incomplete` file

```
blobs/d5d0aef0....04b52615.incomplete   (4.5 GB — nearly complete)
blobs/d5d0aef0...                       (4.6 GB — the complete version)
```

Both are present, which means:
- The complete blob was eventually downloaded successfully.
- The `.incomplete` file is a leftover from an interrupted download attempt.
- It is safe to delete: `rm blobs/d5d0aef0....incomplete`

---

### Useful commands

```bash
# How much space does the HF cache use?
du -sh $WORK/hf_cache/hub/

# See all cached models
ls $WORK/hf_cache/hub/

# Which commit of a model do you have?
cat $WORK/hf_cache/hub/models--unsloth--Qwen3-VL-8B-Instruct/refs/main

# Clean up an incomplete download
rm $WORK/hf_cache/hub/models--unsloth--Qwen3-VL-8B-Instruct/blobs/*.incomplete

# Check checkpoint training progress
python3 -c "
import json
ckpt = '$WORK/checkpoints/track3_sft/sft_v1/checkpoint-2000'
s = json.load(open(f'{ckpt}/trainer_state.json'))
print(f'Step {s[\"global_step\"]} / {s[\"max_steps\"]}  (epoch {s[\"epoch\"]:.2f})')
print(f'Best eval loss: {s[\"best_metric\"]}')
print(f'Last train loss: {s[\"log_history\"][-2][\"loss\"]}')
"
```
