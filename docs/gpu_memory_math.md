# GPU Memory Math — Qwen3-VL-8B Fine-tuning

How to estimate per-GPU memory for any `max_seq_len`, precision, and LoRA rank.
Applies to: `train_sft_bf16.py`, `train_sft_bf16_fps2.slurm`, and any future SFT/GRPO runs on Leonardo A100 80 GiB.

---

## Architecture Constants (Qwen3-VL-8B)

```
D        = 3584   hidden dimension
H        = 28     attention heads
H_kv     = 4      KV heads  (GQA — grouped query attention)
d_head   = D / H  = 3584 / 28 = 128
L        = 28     transformer layers
FFN      = 18944  intermediate size (SwiGLU gate + up projections)
P        = 7.6 B  total parameters
```

---

## Formula for Each Memory Term

### 1 — Model Weights

```
bytes = P × bytes_per_param

BF16:  7.6×10⁹ × 2   = 15.2 GB  ≈ 14.2 GiB
4-bit: 7.6×10⁹ × 0.5 =  3.8 GB  ≈  3.5 GiB
```

4-bit packs 2 parameters per byte (`bytes_per_param = 0.5`).

---

### 2 — KV Cache

One K vector and one V vector stored per layer, per KV head, per token:

```
KV bytes = 2 × L × H_kv × d_head × S × B × 2

  2      = K and V
  L      = 28 layers
  H_kv   = 4 KV heads
  d_head = 128
  S      = sequence length
  B      = batch size (1 during training)
  2      = bytes per BF16 value

Example — S=4096, B=1:
  = 2 × 28 × 4 × 128 × 4096 × 1 × 2
  = 234,881,024 bytes ≈ 0.22 GiB

S=8192  → 0.44 GiB  (2×)
S=16384 → 0.88 GiB  (4×)
```

KV cache scales **linearly** with sequence length and is small relative to weights.

---

### 3 — Activations (gradient checkpointing)

Without checkpointing, all layer outputs are stored for the backward pass.
With `use_gradient_checkpointing="unsloth"`, only `√L ≈ 5` checkpoints are kept
and the rest are recomputed. The dominant activation per layer:

```
per-layer bytes = B × S × (D + 2×FFN) × 2
               = 1 × S × (3584 + 2×18944) × 2
               = S × 82944

With √28 ≈ 5 stored layers:
act bytes = 5 × S × 82944

S=2048:  5 × 2048 × 82944 ≈ 0.77 GiB
S=4096:  5 × 4096 × 82944 ≈ 1.54 GiB   (≈1.9 GiB with overhead)
S=8192:  doubles           ≈ 3.8 GiB
S=16384: 4×                ≈ 7.6 GiB
```

Activations scale **linearly** with S (flash attention avoids the O(S²) memory term).

---

### 4 — LoRA Parameters

LoRA adds two low-rank matrices per target module: `A ∈ ℝ^{D×r}` and `B ∈ ℝ^{r×D}`.
For our config (`r=32`, targeting Q/K/V/O + gate/up/down FFN):

```
D_kv = H_kv × d_head = 4 × 128 = 512

Per layer:
  Q:          D×r + r×D           = 3584×32 + 32×3584   = 229,376
  K:          D×r + r×D_kv        = 3584×32 + 32×512    = 131,072
  V:          same as K                                  = 131,072
  O:          same as Q                                  = 229,376
  gate_proj:  D×r + r×FFN         = 3584×32 + 32×18944  = 721,920
  up_proj:    same                                       = 721,920
  down_proj:  FFN×r + r×D         = 18944×32 + 32×3584  = 721,920
              ─────────────────────────────────────────
  total/layer                                           ≈ 2,886,656

× 28 layers  = 80,826,368 params
× 2 bytes    = 161,652,736 bytes ≈ 0.15 GiB  (BF16)
```

---

### 5 — Adam Optimizer States

Adam keeps two fp32 copies of every trainable parameter (first moment `m`, second moment `v`):

```
optimizer bytes = 2 × LoRA_params × 4   (fp32 = 4 bytes)
               = 2 × 80,826,368 × 4
               = 646,610,944 bytes ≈ 0.60 GiB
```

---

## Total Formula

```
Total = weights + LoRA + optimizer + KV_cache + activations
      then × 1.10 for CUDA buffers, vision encoder, token embeddings, misc
```

### BF16 (our current training precision)

| max_seq_len | weights | LoRA | optim | KV | act | subtotal | +10% = **total** |
|---|---|---|---|---|---|---|---|
| 2048 | 14.2 | 0.15 | 0.60 | 0.11 | 0.77 | 15.83 | **~17.4 GiB** |
| **4096** | 14.2 | 0.15 | 0.60 | 0.22 | 1.54 | 16.71 | **~18.4 GiB** ← current |
| 8192 | 14.2 | 0.15 | 0.60 | 0.44 | 3.08 | 18.47 | **~20.3 GiB** |
| 16384 | 14.2 | 0.15 | 0.60 | 0.88 | 6.16 | 21.99 | **~24.2 GiB** |

### 4-bit (load_in_4bit=True)

| max_seq_len | weights | LoRA | optim | KV | act | subtotal | +10% = **total** |
|---|---|---|---|---|---|---|---|
| 2048 | 3.5 | 0.15 | 0.60 | 0.11 | 0.77 | 5.13 | **~5.6 GiB** |
| **4096** | 3.5 | 0.15 | 0.60 | 0.22 | 1.54 | 6.01 | **~6.6 GiB** |
| 8192 | 3.5 | 0.15 | 0.60 | 0.44 | 3.08 | 7.77 | **~8.5 GiB** |
| 16384 | 3.5 | 0.15 | 0.60 | 0.88 | 6.16 | 11.29 | **~12.4 GiB** |

---

## Which Term Dominates

| Term | BF16 S=4096 | Scales with |
|---|---|---|
| Model weights | 14.2 GiB | fixed (precision only) |
| Activations | ~1.5 GiB | **O(S)** — grows with seq_len |
| Optimizer (Adam) | 0.60 GiB | fixed (LoRA size only) |
| KV cache | 0.22 GiB | O(S) — but negligible |
| LoRA params | 0.15 GiB | fixed |

**Weights dominate in BF16.** Going from `max_seq_len=4096` to `16384` adds only ~5.8 GiB.
All values fit comfortably on Leonardo A100 80 GiB — you never risk OOM from seq_len alone.

---

## fps=2.0 Implication

At fps=2.0, visual tokens double per clip:
- 30s video: 60 frames × 193 tokens/frame = **11,571 visual tokens**
- At `max_seq_len=4096`: only the first ~20 frames fit; the rest are silently truncated
- At `max_seq_len=8192`: a 30s clip fits (~20.3 GiB BF16) — safe on A100 80 GiB

**Recommended setting for fps=2.0 training:**
```bash
--max_seq_len 8192   # covers clips up to ~30s fully; adds ~2 GiB vs default 4096
```

---

## Scaling LoRA Rank

If you increase LoRA rank from `r=32` to `r=64`:
- LoRA params double: 0.15 → 0.30 GiB
- Adam optimizer doubles: 0.60 → 1.20 GiB
- Net addition: ~0.75 GiB — negligible on A100 80 GiB

`r=64` is safe at any `max_seq_len` in this table.
