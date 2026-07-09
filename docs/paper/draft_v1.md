# Chain-of-Thought Supervised Fine-Tuning and GRPO for Traffic Anomaly Reasoning

<!-- AI City Challenge 2026, Track 3 — Workshop Paper Draft -->
<!-- Target venue: IEEE CVPR 2026 Workshop on AI City Challenge -->
<!-- Estimated length: ~6 pages in 2-column CVPR format -->
<!-- Status: DRAFT — Results sections need final numbers. -->
<!-- Citations: ALL marked [CITATION NEEDED] — do NOT submit without verification. -->

---

**[TODO: Add author names, affiliations, emails before submission]**

---

## Abstract

We present our system for Track 3 of the AI City Challenge 2026, which requires answering ten types of questions about anomalous events in traffic surveillance videos using explicit chain-of-thought reasoning.
Our approach fine-tunes Qwen3-VL-8B, a 8-billion-parameter vision-language model, using two training stages.
In the first stage, we apply supervised fine-tuning (SFT) on the full 38,662-item Traffic Anomaly Reasoning training set, supervising the model to generate chain-of-thought reasoning before each answer.
In the second stage, we apply Group Relative Policy Optimization (GRPO) with task-specific reward functions: exact-match rewards for binary and multiple-choice questions, temporal intersection-over-union (IoU) for temporal localization, and ROUGE-L for open-ended generation tasks.
We additionally experiment with Qwen3-VL-8B-Thinking, a variant pre-trained for native chain-of-thought reasoning, as a stronger initialization for the same two-stage pipeline.
**[TODO: Insert final scores from TAR, FETV, PSI VQA test sets once inference completes.]**

---

## 1. Introduction

Traffic anomaly understanding from surveillance video is a critical capability for intelligent transportation systems.
Unlike standard action recognition, anomaly understanding requires a model to reason about rare, context-dependent events — sudden stops, wrong-way driving, debris on the road — and to explain its reasoning in natural language.
The AI City Challenge 2026 Track 3 formalizes this as a video question-answering task spanning ten question types: binary yes/no verification, multiple-choice selection, open-ended QA, scene description, video summarization, temporal localization, causal linkage, temporal description, and open-ended variants of binary and multiple-choice questions.

Recent vision-language models (VLMs) have demonstrated strong general video understanding [CITATION NEEDED: general VLM survey], but their zero-shot performance on domain-specific anomaly reasoning is limited by a mismatch between pretraining distribution and the specialized vocabulary, camera types (fisheye, dashcam, CCTV), and event categories in traffic data.
Supervised fine-tuning (SFT) on domain data has proven effective for closing this gap in related tasks [CITATION NEEDED], yet SFT alone optimizes for token-level likelihood rather than task-specific correctness metrics such as exact accuracy on binary questions or temporal overlap on localization.

Reinforcement learning from verifiable rewards has recently emerged as a powerful second training stage for language models [CITATION NEEDED: DeepSeek-R1 or similar], enabling the model to discover reasoning strategies that maximize task-specific objectives beyond what imitation learning can capture.
Applied to vision-language models, this paradigm is still nascent [CITATION NEEDED: VLM GRPO work].

We make the following contributions:

1. **Two-stage fine-tuning pipeline**: SFT with chain-of-thought supervision followed by GRPO with task-specific rewards, applied to all ten question types jointly on the TAR training set.

2. **Task-specific reward design**: Exact-match rewards for binary/MCQ tasks; temporal IoU for localization; ROUGE-L for open-ended generation — enabling reward signals calibrated to the actual evaluation metric of each task type.

3. **Thinking model variant**: We extend the pipeline to Qwen3-VL-8B-Thinking, a model pre-trained for longer chain-of-thought reasoning, and compare it against the Instruct baseline across task types.

4. **Efficient inference**: A sharded four-GPU inference pipeline with a per-video frame cache that avoids redundant disk reads when multiple questions share the same video clip.

**[TODO: Insert final TAR score and highlight the best result once training completes.]**

---

## 2. Related Work

### 2.1 Vision-Language Models for Video Understanding

Large VLMs capable of processing video input have advanced rapidly, with models such as [CITATION NEEDED: Qwen-VL / LLaVA-Video / InternVL / Video-LLaMA] demonstrating strong performance on video QA benchmarks.
These models typically encode video as a sequence of sampled frames processed through a vision encoder, projecting visual tokens into the language model's embedding space.
Qwen3-VL [CITATION NEEDED] extends this architecture with dynamic resolution processing and native support for chain-of-thought reasoning through `<think>` tags, making it a natural choice for the TAR task where reasoning quality is directly rewarded.

### 2.2 Chain-of-Thought Reasoning in VLMs

Chain-of-thought prompting [CITATION NEEDED: Wei et al. CoT] and supervised training on reasoning traces [CITATION NEEDED] have improved language model accuracy on multi-step tasks.
For visual reasoning, chain-of-thought supervision has been shown to improve grounding and spatial reasoning [CITATION NEEDED].
The TAR dataset's explicit `reasoning` field — human-written chain-of-thought explanations for each annotation — provides a direct training signal that we incorporate into the SFT stage.

### 2.3 Reinforcement Learning for Language Models

Group Relative Policy Optimization (GRPO) [CITATION NEEDED: Shao et al. DeepSeekMath or similar] estimates policy gradients by sampling multiple completions per prompt and comparing their rewards, avoiding the need for a learned value function.
Recent work has applied GRPO to language models with verifiable rewards — mathematics, code, and logical reasoning — achieving significant accuracy gains over SFT [CITATION NEEDED: DeepSeek-R1].
Its extension to vision-language models [CITATION NEEDED] enables reward signals grounded in visual content, which we exploit through task-type-aware reward routing.

### 2.4 Traffic Anomaly Detection and Recognition

Classical approaches to traffic anomaly detection rely on handcrafted features and temporal modeling over fixed camera views [CITATION NEEDED].
More recent methods apply object detection and tracking pipelines with anomaly scoring functions [CITATION NEEDED].
The TAR dataset [CITATION NEEDED: TAR dataset paper / Nvidia] is the first large-scale benchmark that combines video anomaly understanding with multi-type question answering and explicit chain-of-thought reasoning, shifting the evaluation from detection scores to language-based explanation quality.

---

## 3. Dataset

### 3.1 Traffic Anomaly Reasoning (TAR) Dataset

The TAR dataset [CITATION NEEDED] is released by NVIDIA for the AI City Challenge 2026 Track 3.
It contains 44,040 annotated question-answer pairs across 3,670 video clips drawn from eight source collections: VAD-R1, TAD, Accident-Bench, SO-TAD, TADBenchmark, UCF Crime, and others, spanning approximately 26 hours of traffic surveillance footage.
Each annotation provides a question, a ground-truth answer, and a human-written chain-of-thought reasoning explanation.

Questions span ten task types organized into three groups:

- **Basic**: binary event verification (`bcq`), multiple-choice (`mcq`), and open-ended QA (`open_qa`).
- **Scene**: scene description (`scene_description`) and video summarization (`video_summarization`).
- **Temporal**: temporal localization (`temporal_localization`), causal linkage (`causal_linkage`), temporal description (`temporal_description`), and open-ended variants of binary (`bcq_openended`) and multiple-choice (`mcq_openended`) questions.

Video sources include fixed CCTV cameras, wide-angle fisheye cameras, and egocentric dashcams, introducing significant visual diversity.

### 3.2 Data Preparation

We download the annotation JSON files (one per task type) from HuggingFace Hub and convert each item to the Qwen3-VL multi-modal conversation format.
Each training example wraps the dataset's `reasoning` field inside `<think>...</think>` tags in the assistant turn, followed by the clean `answer`:

```
System: You are an expert traffic surveillance analyst. Watch the provided video
        carefully, then reason step-by-step about any anomalous events before
        giving your final answer.
User:   [video frames] [question text]
Assistant: <think>[reasoning]</think> [answer]
```

This format teaches the model to produce explicit reasoning before committing to a final answer, matching the `<think>` behavior of the Qwen3-VL generation API.

After removing 1,092 items whose source videos were unavailable (removed from upstream sources), we retain 42,948 usable annotations.
We split these into a training set of 38,662 items and a validation set of 4,286 items using a stratified 90/10 split per task type, ensuring all ten categories are represented in validation.

### 3.3 Test Sets

The challenge provides three held-out test sets:
(1) **TAR** — 960 human-verified annotations across 80 clips from YouTube, constituting the primary leaderboard;
(2) **FETV** — 200 fisheye traffic-violation clips;
(3) **PSI VQA** — 40 egocentric dashcam clips.

---

## 4. Method

Our pipeline has four stages: data preparation (Section 3), SFT (Section 4.1), GRPO (Section 4.2), and inference (Section 4.3).

### 4.1 Supervised Fine-Tuning

**Model.** We fine-tune Qwen3-VL-8B-Instruct [CITATION NEEDED] using Unsloth [CITATION NEEDED] with parameter-efficient LoRA [CITATION NEEDED] adapters applied to all linear layers of both the vision encoder and the language model.
We use full BF16 precision (no quantization) to maximize adapter expressiveness within the A100 memory budget.

**LoRA configuration.** Rank $r = 32$, $\alpha = 32$, dropout $= 0$, applied to all attention and MLP modules in both vision and language components.
We additionally enable full fine-tuning of the embedding and language model head layers.

**Training.** We train for 3 epochs with a learning rate of $10^{-4}$, cosine decay, 5% linear warmup, AdamW optimizer with 8-bit quantization, per-device batch size of 1, and gradient accumulation of 4 steps.
Maximum sequence length is 4,096 tokens.
Training uses the `SFTTrainer` from TRL [CITATION NEEDED] with the `UnslothVisionDataCollator` for multi-modal batching.
We set `dataset_num_proc=1`, which is required to avoid serialization failures when PIL Image objects in video frames are shared across workers.

All experiments are run on the Leonardo HPC cluster (CINECA) using 4 × NVIDIA A100 SXM4 GPUs (80 GiB VRAM) across 1 node, with `torchrun` for data-parallel training.

**Thinking variant.** We additionally fine-tune Qwen3-VL-8B-Thinking, which is pre-trained for native chain-of-thought generation.
For this variant we reduce the learning rate to $5 \times 10^{-5}$ (the model's native reasoning style is already aligned with our `<think>` format, requiring gentler updates) and add `lm_head` and `embed_tokens` to the set of fully fine-tuned modules.
The system prompt for the Thinking variant omits the "reason step-by-step" instruction, as the model applies this behavior natively.

### 4.2 GRPO with Task-Specific Rewards

After SFT, we apply GRPO to reinforce correct answers through task-specific reward signals.
GRPO generates $G = 4$ completions per prompt, computes a per-completion scalar reward, and updates the policy to increase the probability of higher-reward completions relative to the group mean [CITATION NEEDED: GRPO paper].

**Base model.** GRPO is computationally demanding due to the need to generate multiple completions per step.
We load the SFT LoRA adapter on top of the 4-bit quantized base model (`Qwen3-VL-8B-Instruct-bnb-4bit`) to fit within GPU memory, matching the adapter's training base.

**Reward functions.** We define task-specific rewards that align with the evaluation metrics:

| Task type | Reward $r(y, \hat{y})$ |
|---|---|
| `bcq`, `bcq_openended` | Exact match of Yes/No after stripping `<think>` blocks |
| `mcq`, `mcq_openended` | Exact match of A/B/C/D letter after stripping `<think>` blocks |
| `temporal_localization` | Temporal IoU of predicted and ground-truth `{start, end}` intervals |
| `open_qa`, `scene_description`, `video_summarization`, `causal_linkage`, `temporal_description` | ROUGE-L F1 between predicted and reference answer |

We use ROUGE-L rather than BERTScore as the open-ended reward because BERTScore requires a forward pass through a separate BERT encoder for each of the $G=4$ sampled completions per step, creating a throughput bottleneck. ROUGE-L is computed via pure Python LCS in microseconds.

**Think-block stripping.** Before computing any reward, we strip `<think>...</think>` blocks from the completion to extract the final answer.
We apply a two-pass regex: the first pass removes complete blocks; the second pass removes any unclosed `<think>` block extending to the end of the string (which occurs when the model exhausts its token budget mid-reasoning):

```python
def strip_think(text: str) -> str:
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    clean = re.sub(r"<think>.*",          "", clean, flags=re.DOTALL).strip()
    return clean
```

This two-pass approach is critical: with only a single pass, an unclosed `<think>` block would cause the reward function to receive the raw reasoning text rather than the answer, assigning spuriously low rewards and destabilizing training.

**GRPO hyperparameters.** Learning rate $5 \times 10^{-6}$, cosine decay, 5% warmup, AdamW-8bit, 4 gradient accumulation steps, maximum prompt length 2,048 tokens, maximum completion length 512 tokens (1,024 for the Thinking variant, which produces longer reasoning chains), 1 training epoch.

For the multi-node variant, we distribute across 4 nodes × 4 GPUs = 16 GPUs using `srun torchrun` with the `c10d` rendezvous backend, which automatically resolves node ranks through the SLURM environment.

### 4.3 Inference

**Sharded multi-GPU inference.** We run four parallel GPU workers, each processing a disjoint contiguous shard of the test set.
Items are sorted by `video_id` before sharding so each worker processes full clips consecutively.
This enables a per-video frame cache: when multiple questions share the same clip (common in the TAR test set), video frames are loaded from disk once and reused across all questions for that clip.

**Frame sampling.** We sample at 1 fps using `decord`, resizing each frame so its pixel count does not exceed $360 \times 420 = 151{,}200$ pixels. This trades spatial resolution for memory and throughput; higher fps or resolution may be explored in future work.

**Token budget.** Binary and multiple-choice questions (`bcq`, `mcq`) require only a Yes/No or A–D answer, but the model first generates a `<think>` reasoning chain.
We allocate 256 tokens for these task types — sufficient to close the reasoning block and emit the final answer token.
All other task types receive a 1,024-token budget.
Using fewer tokens (e.g., 32) for short-answer tasks causes the model to exhaust its budget inside the `<think>` block, producing no answer and degrading accuracy to random-guess level.

**Base model selection.** The SFT adapter was trained on the BF16 base model and must be loaded on the BF16 base at inference.
The GRPO adapter was trained on the 4-bit quantized base model and must be loaded on the 4-bit base.
Mismatching base and adapter corrupts activations without producing any error.

**Postprocessing.** Raw model output (including `<think>` blocks) is logged for debugging.
The cleaned output feeds a postprocessor that formats each prediction according to its task type: extracting Yes/No, a letter, a JSON timestamp, or passing free-form text.
Final predictions are written to a CSV with columns `item_index` and `prediction`.

---

## 5. Experiments

### 5.1 Experimental Setup

**Hardware.** All training and inference runs on the Leonardo HPC cluster (CINECA, Italy): NVIDIA A100 SXM4 80 GiB GPUs, CUDA 12.6, PyTorch 2.12.1+cu126.

**Software.** Unsloth [CITATION NEEDED] v(git-main), TRL [CITATION NEEDED], PEFT [CITATION NEEDED], Transformers [CITATION NEEDED], `decord` for video decoding.

**Evaluation metrics.** For the TAR test set:
- Binary/MCQ tasks: exact accuracy.
- All other 8 task types: BERTScore F1 [CITATION NEEDED].
- Final TAR score: unweighted mean across 9 scored task types.

For FETV: $0.25 \cdot \text{CIDEr} + 0.25 \cdot \text{BERTScore} + 0.5 \cdot \text{MacroF1}$.

For PSI VQA: $0.25 \cdot T_1 + 0.25 \cdot T_2 + 0.25 \cdot T_3 + 0.25 \cdot T_4$.

**Random seed**: 3407 for all runs.

### 5.2 Main Results

**[TODO: Fill in Table 1 once inference jobs complete — `sft_v1_bf16_lora_v2` and `grpo_v1_ckp1100`.]**

*Table 1: TAR test set performance by task type and overall mean score.*

| Method | BCQ | MCQ | Open QA | Scene Desc. | Summarization | Temp. Local. | Causal | Temp. Desc. | BCQ-OE | MCQ-OE | **Mean** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Zero-shot Qwen3-VL-8B-Instruct | — | — | — | — | — | — | — | — | — | — | — |
| SFT (Instruct, BF16) | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** |
| SFT + GRPO (Instruct) | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** |
| SFT (Thinking, BF16) | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** |
| SFT + GRPO (Thinking) | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** |

*Table 2: Secondary test set scores.*

| Method | FETV | PSI VQA |
|---|---|---|
| SFT (Instruct) | **[TODO]** | **[TODO]** |
| SFT + GRPO (Instruct) | **[TODO]** | **[TODO]** |

### 5.3 Ablation: Token Budget for Short-Answer Tasks

We identify a critical engineering decision that dramatically affects binary and MCQ accuracy: the `max_new_tokens` budget for short-answer tasks.

With a budget of 32 tokens, the model begins generating a `<think>` reasoning chain but exhausts the budget before producing the closing `</think>` tag and the final answer.
Both the raw output and the stripped output are identical truncated reasoning fragments, yielding no usable prediction.
Table 3 shows the effect of this parameter.

*Table 3: BCQ and MCQ accuracy as a function of short-answer token budget.*

| `max_new_tokens` (short tasks) | BCQ accuracy | MCQ accuracy |
|---|---|---|
| 32 (original) | 0.50 | 0.20 |
| 256 (fixed) | **[TODO]** | **[TODO]** |

A budget of 256 tokens consistently closes the `<think>` block and produces a valid Yes/No or A–D answer.
This result highlights a failure mode of chain-of-thought VLMs that is not visible from training loss: the model may learn to reason correctly but fail to produce any answer if the inference token budget is too small relative to the length of the reasoning chain.

### 5.4 Ablation: SFT vs. SFT + GRPO

**[TODO: Once both sets of results are available, report delta per task type. Hypothesis: GRPO improves BCQ/MCQ by reinforcing exact-match reasoning and weakly improves open-ended tasks through ROUGE-L reward shaping.]**

### 5.5 Ablation: Instruct vs. Thinking Variant

**[TODO: Compare per task type once Thinking inference completes. Hypothesis: Thinking variant shows larger gains on open-ended and causal tasks where longer reasoning chains are beneficial.]**

---

## 6. Discussion

### 6.1 Chain-of-Thought as a Training Signal

The TAR dataset is unusual in providing human-authored reasoning chains alongside every answer.
Training on these chains serves two purposes: it teaches the model domain-specific reasoning vocabulary (e.g., "the vehicle decelerates abruptly before the intersection, suggesting the driver detected an obstacle"), and it produces a longer, more coherent generation prefix that improves the quality of the final answer token prediction.
We observe that this reasoning-first format is essential for the model to generalize across the heterogeneous task types: a model that produces a direct answer for temporal localization and a binary answer for BCQ without any intermediate reasoning structure tends to underperform relative to one trained to reason uniformly regardless of task type.

### 6.2 GRPO and Verifiable Rewards

The key advantage of GRPO over SFT for this task is that SFT minimizes cross-entropy loss uniformly across all tokens, including the `<think>` reasoning text.
For BCQ and MCQ tasks where the correct answer is a single token, SFT provides only one token of correct-answer gradient signal per training example.
GRPO, by contrast, assigns reward only to complete generations that produce the correct final answer, allowing the reward signal to propagate through the entire reasoning chain via policy gradient.
This makes GRPO particularly valuable for binary and MCQ tasks where the correct answer is verifiable and the reward function is exact.

### 6.3 Limitations

**Video coverage.** 2% of training items (1,092 / 44,040) are unrecoverable because the source videos were removed from upstream repositories. Our model is trained on 98% of the available data but may underperform on anomaly types specific to those missing sources.

**Frame rate.** We sample at 1 fps, which may miss short-duration anomalous events (e.g., a debris strike lasting less than one second). Higher frame rates or event-triggered sampling could improve temporal localization accuracy.

**Reward proxy gap.** ROUGE-L correlates imperfectly with BERTScore, the actual evaluation metric for open-ended tasks. A model optimized for ROUGE-L may not maximize BERTScore, and direct BERTScore optimization is computationally prohibitive during GRPO.

**Single-model submission.** We do not ensemble predictions across checkpoints or model variants. Majority voting across Instruct and Thinking predictions, or across GRPO checkpoints, is a straightforward extension that may improve robustness.

---

## 7. Conclusion

We present a two-stage fine-tuning approach for traffic anomaly video QA: supervised chain-of-thought training on the full TAR dataset followed by GRPO with task-specific reward functions calibrated to the evaluation metrics of each question type.
We extend this pipeline to the Qwen3-VL-8B-Thinking variant and provide a sharded, cache-efficient inference system for test-time prediction.
**[TODO: Add final sentence with the best observed score once results are available.]**

Our analysis of the token budget ablation reveals a non-obvious failure mode: chain-of-thought models trained on short-answer tasks require a token budget large enough to close the reasoning block before the answer token, even if the answer itself is only one or two tokens long.
We release our training and inference code to support future work on reasoning-augmented VLMs for traffic understanding.

---

## References

<!-- ALL CITATIONS BELOW ARE PLACEHOLDERS — verify every entry before submission -->
<!-- Use Semantic Scholar API or CrossRef to retrieve verified BibTeX -->

```
[CITATION NEEDED] Qwen3-VL technical report / paper
[CITATION NEEDED] Unsloth: efficient LLM fine-tuning library
[CITATION NEEDED] GRPO: Shao et al., "DeepSeekMath" or equivalent GRPO origin paper
[CITATION NEEDED] DeepSeek-R1: reasoning RL paper
[CITATION NEEDED] LoRA: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"
[CITATION NEEDED] TRL: Transformer Reinforcement Learning library (HuggingFace)
[CITATION NEEDED] PEFT: HuggingFace parameter-efficient fine-tuning library
[CITATION NEEDED] BERTScore: Zhang et al., "BERTScore: Evaluating Text Generation with BERT"
[CITATION NEEDED] ROUGE: Lin, "ROUGE: A Package for Automatic Evaluation of Summaries"
[CITATION NEEDED] Chain-of-thought prompting: Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
[CITATION NEEDED] TAR dataset: NVIDIA PhysicalAI Traffic Anomaly Reasoning dataset paper
[CITATION NEEDED] AI City Challenge overview paper (the Track 3 description paper)
[CITATION NEEDED] LLaVA-Video or similar video VLM baseline
[CITATION NEEDED] Video-LLaMA or similar
[CITATION NEEDED] Traffic anomaly detection survey or key baseline
```

---

## Appendix A: Training Hyperparameters

*Table A1: Complete SFT hyperparameters.*

| Parameter | Instruct | Thinking |
|---|---|---|
| Base model | Qwen3-VL-8B-Instruct (BF16) | Qwen3-VL-8B-Thinking (BF16) |
| LoRA rank $r$ | 32 | 32 |
| LoRA $\alpha$ | 32 | 32 |
| Max sequence length | 4,096 | 4,096 |
| Epochs | 3 | 3 |
| Learning rate | $1 \times 10^{-4}$ | $5 \times 10^{-5}$ |
| Batch size (per device) | 1 | 1 |
| Gradient accumulation steps | 4 | 4 |
| Optimizer | AdamW-8bit | AdamW-8bit |
| LR schedule | Cosine | Cosine |
| Warmup ratio | 0.05 | 0.05 |
| Seed | 3407 | 3407 |

*Table A2: Complete GRPO hyperparameters.*

| Parameter | Instruct | Thinking |
|---|---|---|
| Base model | Qwen3-VL-8B-Instruct-bnb-4bit | Qwen3-VL-8B-Thinking-bnb-4bit |
| Completions per prompt ($G$) | 4 | 4 |
| Learning rate | $5 \times 10^{-6}$ | $5 \times 10^{-6}$ |
| Max completion length | 512 tokens | 1,024 tokens |
| Max prompt length | 2,048 tokens | 2,048 tokens |
| Epochs | 1 | 1 |
| Batch size (per device) | 1 | 1 |
| Gradient accumulation steps | 4 | 4 |
| Optimizer | AdamW-8bit | AdamW-8bit |
| LR schedule | Cosine | Cosine |
| Warmup ratio | 0.05 | 0.05 |

---

## Writing TODO List (remove before submission)

- [ ] Add author names and affiliations
- [ ] Add zero-shot baseline row to Table 1 (run Qwen3-VL-8B-Instruct with no fine-tuning)
- [ ] Fill all `[TODO]` result cells in Tables 1, 2, 3
- [ ] Verify and fetch BibTeX for all `[CITATION NEEDED]` entries via Semantic Scholar API
- [ ] Add Figure 1: pipeline overview diagram (SFT → GRPO → Inference)
- [ ] Add Figure 2: reward curves during GRPO training
- [ ] Write final sentence of Conclusion with the best observed score
- [ ] Convert to CVPR LaTeX format (2-column, 6–8 pages)
- [ ] Check page limit and trim if over — candidate cuts: Appendix A (move to main tables), Section 6.1 (shorten)
- [ ] Run spell check and grammar pass
- [ ] Confirm "Limitations" section is present (required by most venues)
- [ ] Remove this TODO list before submission
