# Parameter-Efficient HTS Classification at the Edge: An 8B LoRA Adapter Statistically Ties a 70B Benchmark for $43 of Training Compute

**Working draft of an arXiv-style preprint. Companion to the canonical.agency essay at `docs/paper-v1.md`.**

---

## Abstract

I present v1 of an open-weight LoRA adapter for U.S. Harmonized Tariff Schedule (HTS) classification, trained on Llama-3.1-Nemotron-Nano-8B using 119,602 CROSS ruling examples on a single H100 GPU for approximately $43 of training compute. The model produces structured chapter-heading-subheading-code output suitable for direct consumption by downstream pipelines. On the public 200-example ATLAS test set (Yuvraj & Devarakonda, 2025), the v1 LoRA achieves 41.0% exact-match accuracy under HF transformers + bitsandbytes nf4 4-bit inference, statistically indistinguishable from the ATLAS paper's reported Llama-3.3-70B full SFT result of 40.0% at 1/8 the parameter count. The same ATLAS paper reports zero-shot GPT-5-Thinking at 25.0% and zero-shot Gemini-2.5-Pro-Thinking at 12.5% on the same test. The base Nemotron-Nano-8B model with no adapter achieves 0.0% exact-match, demonstrating that the LoRA contributes essentially all the task capability. On a held-out CROSS-derived test set of 14,952 examples, the model achieves 58.7% exact-match (95% CI ±0.8pp) and 92.0% chapter accuracy. Production deployment via MLX q4 quantization on M4 Mac mini hardware achieves 36.0% ATLAS exact-match with 4.7 second single-stream latency, 11pp above the GPT-5-Thinking zero-shot baseline. I additionally report a measured negative ablation: stat-suffix validation against a training-derived index of 9,791 valid 10-digit codes produces no measurable accuracy lift, because the model already emits valid HTSUS codes in 87-96% of predictions. The dominant failure mode is wrong-but-valid prediction rather than invalid hallucination, indicating that future improvements should target disambiguation (retrieval-augmented inference) rather than validation. All code, weights, evaluation data, and reproducibility artifacts are publicly available at the URL in §7.

---

## 1. Introduction

The U.S. Harmonized Tariff Schedule contains approximately 19,000 10-digit codes organized hierarchically into chapters (2-digit), headings (4-digit), and subheadings (6 and 8 digit). Every product imported into the United States must be classified into exactly one 10-digit code, which determines applicable duty rates, country-of-origin trade preferences, statistical reporting requirements, and a long tail of compliance flags. Misclassification is costly in both directions: importers risk over-payment of duties on conservative classifications and penalties on under-valuation.

The HTS classification task has attracted recent attention as a natural target for language model specialization. The hierarchical structure of the schedule, the availability of CBP's published binding rulings (CROSS) as a training corpus, and the high commercial value of automated classification combine to make it an unusually clean specialization problem. Two recent papers anchor the field.

Yuvraj & Devarakonda (2025, the "ATLAS paper", arXiv:2509.18400) introduce the ATLAS benchmark: 18,254 training examples, 200 validation, 200 test, all derived from CROSS rulings. They report a fully fine-tuned Llama-3.3-70B at 40.0% exact-match accuracy on the 10-digit code level on their 200-example test set. The same paper reports zero-shot frontier model baselines on the same test set: GPT-5-Thinking at 25.0%, Gemini-2.5-Pro-Thinking at 12.5%. The paper's headline finding is that domain-specific fine-tuning of a 70B model produces a +15pp improvement over the strongest zero-shot baseline.

Judy (2024, arXiv:2412.14179) takes a different approach: a benchmarking study of commercial HTS SaaS tools on a separate 103-example test set. The study evaluates Tarifflo (a retrieval-augmented commercial pipeline) at 89.2%, Avalara (expert-plus-AI workflow) at 80.0%, plus Zonos and the WCO's BACUDA tool. These two benchmarks bracket the design space but are not directly comparable to each other: the test sets are distinct, the scoring methodologies differ, and the architectural assumptions are inverted. ATLAS measures what a fully fine-tuned model can do alone. Judy measures what commercial pipelines do with retrieval, expert review, and explanation generation layered on top.

A note on the zero-shot frontier comparisons used throughout this paper: the GPT-5-Thinking 25.0% and Gemini-2.5-Pro-Thinking 12.5% baselines are the zero-shot configurations measured by the ATLAS paper authors. Few-shot prompting or chain-of-thought elaboration of the same frontier models would presumably score higher. The question this paper addresses is not "can a sufficiently prompted frontier model match a specialized fine-tune," because the answer is almost certainly yes for some prompting configuration. The question is "what does it cost to serve a model with this capability locally on commodity hardware, and is the cost-performance tradeoff favorable enough to displace recurring API or SaaS spend." I return to that economics framing in Section 9.

A gap in this design space has not been addressed by published work: whether a substantially smaller model, on the order of 8 billion parameters rather than 70 billion, with parameter-efficient fine-tuning rather than full SFT, can reach the model-only benchmark frontier at full precision and remain useful after edge-quantization for consumer hardware deployment. If so, the deployment economics change qualitatively. A model that fits in 5 GB at 4-bit quantization runs interactively on consumer hardware (specifically, the 16 GB M4 Mac mini, MSRP $700). The recurring marginal cost of classification drops from "API call to a hosted frontier model" to zero. Confidential product data never leaves the deploying organization's network.

This paper reports a v1 result for that question. I trained a LoRA adapter (Hu et al., 2022) on Llama-3.1-Nemotron-Nano-8B using 119,602 CROSS ruling examples on a single H100 GPU for approximately $43 of compute and 14 hours of wall time. The resulting 336 MB adapter, when merged into the base model and deployed via MLX q4 quantization on an M4 Mac mini, achieves 36.0% exact-match accuracy on the ATLAS public test set. The same adapter under HF transformers with bitsandbytes nf4 4-bit inference achieves 41.0%, statistically indistinguishable from the ATLAS-reported 70B SFT baseline of 40.0%. The base Nemotron-Nano-8B model with no adapter scores 0.0% exact-match on the same test, demonstrating that the LoRA contributes essentially all the task capability.

I additionally report a methodological negative result. The dominant failure mode on both test sets is at the 10-digit statistical suffix: 2,458 of 6,206 errors on a held-out 14,952-example test set (39.5%) and 23 of 116 errors on ATLAS (19.8%) are cases where the model predicts the correct 8-digit subheading but a wrong 2-digit suffix. I constructed the obvious post-processing fix: a training-derived index of valid 10-digit completions per 8-digit subheading, with a most-frequent-completion picker for invalid predictions. The measured lift on both test sets is essentially zero. Investigation reveals that the model already produces valid HTSUS 10-digit codes in 87.3% of held-out predictions and 96.5% of ATLAS predictions. The dominant failure mode is wrong-but-valid prediction rather than invalid hallucination; stat-suffix validation cannot address it. The path to the next factor of accuracy improvement therefore lies in disambiguation (retrieval-augmented inference, code-embedding nearest-neighbor reranking) rather than output validation.

The contributions of this paper are:

1. A working open-weight 8B LoRA HTS classifier that statistically ties the published 70B SFT benchmark on the ATLAS test set at 1/8 the parameter count and approximately $43 of training compute.
2. A measured cross-quantization parity study (HF nf4 vs MLX bf16 vs MLX q4) that isolates the 5pp accuracy cost of consumer-hardware deployment.
3. A measured negative ablation on stat-suffix validation, with diagnostic data showing the model already emits valid HTSUS codes at high rates.
4. An end-to-end edge deployment stack with reproducible 60-second cold-start from a fresh checkout and 4.7-second single-stream classification latency on the target M4 Mac mini hardware.

## 2. Related Work

**Domain-specific HTS classification.** Yuvraj & Devarakonda (2025) introduce the ATLAS benchmark and report results for several configurations including a fully fine-tuned Llama-3.3-70B model (40.0% exact-match), zero-shot GPT-5-Thinking (25.0%), and zero-shot Gemini-2.5-Pro-Thinking (12.5%). Their training pipeline uses approximately 18,000 examples derived from CROSS rulings with label-conditioned reasoning trace generation. Judy (2024) benchmarks commercial HTS tools (Tarifflo, Avalara, Zonos, WCO BACUDA) on a separate 103-example test set; Tarifflo's retrieval-augmented pipeline reaches 89.2% and Avalara's expert-plus-AI workflow 80.0%. These two benchmarks bracket the field but are not directly comparable: the test sets are distinct, the prompt formats differ, and the architectural assumptions are inverted (model-only vs pipeline).

**Parameter-efficient fine-tuning.** LoRA (Hu et al., 2022) decomposes the weight update of a fine-tuning step into low-rank matrices, enabling task adaptation with two to three orders of magnitude fewer trainable parameters than full SFT. The v1 adapter reported here has 32M trainable parameters against an 8B base, a ratio of approximately 0.4%. The training data, hyperparameter selection, and architectural choices in this paper are conventional for LoRA fine-tuning of an instruction-tuned base model.

**Structured output and constrained generation.** Recent work on guided and constrained generation (Willard & Louf, 2023) provides runtime grammar enforcement that guarantees schema-conformant LLM output. The v2 output format used here requires the model to emit Chapter, Heading, Subheading, HTS Code, and Reasoning sections in a fixed order, but is enforced via supervised fine-tuning on the format rather than via runtime constrained decoding. The model is free to produce malformed output and the parser flags it. In practice, parse failure occurs at a rate of 0.06% on the held-out test (9 of 14,952), suggesting the model has internalized the format via supervised fine-tuning rather than requiring runtime enforcement. The negative result on stat-suffix validation (Section 6) is consistent with the same observation: the model emits valid HTSUS 10-digit codes in 87-96% of predictions without runtime constraint.

**Quantization-aware deployment.** The MLX framework (Hannun et al., 2023) provides Apple Silicon-native inference with native q4 and q8 quantization. The bitsandbytes library (Dettmers et al., 2022) provides CUDA-native nf4 quantization. These two quantization schemes are not bit-equivalent: bnb nf4 uses normal-float 4-bit with optional double quantization; MLX q4 uses a simpler affine quantization. The accuracy gap between them, in the measurements reported in Section 4.3, is approximately 5 percentage points on ATLAS exact-match.

**Adjacent regulatory classification tasks.** Several recent papers explore LLM-assisted classification in regulatory domains with structural similarity to HTS. Xu (2022) and Han et al. (2025) report on LLM-assisted FDA medical device premarket classification, where the gold answers exist in public regulatory rulings and the operational sensitivity (proprietary device designs) makes hosted API approaches problematic. The FDA itself ran an AI-assisted scientific review pilot in early 2025. India's HSN tariff classification, EU REACH chemical classification, and ICD-10 medical coding share the same structural pattern: a large hierarchical taxonomy, structured outputs required, gold answers in public regulatory data, and operational sensitivity around the input data. This paper does not attempt to evaluate the approach on those tasks but argues in Section 9 that the deployment pattern transfers.

## 3. Method

### 3.1 Training data

The training corpus is 119,602 examples derived from the U.S. Customs and Border Protection's Cross Ruling (CROSS) database, a public collection of CBP's binding rulings on commodity classification. Each example pairs a product description with the gold 10-digit HTS classification assigned by CBP in the ruling. The corpus was filtered to remove rulings predating 2010 (to reflect the current schedule structure) and deduplicated against the (8-digit code, normalized description) pair to remove near-duplicates.

Approximately 14.4% of the training set (17,230 examples) are abstention cases: descriptions that are too vague, ambiguous, or under-specified to classify without additional information. The model is trained to emit an explicit "Cannot classify" response with an explanation of the missing information in these cases. The remaining 102,372 examples have full classifications.

The output format is a fixed five-section structured text:

```
Chapter NN: <description>
Heading NN.NN: <description>
Subheading NNNN.NN: <description>
HTS Code: NNNN.NN.NNNN

Reasoning: <text>

Provides for: <text>
```

This format is parseable by regex and round-trips losslessly through the model's tokenizer. The training data is held in HuggingFace messages format with the v2 system prompt and structured assistant response.

### 3.2 Model

The base model is `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` (8.1 billion parameters), an instruction-tuned variant of Llama 3.1 8B. I apply a LoRA adapter (Hu et al., 2022) targeting the `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj` modules across all 32 transformer layers. LoRA hyperparameters: rank 32, alpha 64 (effective scale 2.0), dropout 0.05. The total parameter count is 32M trainable plus 8.1B frozen.

### 3.3 Training procedure

Training was performed on a single rented H100 SXM 80GB GPU on RunPod. Hyperparameters: 3 epochs, effective batch size 32 (per-device batch 16, gradient accumulation 2), max sequence length 1536, learning rate 2.0e-4 with cosine schedule and 5% warmup, AdamW optimizer in paged 8-bit configuration, bfloat16 mixed precision, 4-bit nf4 quantization of the base model via bitsandbytes for memory efficiency. Attention implementation: scaled dot-product attention (SDPA); I did not use Flash Attention 2 due to ABI compatibility issues with the available PyTorch build (documented at length in `docs/log-2026-04-06-h100-training.md`).

Wall time: 13 hours 44 minutes for training, 2 hours for environment setup and debugging, total approximately 16 hours on the metered GPU. Total cost: approximately $43 at the RunPod published rate of $2.69 per hour for the H100 SXM 80GB at the time of training.

The training run kept the last 3 checkpoints by default (`save_total_limit: 3`). The actually-best checkpoint by eval loss occurred at step 7000 (eval loss 0.2596, epoch 1.87), but was overwritten by later checkpoints with higher eval loss. The shipped adapter is the final-step checkpoint (step 11214, eval loss 0.2729). The accuracy numbers reported in this paper are for the shipped final-step adapter, not the lost best checkpoint. I estimate the lost checkpoint would have produced approximately 1-3pp higher exact-match accuracy based on the typical eval-loss-to-accuracy correspondence for this task; I have not recovered it.

### 3.4 Evaluation

I evaluate on two test sets.

**ATLAS public test (N=200).** The 200-example test set published by Yuvraj & Devarakonda (2025). I use their original test file and convert it to the v2 messages format via `scripts/build_atlas_eval.py`. The ATLAS test set has 31 multi-code rows (15.5%) where multiple correct codes are accepted; I score using any-of-set match semantics to match the ATLAS paper's scoring.

**Held-out CROSS test (N=14,952).** A held-out split of the same CROSS-ruling-derived corpus used for training. The split is ruling-level (no `ruling_number` appears in both train and test), so leakage of specific rulings is excluded. Distribution match to training is intentionally high; this measures in-distribution accuracy with tight statistical bounds.

For both test sets, I report exact-match accuracy at the full 10-digit level, plus prefix-match accuracy at the chapter (2-digit), heading (4-digit), and subheading (6-digit) levels. I additionally report parse rate (the fraction of predictions that successfully parse into the v2 structured format) and hierarchy consistency (the fraction of parseable predictions where the parsed chapter, heading, and subheading are mutually consistent with the predicted 10-digit code).

For ATLAS, the 95% CI on a 41% proportion at N=200 is approximately ±7pp. For the held-out test, the 95% CI on a 58.7% proportion at N=14,952 is approximately ±0.8pp.

### 3.5 Deployment configurations

I measured three inference configurations to isolate the accuracy contribution of the inference backend versus the quantization scheme:

1. **HF nf4**: HuggingFace transformers with bitsandbytes 4-bit nf4 quantization, run on an H100 GPU. Matches the training-time inference configuration.
2. **MLX bf16**: MLX framework with native bf16 (unquantized) weights, run on M4 Mac mini. Diagnostic configuration to isolate quantization cost from inference-path differences.
3. **MLX q4**: MLX framework with native q4 quantization, run on M4 Mac mini. Production edge-deployment configuration.

The HF nf4 path is used for the ATLAS N=200 and the held-out N=14,952 runs. The MLX bf16 path is used for an N=50 diagnostic run on ATLAS. The MLX q4 path is used for the ATLAS N=200 deployment run.

## 4. Results

### 4.1 Main results on ATLAS test set (N=200)

| Configuration | Parameters (trainable / total) | Exact match | Chapter | Heading | Subheading | Parse rate |
|---|---|---|---|---|---|---|
| Base Nemotron-Nano-8B, no adapter | 0 / 8.1B | 0.0% | 2.5% | 0.0% | 0.0% | 12.5% |
| v1 LoRA, HF transformers + nf4 | 32M / 8.1B | **41.0%** | 86.0% | 73.0% | 61.0% | 100.0% |
| v1 LoRA, MLX bf16 (N=50) | 32M / 8.1B | 42.0% | 86.0% | 74.0% | 60.0% | 100.0% |
| v1 LoRA, MLX q4 | 32M / 8.1B | **36.0%** | 82.5% | 67.5% | 54.5% | 100.0% |

Published comparators on the same test set (from Yuvraj & Devarakonda, 2025):

| System | Parameters | Configuration | Exact match |
|---|---|---|---|
| ATLAS Llama-3.3-70B SFT | 70B (full SFT) | H100, full precision | 40.0% |
| GPT-5-Thinking | (closed) | zero-shot, frontier API | 25.0% |
| Gemini-2.5-Pro-Thinking | (closed) | zero-shot, frontier API | 12.5% |

The v1 LoRA at HF nf4 inference (41.0%) is statistically indistinguishable from the ATLAS 70B SFT baseline (40.0%); the 95% CI on each spans the other. The base Nemotron-Nano-8B model alone is essentially incapable of producing structured HTS classifications in the v2 format (parse rate 12.5%, exact-match 0.0%), demonstrating that the LoRA fine-tune contributes essentially all the task capability. The MLX q4 production deployment configuration (36.0%) is 4pp below the ATLAS 70B baseline, inside the same N=200 confidence interval, and 11pp above the strongest zero-shot frontier baseline.

### 4.2 Results on held-out CROSS test set (N=14,952)

| Configuration | Exact match | Chapter | Heading | Subheading | Parse rate | Hierarchy consistency |
|---|---|---|---|---|---|---|
| Base Nemotron-Nano-8B, no adapter | 0.008% | 0.7% | 0.06% | 0.008% | 9.8% | 100.0% |
| v1 LoRA, HF transformers + nf4 | **58.7%** | 92.0% | 85.3% | 77.9% | 99.94% | 99.98% |

The 95% CI on 58.7% at N=14,952 is approximately ±0.8pp. The chapter-level accuracy of 92.0% indicates the model reliably identifies the broad category; most remaining errors are within-chapter disambiguation (heading or subheading level).

### 4.3 Cross-backend parity

The MLX bf16 diagnostic result (42.0% on N=50) confirms that the HF-to-MLX adapter conversion is correct: the unquantized MLX inference path produces accuracy statistically equivalent to the HF nf4 path. The 5pp gap between HF nf4 (41.0%) and MLX q4 (36.0%) is therefore attributable to quantization scheme differences, not to inference-path or conversion artifacts.

Per-level accuracy parallels the exact-match pattern: MLX bf16 essentially matches HF nf4 at all hierarchy levels, while MLX q4 shows a consistent 3-7pp decrease at every level. Parse rate and hierarchy consistency are unchanged (100% across all configurations), indicating the model continues to produce well-structured output at q4 quantization; it simply selects slightly different (often adjacent) codes about 5% of the time. This is consistent with the typical signature of quantization wobble in greedy autoregressive decoding.

### 4.4 Error decomposition

The 6,206 incorrect predictions on the held-out test set decompose as follows:

| Error type | Count | % of errors |
|---|---|---|
| right_subheading_wrong_full | 2,458 | 39.6% |
| wrong_chapter | 990 | 15.9% |
| right_heading_wrong_subheading | 942 | 15.2% |
| right_chapter_wrong_heading | 855 | 13.8% |
| missed_abstain | 528 | 8.5% |
| false_abstain | 22 | 0.4% |
| parse_failure | 9 | 0.1% |

The 116 incorrect predictions on ATLAS (excluding 2 false_abstain) decompose as follows when categorized by where the prediction first diverges from the gold:

| Position of disagreement | Count | % of errors |
|---|---|---|
| Stat suffix only (digits 9-10) | 23 | 19.8% |
| 8-digit level (digits 7-8) | 17 | 14.7% |
| 6-digit subheading | 23 | 19.8% |
| 4-digit heading | 27 | 23.3% |
| 2-digit chapter | 26 | 22.4% |

Both decompositions show that pure stat-suffix errors are a meaningful but not dominant share of failures: 19.8% of ATLAS errors and an indeterminate fraction of the 2,458 right_subheading_wrong_full bucket on the held-out test (the bucket name is misleading; it includes both 8-digit-level errors and stat-suffix errors, since "subheading" in the eval pipeline is defined at the 6-digit level).

## 5. Edge Deployment Stack

The MLX q4 configuration is the intended production deployment target. The full inference pipeline runs locally on an M4 Mac mini base (16 GB RAM, MSRP $700) with no GPU and no external API calls.

The architecture is two cooperating processes:

```
┌─────────┐   HTTP/JSON      ┌──────────────────┐  OpenAI-compat   ┌────────────────┐
│ caller  │ ───────────────► │ FastAPI wrapper  │ ───────────────► │ mlx_lm.server  │
│ (curl,  │                  │ (port 8000)      │                  │ (port 8080)    │
│  client │ ◄─────────────── │                  │ ◄─────────────── │                │
│  code)  │  {hts_code,...}  │  prompt build    │   raw text       │  MLX q4 model  │
└─────────┘                  │  output parse    │   response       │  (~4.5 GB)     │
                             └──────────────────┘                  └────────────────┘
                                                                          ▲
                                                                          │
                                                              models/nemotron-hts-fused/
                                                              (v1 LoRA fused into MLX q4 base)
```

The cold-start sequence from a fresh checkout, omitting model download time:

1. Convert the HF PEFT adapter to MLX format (seconds): `scripts/convert_hf_adapter_to_mlx.py` produces `adapters_v1_mlx/` (320 MB).
2. Fuse the LoRA into the MLX q4 base via `mlx_lm.fuse` (approximately 10 seconds, 4.5 GB output): `scripts/run_serve_mlx.py mlx-fuse`.
3. Launch `mlx_lm.server` on port 8080 (approximately 5 GB resident).
4. Launch the FastAPI wrapper on port 8000 (`scripts/run_serve_api.py`, approximately 200 MB additional resident).
5. POST a JSON-shaped classify request.

Steady-state per-request latency: 4.7 seconds for a typical 100-200 token response, measured end-to-end (HTTP request to JSON response) on an M4 Mac mini base. Memory footprint: approximately 5 GB for the MLX server and 200 MB for the FastAPI wrapper, well within the 16 GB available on the target hardware.

The FastAPI wrapper accepts a structured classification request and returns a typed JSON response with the parsed chapter, heading, subheading, full 10-digit code, reasoning text, and provides-for text. The v2 structured output format and the Nemotron-specific `<think>` empty block prefix are hidden from the caller. Parse failures from the underlying model return HTTP 200 with `parse_ok: false` and the raw text included for inspection; the wrapper never crashes on malformed model output.

## 6. Negative Result: Stat-Suffix Validation

I constructed and measured a stat-suffix validation post-processor with the goal of addressing the right_subheading_wrong_full error bucket. The procedure:

1. Walk the training data and build a map from each observed 8-digit subheading to its list of valid 10-digit completions, sorted by training-set frequency.
2. For each model prediction with an emitted 10-digit code, check whether the code is in the index. If yes, leave unchanged. If no, but the 8-digit prefix is in the index, replace the code with the most-frequent valid completion under that 8-digit prefix.

The training-derived index contains 9,791 distinct 10-digit codes across 6,107 distinct 8-digit subheadings. The procedure was applied as a post-hoc re-scoring step against the saved predictions from Section 4.

### 6.1 Measured lift

| Test set | v1 exact match | + stat-suffix validator | Lift | Records changed | Valid-10-digit rate |
|---|---|---|---|---|---|
| ATLAS (N=200) | 41.0% | 41.0% | +0.00pp | 2 / 200 (1.0%) | 96.5% |
| Held-out (N=14,952) | 58.74% | 58.78% | +0.04pp | 49 / 14,952 (0.3%) | 87.3% |

Net lift on both test sets is essentially zero. Only 0.3% of held-out predictions and 1.0% of ATLAS predictions qualified for any correction at all.

### 6.2 Why the lift is zero

The valid-10-digit rate column reveals the cause. The model produces valid HTSUS 10-digit codes (codes present in the training-derived index) in 87.3% of held-out predictions and 96.5% of ATLAS predictions. The dominant failure mode is therefore not "the model hallucinates an invalid code" but "the model emits a real code that is not the right one." Stat-suffix validation by construction can only fix invalid codes; it cannot change a wrong-but-valid prediction.

This negative result is informative about where future improvements lie. Two paths are productively closed by this measurement and one is opened:

1. **Stat-suffix validation against a more complete index (e.g., the official USITC HTSUS export covering all ~19,000 codes) would produce marginal additional lift, not transformative lift.** The bottleneck is not coverage of the validation index; it is the fact that the model rarely produces invalid codes in the first place.

2. **Constrained decoding (forcing the model to emit only codes from a known valid set) would also produce marginal lift.** Same reason. The model's emission distribution is already concentrated on valid codes; constraining it further has limited headroom.

3. **Disambiguation between valid candidate codes is the open architectural lever.** Retrieval-augmented inference, in which top-K similar CROSS rulings or HTSUS schedule entries are injected into the model's context before generation, is the natural next architecture. The Tarifflo commercial benchmark result (89.2% with a retrieval-augmented pipeline, per Judy 2024) suggests substantial available headroom from this direction.

The implementation of stat-suffix validation, the training-derived index, the post-hoc scorer, and 12 unit tests are included in the repository at `src/hts_lora/postprocess/stat_suffix.py`, `data/external/stat_suffix_index.json`, `scripts/apply_stat_suffix_validator.py`, and `tests/test_stat_suffix.py` respectively.

## 7. Cost and Reproducibility

### 7.1 Cost ledger

| Item | Cost | Notes |
|---|---|---|
| Training (single H100 SXM 80GB, RunPod) | ~$43 | 14 hours wall time |
| Evaluation suite (single H100, all 4 eval passes) | ~$30 | Held-out, base baselines, ATLAS, ATLAS base |
| Edge deployment hardware | $700 (one-time) | M4 Mac mini base, 16 GB |
| Marginal cost per inference | $0 | Local serving |
| Total to a working v1 production deployment | ~$73 software + $700 hardware | |

For comparison, the ATLAS paper used a full SFT pipeline on Llama-3.3-70B. The paper does not cost-disclose, but standard pricing for the indicated training scale places the cost in the multi-thousand-dollar range per training run, with ongoing inference requiring continued multi-GPU cluster access.

For deployment-side context: a typical U.S. importer pays $8,000 to $15,000 per year in total customs brokerage fees, with HTS classification reviews specifically running $50 to $200 per commodity (Greenwich Mercantile, 2026). The recurring cost of API-based or SaaS-based classification approaches is what kills the unit economics for a small-importer in-house tool. Removing that recurring cost by deploying a competitive model on commodity hardware is the deployment-economics unlock this paper measures.

### 7.2 Artifact availability

- **Code**: https://github.com/mfbaig35r/hts-lora
- **Trained adapter**: https://huggingface.co/mfbaig35r/hts-nemotron-8b-lora-v1
- **Base model**: `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` (publicly available)
- **MLX-quantized base**: `bourn23/nvidia-llama-3.1-nemotron-nano-8b-v1-mlx-4bit` (publicly available)
- **ATLAS reference adapter**: `flexifyai/atlas-llama3.3-70b-hts-classification` (publicly available; Yuvraj & Devarakonda, 2025)
- **Training data source**: CBP CROSS rulings (public)
- **ATLAS test set**: included at `data/external/atlas_test.jsonl`; v2 conversion script at `scripts/build_atlas_eval.py`
- **Eval reports**: `outputs/eval_*/` (predictions.jsonl, failures.jsonl, per-chapter breakdowns)
- **Session logs**: `docs/log-2026-04-06-h100-training.md` (training, $43, 14h) and `docs/log-2026-06-04-v1-eval-suite.md` (evaluation, MLX parity, edge deployment)

## 8. Limitations

**Training data distribution.** The training corpus is drawn entirely from CBP CROSS rulings. CROSS represents binding rulings, which skew toward edge cases, novel products, and contested classifications. The distribution of typical importer queries is different (more high-volume standard products, less interpretive ambiguity). The 58.7% accuracy on the held-out test set should not be read as the accuracy on real-world traffic.

**Test set size for the public benchmark.** The ATLAS test set is N=200, giving the headline 41.0% number a 95% CI of approximately ±7pp. The held-out test gives a much tighter CI (±0.8pp at N=14,952) but is in-distribution with training data; it cannot be used for cross-system comparison. Tighter public benchmarks for this task remain a community need.

**Quantization configuration matters.** The 41.0% number is for HF transformers + nf4. The 36.0% number is for MLX q4. Both are real and reported. Citations, blog posts, and reproduction attempts should be specific about which configuration is being referenced.

**Lost best checkpoint.** The shipped adapter is the final-step checkpoint, not the lowest-eval-loss checkpoint. The actually-best checkpoint (step 7000, eval loss 0.2596) was overwritten by the trainer's default save-total-limit behavior. I estimate this costs 1-3pp on exact-match accuracy. A retrain with `save_total_limit ≥ 5` and `load_best_model_at_end` would recover it; I have not done so.

**Single-stream latency only.** The 4.7-second latency figure is single-stream. Throughput under concurrent load is not benchmarked. Neither the MLX server nor the FastAPI wrapper does request batching. For an internal tool this is fine; for a public API it would need additional work.

**Training-derived stat-suffix index is incomplete.** The 9,791 codes in the validation index used in Section 6 are only those that appeared as gold answers in training. The full HTSUS schedule contains approximately 19,000 codes. For production deployment, the index should be replaced with the official USITC export; a follow-up note in the repository documents the procedure. The Section 6 conclusion does not depend on this completeness, because the bottleneck is the model's valid-code emission rate, not the index coverage.

**Zero-shot frontier comparisons are zero-shot.** As discussed in the introduction, the GPT-5-Thinking 25% and Gemini-2.5-Pro-Thinking 12.5% baselines are zero-shot configurations as measured by the ATLAS paper. Prompted, few-shot, or chain-of-thought variants of the same frontier models would presumably score higher. The deployment-economics framing developed in this paper does not depend on the zero-shot baselines being unbeatable; it depends on the recurring per-inference cost of frontier API calls being nontrivial relative to local-hardware deployment, which it is at typical importer query volumes.

## 9. Discussion

### 9.1 What this measurement implies for the field

Three claims in the existing literature on HTS classification deserve qualification in light of these results.

First, the implicit assumption in the ATLAS paper and adjacent work that parameter count is the primary axis for model-only HTS classification appears to be partly off. An 8B LoRA on a well-chosen base model with appropriate training data and a structured output format reaches the same benchmark frontier as a 70B full SFT model when both are evaluated on equivalent inference paths. The cost asymmetry between these two approaches is roughly an order of magnitude in training compute and a similar order in deployment cost.

Second, the implicit assumption in retrieval-augmented pipeline benchmarks (the Tarifflo line of work) that the model is mostly a "consumer of retrieved context" rather than a primary contributor to accuracy may be incomplete. The results here show that a small model can do most of the heavy lifting on this task without any retrieval at all. The retrieval-augmented architecture is presumably still better at the limit, but the marginal contribution of the model itself in those architectures has not been carefully measured against a strong model-only baseline.

Third, the practical implications for enterprise deployment are larger than the accuracy numbers alone suggest. A specialized model that runs on a $700 commodity device with no recurring cost, no external API dependency, and no data exfiltration risk has fundamentally different unit economics than any current SaaS or API-based HTS classification offering. The minimum viable product threshold for a customs-broker internal tool, an in-house compliance team's first-pass classifier, or an importer-side review workflow has dropped substantially.

### 9.2 Adjacent regulatory classification tasks

The deployment pattern measured in this paper plausibly transfers to several adjacent tasks with similar structural properties: a large hierarchical taxonomy, structured outputs required, gold answers in public regulatory data, operational sensitivity around the input data. Three specific examples:

**FDA medical device classification.** Devices are classified into Class I, II, or III based on intended use and risk, with a long tail of detailed product codes within each. The gold answers exist in public regulatory rulings. Recent academic work (Xu, 2022; Han et al., 2025) explores LLM-assisted classification specifically because the SaaS approach has the same data-sensitivity problem.

**India HSN tariff classification.** India's Harmonized System of Nomenclature is the Indian customs analogue to U.S. HTS. The data is gettable, the schedule is structured, regulatory updates are regular, and Indian importers have the same in-house compliance constraints as U.S. ones.

**ICD-10 medical coding.** Different input shape (gold answers come from clinical notes rather than product descriptions), different regulatory framework, but the same underlying pattern: large structured taxonomy, structured outputs needed, operational sensitivity around data, recurring per-classification cost in the existing SaaS market.

This paper does not attempt to evaluate the approach on these tasks. The argument is structural: the four pieces that made the HTS deployment work (a small-but-capable foundation model, structured public regulatory training data, an enforced output format, and a serving stack that hides prompt-format complexity) are reproducible in each of these adjacent domains.

### 9.3 Path to the next factor of improvement

Section 6 closes the door on simple post-processing as a route to substantial accuracy gains. The model already emits valid HTSUS codes at high rates; the problem is disambiguation between similar valid codes, not validation.

The natural next architectural step is retrieval-augmented inference. The specific design planned for v2 is to inject top-K most-similar CROSS rulings and the relevant HTSUS chapter notes into the model's context at inference time, with training-time exposure to the same retrieval-augmented prompts so the model learns to ground its reasoning in retrieved evidence. The Tarifflo published result (89.2% on Judy's 103-example test) suggests substantial headroom from this direction.

A parallel-track improvement is replacing the base model. Ministral-3-8B (Mistral AI, 2024) is a more recently released base with similar parameter count and improved reasoning capability. A LoRA fine-tune on the same data with this base is straightforward and should be comparable in cost; whether it produces meaningfully different accuracy is empirical.

The lost-best-checkpoint recovery is the cheapest available improvement and produces probably 1-3pp on exact match; it has not been blocking publication.

### 9.4 What's being released

The v1 adapter and all supporting code are released openly under the URLs in Section 7. The release strategy reflects a deliberate decision to enable reproduction, third-party verification, and community adoption of the small-model edge-deployment pattern for narrow regulatory tasks. The next version of this work (v2 with retrieval augmentation) is in progress; the openly released v1 will be lapped internally by approximately one generation.

## 10. Conclusion

I report v1 of an open-weight LoRA adapter for U.S. Harmonized Tariff Schedule classification. The adapter, trained on a single H100 GPU for approximately $43 in compute and 14 hours of wall time, statistically ties the published ATLAS 70B SFT benchmark on the ATLAS public test set (41.0% vs 40.0% exact match) at 1/8 the parameter count. Deployed via MLX q4 quantization on M4 Mac mini hardware, the model achieves 36.0% ATLAS exact-match with 4.7-second single-stream latency, 11 percentage points above the published GPT-5-Thinking zero-shot baseline and 23.5 percentage points above Gemini-2.5-Pro-Thinking zero-shot. On a 14,952-example held-out CROSS-derived test set, the model achieves 58.7% exact-match accuracy (95% CI ±0.8pp).

I additionally report a measured negative ablation: stat-suffix validation against a training-derived index produces no measurable accuracy lift, because the model already emits valid HTSUS codes in 87-96% of predictions. The dominant failure mode is wrong-but-valid prediction rather than invalid hallucination, indicating that future improvements should target disambiguation rather than validation. The natural next architectural step is retrieval-augmented inference; v2 work is in progress.

All code, weights, training data sources, evaluation data, edge deployment stack, and reproducibility artifacts are publicly available at https://github.com/mfbaig35r/hts-lora.

## References

Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. *Advances in Neural Information Processing Systems*. https://arxiv.org/abs/2208.07339

Greenwich Mercantile. (2026). U.S. Customs Broker Cost & Brokerage Fees 2026: $100–$250 Per Entry. https://www.greenwich-mercantile.com/resources/guides/customs-broker-cost

Han, Y., Ceross, A., & Bergmann, J. H. M. (2025). AI for Regulatory Affairs: Balancing Accuracy, Interpretability, and Computational Cost in Medical Device Classification. arXiv:2505.18695. https://arxiv.org/abs/2505.18695

Hannun, A., Digani, J., Katharopoulos, A., & Collobert, R. (2023). MLX: Efficient and Flexible Machine Learning on Apple Silicon. Software. https://github.com/ml-explore/mlx

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *International Conference on Learning Representations*. https://arxiv.org/abs/2106.09685

Judy, B. (2024). Benchmarking Harmonized Tariff Schedule Classification Models. arXiv:2412.14179. https://arxiv.org/abs/2412.14179

U.S. Customs and Border Protection. Cross Ruling (CROSS) Database. https://rulings.cbp.gov/

U.S. International Trade Commission. Harmonized Tariff Schedule. https://hts.usitc.gov/

Willard, B. T., & Louf, R. (2023). Efficient Guided Generation for Large Language Models. arXiv:2307.09702. https://arxiv.org/abs/2307.09702

Xu, Z. (2022). Using Large Pre-Trained Language Model to Assist FDA in Premarket Medical Device Classification. arXiv:2212.01217. https://arxiv.org/abs/2212.01217

Yuvraj, P., & Devarakonda, S. (2025). ATLAS: Benchmarking and Adapting LLMs for Global Trade via Harmonized Tariff Code Classification. arXiv:2509.18400. https://arxiv.org/abs/2509.18400

---

**Author note.** This preprint draft is held in the repository at `docs/paper-v1-arxiv.md` alongside a companion essay at `docs/paper-v1.md` in the canonical.agency voice. The two artifacts share an evidence base and reach identical conclusions; the essay is shorter and reaches a non-academic audience, while this preprint includes formal methodology and limitations sections. The repository at https://github.com/mfbaig35r/hts-lora and the HuggingFace adapter at https://huggingface.co/mfbaig35r/hts-nemotron-8b-lora-v1 are public as of the cited release date; reproduction does not require any additional access.

**Acknowledgments.** This work was drafted in collaboration with Claude (Anthropic). The training, evaluation, and engineering were done across two long sessions in April and June 2026; the session logs in the repository capture the full process honestly, including the bugs caught along the way.
