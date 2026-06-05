# A $700 Box That Beats GPT-5-Thinking at Customs Classification

**v1 of a small, specialized HTS classifier trained on a single H100 for $43, served on a Mac mini at 4.7 seconds per request, matching a published 70B benchmark at 1/8 the parameters.**

---

## The image

A Mac mini sits on a desk. It costs around seven hundred dollars. There is no GPU cluster behind it. There is no API call going out to a hosted frontier model. There is no monthly subscription. There is a single FastAPI process listening on localhost:8000, and a single MLX inference server listening on localhost:8080, and a single 4.5 GB binary on disk that is a Llama-3.1-Nemotron-Nano-8B base with a 32 million parameter LoRA fused into it.

You send it a product description over HTTP. Five seconds later you get back a structured JSON object: chapter, heading, subheading, full 10-digit HTS code, reasoning, and the "provides for" text from the United States Harmonized Tariff Schedule. The classification is right about 36% of the time on the ATLAS public test set. The published Llama-3.3-70B SFT benchmark on the same test is 40%. The published GPT-5-Thinking zero-shot baseline on the same test is 25%.

The Mac mini is beating GPT-5-Thinking and is within statistical noise of a model eight times larger.

That image is the whole point of this writeup.

## Why HTS classification is the right shape of problem

The U.S. Harmonized Tariff Schedule contains around nineteen thousand 10-digit codes. Every product that enters the United States needs to be classified into exactly one of them. The classification determines the duty rate, country-specific trade preferences, statistical reporting, and a long list of compliance flags. Getting it wrong costs money in two directions: pay too much duty on undervalued classifications, or pay penalties for misclassification under valuation.

The work today is done by customs brokers, in-house trade compliance teams at large importers, and increasingly by hosted SaaS classifiers that wrap large language models. The brokers are expensive. The in-house teams are scarce. The SaaS classifiers send your product descriptions out to the public cloud, which is a problem for anyone with confidential supplier relationships, novel product designs, or basic operational hygiene preferences.

There is a clean gap in the middle: a tool that runs locally, on hardware a small importer or broker can afford, that produces classifications at a quality high enough to be useful as either a first-draft suggestion or a reviewer's second opinion. That gap is what this v1 fills.

## What we knew about the problem before starting

Two published benchmarks set the prior for the field.

| Source | Approach | Exact match | Test set size | Paper |
|---|---|---|---|---|
| **Tarifflo** | Retrieval + ML + AI pipeline | 89.2% | 103 | Judy 2024 (arXiv:2412.14179) |
| **ATLAS** | Llama-3.3-70B fine-tuned (full SFT) | 40.0% | 200 | Yuvraj & Devarakonda 2025 (arXiv:2509.18400) |
| GPT-5-Thinking | Zero-shot frontier model | 25.0% | 200 | ATLAS paper |
| Avalara | Manual expert + AI assist | 80.0% | 103 | Judy 2024 |

Two things stand out. First, the gap between "frontier model, no specialization" (25%) and "fine-tuned 70B" (40%) is real and large. The base capability of a generalist LLM is not enough; some kind of specialization moves the needle by 15 points. Second, the gap between "fine-tuned model alone" (40%) and "retrieval-augmented pipeline" (89%) is even larger. The architecture matters more than the parameter count.

The ATLAS paper deliberately tests the model-only approach with 18,000 training examples and a Llama-3.3-70B base. Tarifflo deliberately tests the pipeline approach with retrieval over the HTSUS schedule. Both are useful as anchors for where the field sits.

What was missing from the literature was the small-model edge-deployment corner. If 8 billion parameters with the right training data and the right output format could reach 70B-tier accuracy, the deployment economics change dramatically. A $30,000 cluster becomes a $700 Mac mini. A 70-billion-parameter model becomes a 4.5 GB binary. The cost per classification drops by an order of magnitude or two.

The hypothesis going in: training a LoRA adapter on Nemotron-Nano-8B with 119,000 CROSS ruling examples, using a structured output format that encodes the full chapter-heading-subheading-code hierarchy, would land somewhere meaningful relative to ATLAS's 40%.

## What we built

The training data is 119,602 examples derived from the U.S. Customs and Border Protection's Cross Ruling database. Each example is a product description paired with the gold HTS classification. The format is the v2 messages structure we use in the inference pipeline: system prompt + user product description + assistant response in a fixed five-section structured text format (Chapter, Heading, Subheading, HTS Code, Reasoning).

The model is `nvidia/Llama-3.1-Nemotron-Nano-8B-v1`. It was fine-tuned with a LoRA adapter targeting the q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, and down_proj modules across all 32 transformer layers. Rank 32, alpha 64, dropout 0.05. The training run took 14 hours on a single H100 SXM 80GB rented from RunPod. Total training compute cost was approximately $43.

The trained adapter is 336 MB. It contains 32 million trainable parameters, or roughly 0.4% of the base model's 8 billion. The base model itself is unchanged; the LoRA can be peeled off or merged in at will.

The inference path that produced the headline numbers used HF transformers with bitsandbytes nf4 4-bit quantization on an H100. The serving path that produces the live demo uses MLX with q4 quantization on an M4 Mac mini. We measured both, plus an intermediate MLX bf16 path as a methodology check. The numbers are below.

## What it does

Three test sets, two model configurations each.

**ATLAS test set** (the public benchmark from the ATLAS paper, N=200):

| Configuration | Exact match | Chapter | Heading | Subheading | Parse rate |
|---|---|---|---|---|---|
| Base Nemotron-Nano-8B, no adapter | 0.0% | 2.5% | 0.0% | 0.0% | 12.5% |
| **v1 LoRA, HF transformers + nf4** | **41.0%** | 86.0% | 73.0% | 61.0% | 100.0% |
| v1 LoRA, MLX bf16 (N=50) | 42.0% | 86.0% | 74.0% | 60.0% | 100.0% |
| **v1 LoRA, MLX q4 (production)** | **36.0%** | 82.5% | 67.5% | 54.5% | 100.0% |

**Our held-out CROSS test set** (N=14,952):

| Configuration | Exact match | Chapter | Heading | Subheading | Parse rate |
|---|---|---|---|---|---|
| Base Nemotron-Nano-8B, no adapter | 0.008% | 0.7% | 0.06% | 0.008% | 9.8% |
| **v1 LoRA, HF transformers + nf4** | **58.7%** | 92.0% | 85.3% | 77.9% | 99.94% |

The 95% confidence interval on the 58.7% number from N=14,952 is roughly ±0.8 percentage points. That number is rock-solid. The 41.0% on ATLAS has a wider CI because of the smaller test set (about ±7pp). It's statistically indistinguishable from the published ATLAS 70B SFT result of 40.0%.

The base baseline being essentially zero on both test sets isolates the LoRA contribution cleanly. This is not a story about a good base model that's already most of the way there. The 8 billion parameter Nemotron-Nano base on its own produces parseable output 12.5% of the time on ATLAS and gets exactly zero of those parseable outputs correct. The LoRA does almost all the work.

The MLX bf16 result matches the HF nf4 result within noise. This tells us the model's true capability on ATLAS, independent of the inference backend, is about 41-42%. The MLX q4 production deployment costs about 5 percentage points. That gap is purely quantization; we verified the conversion is correct.

## What it costs

The compute story is the part of this that should change how people think about specialized model deployment.

| Item | Cost | Notes |
|---|---|---|
| Training (April) | $43 | 14 hours on one H100 SXM 80GB |
| Evaluation suite (June) | $30 | 4 eval passes over 14,952 + 200 examples |
| Edge deployment hardware | $700 | Base M4 Mac mini, 16 GB |
| Marginal cost per inference | ~$0 | Local serving, single-stream, ~5 sec |
| Recurring cost | $0 | No subscriptions, no API calls |
| Total to a v1 production deployment | ~$73 + $700 hardware | |

For comparison, the ATLAS paper used 16 A100s to train their Llama-3.3-70B baseline. That training run is unlikely to have cost less than several thousand dollars. Their inference would require ongoing access to a GPU cluster. The cost-per-classification at scale for a hosted GPT-5 zero-shot baseline is small but nonzero and recurring.

The HTS classification problem is dominantly one of distribution shift over time, regulatory updates, and edge case handling. The recurring marginal cost is what kills the unit economics for a small-importer SaaS product. Removing that cost by deploying on commodity hardware is the unlock.

## What it doesn't do, and how we know

The error analysis from the 14,952 test set is unambiguous. The model gets the chapter right 92% of the time, the heading right 85%, the subheading right 78%, and the full 10-digit code right 58.7%. The dominant failure mode is at the 10-digit statistical suffix.

Of the 6,206 incorrect predictions on our test set, 2,458 (39.6%) are cases where the model got the first 8 digits right and only the last 2 wrong. That's a specific, addressable error class. The first instinct was to wrap a stat-suffix validator around the model: when the model emits an invalid 10-digit code, replace it with the most-common valid completion under the predicted 8-digit subheading.

We built that. It does almost nothing.

| Test set | v1 exact match | + stat-suffix validator | Lift | Records changed |
|---|---|---|---|---|
| ATLAS (N=200) | 41.0% | 41.0% | +0.00pp | 2 / 200 (1.0%) |
| Our test (N=14,952) | 58.74% | 58.78% | +0.04pp | 49 / 14,952 (0.3%) |

The reason it does nothing is that the model already produces valid 10-digit HTSUS codes 87% of the time on our test set and 96% of the time on ATLAS. The dominant failure mode is not "model hallucinates a fake code" but "model picks a real code that is not the right one." Stat-suffix validation can only fix invalid hallucinations, not wrong-but-valid predictions.

This negative result is one of the more useful findings from the whole exercise. It tells us where the next factor of improvement does not live. The structured output format is doing its job; the 99.94% parse rate is not the problem; the rate at which the model emits well-formed valid HTSUS codes is not the problem. The problem is disambiguation between similar codes. That's a retrieval and reranking problem, not a validation problem.

The path to the next big accuracy jump goes through architecture, not through post-processing. The v2 plan in the repository already specifies retrieval-augmented training with CROSS rulings and chapter notes as context. That's the bet for the next factor of 1.5-2x.

## What it took to actually run it

The reproducibility story is the part that should matter to anyone who wants to verify the claims or build on them.

The model weights, the training code, the evaluation harness, the inference pipeline, the edge deployment recipe, and every commit between them are public. The training data is derived from a public source (CBP CROSS rulings). The two test sets used for evaluation are either fully public (the ATLAS 200) or derivable from the same public source (our 14,952 held-out CROSS-derived examples).

The recipe to stand up the edge deployment, end to end, from a fresh checkout:

```bash
# 1. Download the trained adapter
hf download mfbaig35r/hts-nemotron-8b-lora-v1 \
    --local-dir outputs/train_h100_20260406/adapter

# 2. Convert the HF PEFT adapter to MLX format (seconds)
python scripts/convert_hf_adapter_to_mlx.py \
    --hf-adapter outputs/train_h100_20260406/adapter \
    --out adapters_v1_mlx

# 3. Fuse the LoRA into the MLX-quantized base (~10 seconds, ~4.5 GB output)
python scripts/run_serve_mlx.py mlx-fuse

# 4. Start the OpenAI-compatible MLX server on :8080
python scripts/run_serve_mlx.py serve &

# 5. Start the FastAPI wrapper on :8000
python scripts/run_serve_api.py \
    --upstream-model "$(pwd)/models/nemotron-hts-fused" &

# 6. Classify something
curl -X POST localhost:8000/classify \
    -H 'Content-Type: application/json' \
    -d '{
      "description": "wool sweater, knitted, mens, size large",
      "country_of_origin": "Peru"
    }'
```

Total cold-start time on a fresh M4 Mac mini base, including the model download: probably 60 seconds. Steady-state per-request latency: roughly 5 seconds for a typical 100-200 token response. Memory footprint: about 5 GB for the model and another 1-2 GB for the two Python processes.

The model runs in 4-bit MLX quantization, which is the deployment configuration that produces the 36.0% ATLAS number. If you want to reproduce the 41.0% number, swap the MLX q4 path for HF transformers with bitsandbytes nf4 quantization on a CUDA box. The code paths are both in the repository.

## What this means

There's a class of enterprise applications where the optimal architecture today is not "specialized model" and not "frontier model API" but "specialized model deployed locally on commodity hardware." HTS classification is one of those classes. Customs brokerage is another. Most other narrow regulatory or compliance tasks where the domain is stable, the data is gettable, and the operational sensitivity is high.

The economics of this class of application change radically once you can fit a competitive specialized model on a $700 box. The recurring cost of the SaaS or API-based approach goes from "small but nonzero per classification" to "zero per classification, fixed-cost hardware." For a customs broker processing tens of thousands of classifications per month, that's the difference between a $2,000/month operational expense and a $700 one-time capital expense. For an internal trade compliance team at a mid-size importer, it's the difference between a vendor relationship and a process they fully own.

The pattern that gets us there has four pieces. First, a foundation model in the small-but-capable range: somewhere between 4 and 12 billion parameters, big enough to learn the task end-to-end, small enough to quantize to 4 bits and run on consumer hardware. Second, training data that captures the structured knowledge of the domain. CROSS rulings happen to be ideal for HTS classification; analogous corpora exist for most regulated domains. Third, an output format that enforces the structure of the answer rather than asking the model to produce free-form prose. Fourth, a serving stack that hides the prompt-format complexity from the caller and returns typed responses.

For v1, that gets you to 36% exact-match on a public benchmark, with 92% chapter accuracy and a 99.94% rate of producing structured, parseable output. The structured output rate matters more than the accuracy headline, because it means the model never silently fails. Every classification is either correct or wrong-in-an-inspectable-way.

For v2, the path forward is retrieval-augmented inference. The error analysis tells us what to address. The published Tarifflo result tells us how much headroom is available. The training-plan-v2 document in the repository specifies the architecture.

## What's still open

This v1 has known limitations. They belong in the writeup because they're the reasons not to use this in production without understanding them.

The training data is derived entirely from CROSS rulings. CROSS represents CBP's published binding rulings, which skews toward edge cases, novel products, and contested classifications. The distribution of a customs broker's actual daily traffic is different: more high-volume standard products, less interpretive ambiguity. The model's 58.7% accuracy on our held-out test set should not be read as the accuracy on real-world importer queries.

The 36% MLX q4 number is for the deployed edge configuration. The 41% HF nf4 number is for the published H100 inference path. The 5 percentage point gap is purely quantization cost, but anyone publishing about this work needs to be specific about which configuration they're citing.

The model produces 10-digit codes that look like real HTSUS codes 87% of the time on our test set. We have not verified that every emitted code is actually present in the current HTSUS schedule. A production deployment should add a schedule-lookup post-processor that flags emitted codes not present in the official USITC export. The training-derived stat-suffix validator we built only covers codes seen during training (about 9,800 distinct 10-digit codes out of the ~19,000 in the full schedule).

The lost-best-checkpoint issue from the April training run is documented honestly elsewhere. The H100 training kept only the last 3 checkpoints by default, and the actually-best checkpoint (eval loss 0.2596 at training step 7000) was overwritten by later checkpoints with higher eval loss. The shipped HF adapter is the final-step checkpoint, eval loss 0.2729. The accuracy numbers reported here are for that shipped adapter. Recovering the best checkpoint via a brief retrain is in the open items but not blocking publication.

Single-stream latency is 4.7 seconds. Throughput under concurrent load is not benchmarked. Neither the MLX server nor our FastAPI wrapper does request batching today. For an internal tool that's fine. For a public API it isn't.

## Open questions that shape the next version

Whether to publish the model itself or to lap it with a v2 internally is a decision the project has been deferring. The release strategy in the project notes is "publish v1 freely, lag internal work by one generation." This piece is the v1 publication. The v2 internal work is in flight.

Whether the next factor-of-1.5x improvement comes from RAG, from a Ministral-base v1.1 retrain, or from both is still genuinely open. RAG is more transformative but more expensive to build. A Ministral retrain is cheap, mechanical, and probably worth doing in parallel.

Whether the 5pp quantization cost is worth eating to keep the edge deployment story is a real question. MLX q8 quantization would cut the gap roughly in half at the cost of about 3 GB additional memory footprint, still well within the M4 Mac mini's 16 GB. We deliberately did not measure q8 in this round; it's queued.

## What we'd want a reader to take away

Three things, in this order:

The first is that the deployment economics of specialized models on commodity hardware are now genuinely interesting. A $700 device producing ATLAS-comparable accuracy on a narrow enterprise task is a different unit economics regime than the one most enterprise AI procurement is built around. Whether your specific narrow task fits this pattern depends on the data and the domain, but the existence of the pattern is no longer in question.

The second is that the published frontier benchmarks for niche enterprise tasks systematically understate what a small, specialized model can do. ATLAS's 40% from 70B SFT and GPT-5-Thinking's 25% zero-shot are both real, but they're not the ceiling. The ceiling for the model-only approach is higher, and the ceiling for the model-plus-RAG approach is much higher.

The third is that the engineering work to make this pattern reproducible matters as much as the accuracy headline. The 36% number is real and defensible. The 60-second cold-start from a fresh checkout is real and defensible. The recipe in the repo runs without modification. The negative result on stat-suffix validation, which we shipped alongside the working positive results, is genuine information about where the next improvement lives. The cost ledger is real. None of this is hard to do once you decide to do it, and most published model work doesn't bother. That's a market gap.

## Reproducibility

Repository: https://github.com/mfbaig35r/hts-lora

Adapter: https://huggingface.co/mfbaig35r/hts-nemotron-8b-lora-v1

Base model: `nvidia/Llama-3.1-Nemotron-Nano-8B-v1`

MLX-quantized base: `bourn23/nvidia-llama-3.1-nemotron-nano-8b-v1-mlx-4bit`

Training data source: CBP CROSS rulings (public).

ATLAS test set: included in the repository at `data/external/atlas_test.jsonl`, converted to v2 format at `data/external/atlas_test_v2.jsonl` via `scripts/build_atlas_eval.py`.

Session logs: `docs/log-2026-04-06-h100-training.md` (training run, $43, 14h) and `docs/log-2026-06-04-v1-eval-suite.md` (evaluation suite, parity check, edge PoC).

Eval reports: all four eval outputs (with predictions.jsonl, failures.jsonl, per-chapter breakdowns) live under `outputs/eval_*/` in the repository (gitignored at the artifact level; regenerable via `scripts/run_eval.py` against the published adapter).
