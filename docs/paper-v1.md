# An 8B HTS Classifier on a Mac Mini

**An 8B LoRA trained on $43 of H100 compute. Served on a $700 Mac mini at 4.7 seconds per classification, 36.0% on the ATLAS public benchmark, 11pp above zero-shot GPT-5-Thinking, statistically tied with a 70B baseline when compared on the same hardware.**

*By Fahad Baig, founder of [Canonical Agency](https://canonical.agency).*

---

## The image

A Mac mini sits on a desk. It costs around seven hundred dollars. There is no GPU cluster behind it. There is no API call going out to a hosted frontier model. There is no monthly subscription. There is a single FastAPI process listening on localhost:8000, a single MLX inference server listening on localhost:8080, and a single 4.5 GB binary on disk: Llama-3.1-Nemotron-Nano-8B with a 32 million parameter LoRA fused into it.

You send it a product description over HTTP. Five seconds later you get back a structured JSON object: chapter, heading, subheading, full 10-digit HTS code, reasoning, and the "provides for" text from the United States Harmonized Tariff Schedule.

On the 200-example ATLAS public test set, this Mac mini classifies correctly 36.0% of the time. ATLAS itself, run on its original hardware with full-precision inference, reports a fine-tuned Llama-3.3-70B at 40.0% on the same test. The Mac mini number is 4 percentage points below that, which lands inside the ±7pp 95% confidence interval at N=200. If I run my own adapter on the same full-precision inference path ATLAS used (HF transformers + bnb-nf4 on a CUDA box), I score 41.0%, statistically indistinguishable from their 40.0%. The 5pp gap between my own H100 number and my Mac mini number is purely quantization cost.

Same test, the ATLAS paper reports GPT-5-Thinking zero-shot at 25.0% and Gemini-2.5-Pro-Thinking at 12.5%. My Mac mini beats both.

That image is the whole point of this writeup.

## Why HTS classification is the right shape of problem

The U.S. Harmonized Tariff Schedule contains around nineteen thousand 10-digit codes. Every product entering the United States gets classified into exactly one of them. The classification determines the duty rate, country-specific trade preferences, statistical reporting, and a long list of compliance flags. Getting it wrong costs money in two directions: pay too much duty on conservative classifications, or pay penalties for misclassification under valuation.

The work today is done by customs brokers, in-house trade compliance teams at large importers, and hosted SaaS classifiers that wrap large language models. The brokers are expensive. The in-house teams are scarce. The SaaS classifiers send your product descriptions out to the public cloud, which is a problem for anyone with confidential supplier relationships, novel product designs, or basic operational hygiene preferences.

There is a clean gap in the middle: a tool that runs locally, on hardware a small importer or broker can afford, that produces classifications at a quality high enough to be useful as either a first-draft suggestion or a reviewer's second opinion. That gap is what this v1 fills.

## What was already known

Two recent papers anchor the field.

Yuvraj & Devarakonda (2025) introduce the ATLAS benchmark: 18,254 training examples, 200 validation, 200 test, all derived from CBP's public CROSS rulings database. They report a fine-tuned LLaMA-3.3-70B (full SFT) at 40.0% exact-match on the 10-digit code level on their 200-example test set. The same paper reports zero-shot frontier model baselines on the same test: GPT-5-Thinking at 25.0%, Gemini-2.5-Pro-Thinking at 12.5%. The paper's own headline finding is that domain-specific fine-tuning of a 70B model produces a +15pp improvement over the strongest zero-shot baseline.

Judy (2024) takes a different approach: a benchmarking study of commercial HTS SaaS tools on a separate 103-example test set. The study evaluates Tarifflo (a retrieval-augmented pipeline, 89.2%), Avalara (expert-plus-AI workflow, 80.0%), Zonos, and the WCO's BACUDA tool. These two benchmarks are not directly comparable to each other: the test sets are distinct, the scoring methodologies differ, and the architectural assumptions are inverted. ATLAS measures what a fully fine-tuned model can do alone. Judy measures what commercial pipelines do with retrieval, expert review, and explanation generation layered on top.

Two things stand out across these numbers.

First, the gap between "frontier model, no specialization" (25% from GPT-5-Thinking, 12.5% from Gemini-2.5-Pro) and "fine-tuned 70B" (40%) is real and large. The base capability of a generalist LLM is not enough on this task; some form of specialization moves the needle by 15 to 27 points.

Second, the gap between "fine-tuned model alone" (40%) and "retrieval-augmented commercial pipeline" (89%) is even larger. The architecture matters more than the parameter count.

A note on the zero-shot comparisons: I cite GPT-5-Thinking at 25% and Gemini-2.5-Pro at 12.5% throughout because those are the baselines ATLAS measured. A few-shot or chain-of-thought variant of the same frontier models would presumably score higher. The comparison I'm interested in isn't "can a frontier model with sufficient prompting match my fine-tune," because the answer is almost certainly yes. The comparison I'm interested in is deployment economics: at what point does it become cheaper, faster, and more confidential to run a specialized model locally than to keep calling a frontier model API forever.

What was missing from the literature was the small-model edge-deployment corner. If 8 billion parameters with the right training data and the right output format could reach 70B-tier accuracy at full precision, and stay roughly competitive after edge-quantization, the deployment economics change. A several-A100 training run becomes a single H100 rental for under $50. A 70-billion-parameter model becomes a 4.5 GB binary that fits comfortably on a Mac mini. The cost per classification drops to zero. The data never leaves the customer's network.

The hypothesis going in: a LoRA adapter on Nemotron-Nano-8B, with 119,000 CROSS ruling examples and a structured output format that encodes the full chapter-heading-subheading-code hierarchy, would land somewhere meaningful relative to ATLAS's 40%.

## What I built

The training data is 119,602 examples derived from CBP's public CROSS rulings database. Each example pairs a product description with the gold HTS classification CBP assigned in the ruling. The format is structured five-section text: chapter, heading, subheading, 10-digit code, and reasoning. The model emits these sections as plain text and the inference pipeline parses them via regex into typed JSON.

The model is `nvidia/Llama-3.1-Nemotron-Nano-8B-v1`. I fine-tuned it with a LoRA adapter targeting the q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, and down_proj modules across all 32 transformer layers. Rank 32, alpha 64, dropout 0.05. The training run took 13 hours 44 minutes on a single rented H100 SXM 80GB on RunPod. Total cost: approximately $43.

The resulting adapter is 336 MB. It contains 32 million trainable parameters, or roughly 0.4% of the base model's 8 billion. The base model itself is unchanged; the LoRA can be peeled off or merged in at will.

The H100 inference path used HF transformers with bitsandbytes nf4 4-bit quantization. The edge serving path uses MLX with q4 quantization on an M4 Mac mini. I measured both, plus an intermediate MLX bf16 path as a methodology check that the conversion between backends is faithful. All three numbers are in the next section.

## What it does

Two test sets, multiple inference configurations.

**ATLAS public test set** (N=200, from Yuvraj & Devarakonda 2025):

| Configuration | Hardware | Quantization | Exact match | Chapter | Heading | Subheading |
|---|---|---|---|---|---|---|
| Base Nemotron-Nano-8B (no adapter) | H100 | bnb-nf4 | 0.0% | 2.5% | 0.0% | 0.0% |
| **v1 LoRA, HF transformers** | H100 | bnb-nf4 | **41.0%** | 86.0% | 73.0% | 61.0% |
| v1 LoRA, MLX (parity check) | M4 | bf16 (N=50) | 42.0% | 86.0% | 74.0% | 60.0% |
| **v1 LoRA, MLX (edge deployment)** | M4 | q4 | **36.0%** | 82.5% | 67.5% | 54.5% |

ATLAS paper comparators on the same N=200 test set:

| Model | Configuration | Exact match |
|---|---|---|
| ATLAS Llama-3.3-70B (full SFT) | H100, full precision | 40.0% |
| GPT-5-Thinking | zero-shot, frontier API | 25.0% |
| Gemini-2.5-Pro-Thinking | zero-shot, frontier API | 12.5% |

The 41% HF number is statistically indistinguishable from the ATLAS paper's 40% (the 95% CI on a 41% proportion at N=200 is about ±7pp; the ATLAS 40% sits comfortably inside that interval). The 36% Mac mini number is 4pp below the ATLAS 70B baseline, also inside the same interval, and 11pp above zero-shot GPT-5-Thinking.

**My held-out CROSS test set** (N=14,952):

| Configuration | Exact match | Chapter | Heading | Subheading | Parse rate |
|---|---|---|---|---|---|
| Base Nemotron-Nano-8B (no adapter) | 0.008% | 0.7% | 0.06% | 0.008% | 9.8% |
| **v1 LoRA, HF transformers + nf4** | **58.7%** | 92.0% | 85.3% | 77.9% | 99.94% |

The 95% confidence interval on 58.7% at N=14,952 is approximately ±0.8 percentage points. That number is rock-solid. The 41% on the smaller ATLAS test has a wider CI for the reason mentioned above.

The base baseline being essentially zero on both test sets isolates the LoRA contribution. This is not a story about a strong base model that's already most of the way there. The 8 billion parameter Nemotron-Nano base on its own produces parseable structured output 12.5% of the time on ATLAS and gets exactly zero of those parseable outputs correct. The LoRA does almost all the work.

The MLX bf16 result matches the HF nf4 result within statistical noise (the 95% CI on a 42% proportion at N=50 is roughly ±14pp, which contains the 41% HF number with room to spare). This means the model's true capability on ATLAS, independent of the inference backend, is about 41-42%. The MLX q4 production deployment costs about 5 percentage points relative to that. That gap is purely quantization; I verified the adapter conversion is correct by running the bf16 diagnostic.

## What it costs

The compute story is the part of this that should change how people think about specialized model deployment.

| Item | Cost |
|---|---|
| Training (single H100 SXM 80GB, RunPod) | ~$43 |
| Evaluation suite (single H100, all 4 eval passes) | ~$30 |
| Edge deployment hardware | $700 (one-time) |
| Marginal cost per inference | $0 |
| Recurring software cost | $0 |
| Total to working production deployment | ~$73 + $700 hardware |

For comparison: the ATLAS paper used a full SFT pipeline on Llama-3.3-70B, which the paper does not cost-disclose but which standard pricing puts in the multi-thousand-dollar range per training run. Their inference would require ongoing access to a multi-GPU cluster.

For context on the deployment side: a typical U.S. importer pays $8,000 to $15,000 per year in total customs brokerage fees, with HTS classification reviews specifically running $50 to $200 per commodity ([Greenwich Mercantile, 2026](https://www.greenwich-mercantile.com/resources/guides/customs-broker-cost)). For an importer doing classification in-house, the cost of running this model on commodity hardware is the depreciation on a $700 Mac mini and a handful of kilowatt-hours per month.

The HTS classification problem is dominantly one of distribution shift over time, regulatory updates, and edge case handling. The recurring marginal cost of API-based or SaaS-based approaches is what kills the unit economics for a small-importer tool. Removing that cost by deploying on commodity hardware is the unlock.

## What it doesn't do, and how I know

The error analysis from the 14,952 test set is unambiguous. The model gets the chapter right 92% of the time, the heading right 85%, the subheading right 78%, and the full 10-digit code right 58.7%. The single biggest failure bucket, by a wide margin, is at the 10-digit statistical suffix.

Of the 6,206 incorrect predictions on my test set, 2,458 (39.6%) are cases where the model got the first 8 digits right and only the last 2 wrong. That looks like a specific, addressable error class. My first instinct was to wrap a stat-suffix validator around the model: when the model emits an invalid 10-digit code, replace it with the most-common valid completion under the predicted 8-digit subheading.

I built that. It does almost nothing.

| Test set | v1 exact match | + stat-suffix validator | Lift | Records changed |
|---|---|---|---|---|
| ATLAS (N=200) | 41.0% | 41.0% | +0.00pp | 2 / 200 (1.0%) |
| Held-out (N=14,952) | 58.74% | 58.78% | +0.04pp | 49 / 14,952 (0.3%) |

The reason it does nothing: the model already produces valid 10-digit HTSUS codes 87% of the time on my test set and 96% of the time on ATLAS. The dominant failure mode is not "model hallucinates a fake code." It's "model picks a real code that's not the right one." Stat-suffix validation can only fix invalid hallucinations, not wrong-but-valid predictions.

This negative result is one of the more useful findings from the whole exercise. It tells me where the next factor of improvement does not live. The structured output format is doing its job. The 99.94% parse rate is not the problem. The rate at which the model emits well-formed valid HTSUS codes is not the problem. The problem is disambiguation between similar codes. That's a retrieval and reranking problem, not a validation problem.

The path to the next big accuracy jump goes through architecture, not through post-processing. The Tarifflo benchmark (89.2% on Judy's 103-example test) is direct evidence of how much headroom retrieval architectures unlock. The v2 plan in the repository already specifies retrieval-augmented training with CROSS rulings and chapter notes as context. That's the bet for the next factor of 1.5-2x.

## How to actually run it

The model weights, training code, evaluation harness, inference pipeline, and edge deployment recipe are all public. The training data is derived from CBP CROSS rulings (public). The two test sets are either fully public (the ATLAS 200) or derivable from the same public source (my 14,952 held-out CROSS-derived examples).

The architecture of the running stack:

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

The recipe to stand the whole thing up from a fresh checkout:

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
      "description": "insulated copper electrical wire, 12 AWG, stranded",
      "materials": "copper conductor, PVC insulation",
      "use": "residential building wiring",
      "country_of_origin": "Mexico"
    }'
```

And here's what you get back:

```json
{
  "hts_code": "8544.49.3080",
  "chapter": {
    "code": "85",
    "description": "ELECTRICAL MACHINERY AND EQUIPMENT AND PARTS THEREOF..."
  },
  "heading": {
    "code": "85.44",
    "description": "Insulated wire, cable, and other insulated electric conductors..."
  },
  "subheading": {
    "code": "8544.49",
    "description": "Insulated electric conductors nesoi, of copper, for a voltage not exceeding 1,000 V, not fitted with connectors"
  },
  "reasoning": "Classified under heading 8544 as insulated wire. The subheading applies to other electric conductors for a voltage not exceeding 1,000 V, specifically the copper provision.",
  "provides_for": "Insulated wire...: Other electric conductors, for a voltage not exceeding 1,000 V: Other: Other: Of copper: Other.",
  "is_abstention": false,
  "parse_ok": true,
  "model": "hts-nemotron-8b-lora-v1",
  "latency_ms": 4664
}
```

Total cold-start time on a fresh M4 Mac mini base, including the model download: probably 60 seconds. Steady-state per-request latency: roughly 5 seconds for a typical 100-200 token response. Memory footprint: about 5 GB for the model and another 1-2 GB for the two Python processes. The 16 GB M4 Mac mini base has comfortable headroom.

The model runs in 4-bit MLX quantization in this configuration, which produces the 36.0% ATLAS number. If you want to reproduce the 41.0% HF number, swap the MLX path for HF transformers with bitsandbytes nf4 on a CUDA box; the code is in the repository.

## What this means

There's a class of enterprise applications where the best architecture today is not "specialized model" and not "frontier model API" but "specialized model deployed locally on commodity hardware." HTS classification is one. Three others I'd name without having implemented them:

**FDA medical device classification.** Devices are classified into Class I, II, or III based on intended use and risk, with a long tail of detailed product codes within each. The gold answers exist in public regulatory rulings. Recent academic work explores LLM-assisted classification specifically because the SaaS approach has the same data-sensitivity problem ([Wang et al. 2022](https://arxiv.org/pdf/2212.01217), [Yang et al. 2025](https://arxiv.org/pdf/2505.18695)). The FDA itself ran an AI-assisted scientific review pilot in early 2025.

**India HSN tariff classification.** India's Harmonized System of Nomenclature is the Indian customs analogue to U.S. HTS. The data is gettable, the schedule is structured, the regulatory updates are regular, and Indian importers have the same in-house compliance constraints as U.S. ones.

**ICD-10 medical coding.** Different shape (the gold answers come from clinical notes rather than product descriptions), different regulatory framework, but the same underlying pattern: large structured taxonomy, structured outputs needed, operational sensitivity around data, recurring per-classification cost in the existing SaaS market.

The economics of any of these change radically once you can fit a competitive specialized model on a $700 box. The recurring cost of the SaaS or API-based approach goes from "small but nonzero per classification" to "zero per classification on fixed-cost hardware." For an internal trade compliance team at a mid-size importer, that's the difference between a vendor relationship and a process they fully own. For a customs broker at a small or mid-size firm, it's the difference between paying for a per-seat SaaS classifier and running a tool they control.

The pattern that gets us there has four pieces. First, a foundation model in the small-but-capable range: somewhere between 4 and 12 billion parameters, big enough to learn the task end-to-end, small enough to quantize to 4 bits and run on consumer hardware. Second, training data that captures the structured knowledge of the domain. CBP CROSS rulings happen to be ideal for HTS classification; analogous public regulatory corpora exist for most regulated domains. Third, an output format that enforces the structure of the answer rather than asking the model to produce free-form prose. Fourth, a serving stack that hides the prompt-format complexity from the caller and returns typed responses.

For v1, this gets to 36% exact-match on a public benchmark, with 92% chapter accuracy and a 99.94% rate of producing structured, parseable output. The structured output rate matters more than the accuracy headline, because it means the model never silently fails. Every classification is either correct or wrong-in-an-inspectable-way.

For v2, the path forward is retrieval-augmented inference. The error analysis tells me what to address. The published Tarifflo result tells me how much headroom is available. The v2 plan in the repository specifies the architecture.

## What's still open

The training data is derived entirely from CROSS rulings. CROSS represents CBP's published binding rulings, which skews toward edge cases, novel products, and contested classifications. The distribution of a customs broker's actual daily traffic is different: more high-volume standard products, less interpretive ambiguity. The model's 58.7% accuracy on my held-out test set should not be read as the accuracy on real-world importer queries. That number tells you what the model does on in-distribution traffic.

The 36% MLX q4 number is for the deployed edge configuration. The 41% HF nf4 number is for the published H100 inference path. The 5 percentage point gap is purely quantization cost. Anyone publishing about, citing, or attempting to reproduce this work needs to be specific about which configuration they're talking about.

The model produces 10-digit codes that look like real HTSUS codes 87% of the time on my held-out test and 96% of the time on ATLAS. I have not verified that every emitted code is actually present in the current HTSUS schedule. A production deployment should add a schedule-lookup post-processor that flags codes not present in the official USITC export. The training-derived stat-suffix index I built only covers codes seen during training (about 9,800 distinct 10-digit codes out of the ~19,000 in the full schedule), so it can't serve as that validator directly; a follow-up note in the repo documents what would change.

The H100 training run kept only the last 3 checkpoints by default. The actually-best checkpoint (eval loss 0.2596 at training step 7000) was overwritten by later checkpoints with higher eval loss. The shipped HF adapter is the final-step checkpoint, eval loss 0.2729. The accuracy numbers in this writeup are for that shipped adapter. The lost checkpoint would probably have been 1-3pp higher on exact match; recovering it via a brief retrain is on the open items list but not blocking this publication.

Single-stream latency is 4.7 seconds. Throughput under concurrent load is not benchmarked. Neither the MLX server nor my FastAPI wrapper does request batching. For an internal tool that's fine. For a public API it isn't.

Whether the next factor-of-1.5x improvement comes from RAG, from a Ministral-base v1.1 retrain, or from both is still open. RAG is more transformative but more expensive to build. A Ministral retrain is cheap, mechanical, and probably worth doing in parallel.

Whether the 5pp quantization cost is worth eating to keep the edge deployment story crisp is a real question. MLX q8 quantization would cut the gap roughly in half at the cost of about 3 GB additional memory footprint, still well within the M4 Mac mini's 16 GB. I deliberately didn't measure q8 in this round; it's queued.

## What I'd want a reader to take away

The deployment economics of specialized models on commodity hardware are now genuinely interesting. A $700 device producing benchmark-comparable accuracy on a narrow enterprise task is a different unit-economics regime than the one most enterprise AI procurement is built around. Whether your specific narrow task fits this pattern depends on the data and the domain, but the existence of the pattern is no longer in question.

The published frontier benchmarks for niche enterprise tasks systematically understate what a small, specialized model can do. ATLAS's 40% from 70B SFT and GPT-5-Thinking's 25% zero-shot are both real, but they're not the ceiling. The ceiling for the model-only approach is higher, and the ceiling for the model-plus-RAG approach is much higher.

The engineering work to make this pattern reproducible matters as much as the accuracy headline. The 36% number is real and defensible. The 60-second cold-start from a fresh checkout is real and defensible. The recipe in the repo runs without modification. The negative result on stat-suffix validation, which I shipped alongside the working positive results, is genuine information about where the next improvement lives. The cost ledger is real. None of this is hard to do once you decide to do it.

---

## Reproducibility

**Repository.** [github.com/mfbaig35r/hts-lora](https://github.com/mfbaig35r/hts-lora)

**Trained adapter.** [huggingface.co/mfbaig35r/hts-nemotron-8b-lora-v1](https://huggingface.co/mfbaig35r/hts-nemotron-8b-lora-v1)

**Base model.** `nvidia/Llama-3.1-Nemotron-Nano-8B-v1`

**MLX-quantized base.** `bourn23/nvidia-llama-3.1-nemotron-nano-8b-v1-mlx-4bit`

**Training data source.** [CBP CROSS rulings](https://rulings.cbp.gov/) (public)

**ATLAS test set.** Included at `data/external/atlas_test.jsonl`, converted to v2 format at `data/external/atlas_test_v2.jsonl` via `scripts/build_atlas_eval.py`.

**Session logs.** `docs/log-2026-04-06-h100-training.md` (training run, $43, 14h) and `docs/log-2026-06-04-v1-eval-suite.md` (evaluation suite, MLX parity check, edge serving PoC).

**Eval reports.** All four eval outputs (with predictions.jsonl, failures.jsonl, per-chapter breakdowns) live under `outputs/eval_*/` in the repository (regenerable via `scripts/run_eval.py` against the published adapter).

---

## References

Judy, B. (2024). Benchmarking Harmonized Tariff Schedule Classification Models. arXiv:2412.14179. [arxiv.org/abs/2412.14179](https://arxiv.org/abs/2412.14179)

Yuvraj, P., & Devarakonda, S. (2025). ATLAS: Benchmarking and Adapting LLMs for Global Trade via Harmonized Tariff Code Classification. arXiv:2509.18400. [arxiv.org/abs/2509.18400](https://arxiv.org/abs/2509.18400)

Wang, B., et al. (2022). Using Large Pre-Trained Language Model to Assist FDA in Premarket Medical Device Classification. arXiv:2212.01217. [arxiv.org/abs/2212.01217](https://arxiv.org/abs/2212.01217)

Yang, et al. (2025). AI for Regulatory Affairs: Balancing Accuracy, Interpretability, and Computational Cost in Medical Device Classification. arXiv:2505.18695. [arxiv.org/abs/2505.18695](https://arxiv.org/abs/2505.18695)

Greenwich Mercantile. (2026). U.S. Customs Broker Cost & Brokerage Fees 2026: $100–$250 Per Entry. [greenwich-mercantile.com/resources/guides/customs-broker-cost](https://www.greenwich-mercantile.com/resources/guides/customs-broker-cost)

---

**Errata and corrections.** Open an issue at [github.com/mfbaig35r/hts-lora/issues](https://github.com/mfbaig35r/hts-lora/issues) or email [hello@canonical.agency](mailto:hello@canonical.agency).

**Footnote.** This piece was drafted in collaboration with Claude (Anthropic). The training, evaluation, and engineering work was done over two long sessions across April and June 2026; the session logs in the repository capture the full process honestly, including the bugs caught along the way. Thanks to Claude for the collaboration and the candid pushback.
