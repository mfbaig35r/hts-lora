# HTS LoRA Training Plan v2

## Overview

Fine-tune **Llama-3.1-Nemotron-Nano-8B** to classify products under the U.S. Harmonized Tariff Schedule using **~250k structured training examples** extracted from CBP CROSS rulings. The model learns to reason through the HTS hierarchy (chapter → heading → subheading → statistical suffix) and produce a classification with structured reasoning.

**What changed from v1**: Staged training architecture, current-valid HTS code filtering as a hard requirement, retrieval-augmented inference as a first-class design element, standardized reasoning targets, better abstention taxonomy, revised accuracy targets based on published benchmarks, and a pilot-first experimental approach.

## Competitive Landscape

Two published benchmarks exist for HTS classification:

| System | Approach | 10-digit exact | 6-digit | Test N | Source |
|--------|----------|---------------|---------|--------|--------|
| **Tarifflo** | Retrieval + ML + AI pipeline | 89.2% | — | 103 | Judy 2024 (arXiv:2412.14179) |
| **ATLAS** | Fine-tuned Llama-3.3-70B (full SFT) | 40.0% | 57.5% | 200 | Yuvraj & Devarakonda 2025 (arXiv:2509.18400) |
| **GPT-5-Thinking** | Zero-shot frontier model | 25.0% | 55.5% | 200 | ATLAS paper |
| **Avalara** | Manual expert + AI assist | 80.0% | — | 103 | Judy 2024 |
| **WCO BACUDA** | Deep learning (no LLM) | — | 12.8% (HS6) | 103 | Judy 2024 |

**Key insight**: Published results suggest that model-only fine-tuning leaves substantial headroom, and that retrieval- or workflow-augmented systems can perform much better on benchmark-style evaluations. Judy's 103-item benchmark and ATLAS's 200-item test set are useful directional signals but are not controlled ablations — we cannot attribute the full performance gap to any single factor. This supports hybrid system design as a primary research direction. Both benchmarks are statistically weak (N=103-200).

**Our advantages over ATLAS** (the only published fine-tuning approach):

| Dimension | ATLAS | Ours | Advantage |
|-----------|-------|------|-----------|
| Training examples | 18,654 | **317,606** | 17x more data |
| Unique HTS codes | 2,992 | **18,666** | 6x code coverage |
| Multi-product extraction | No | **Yes** | Captures all products per ruling |
| Structured fields | No (blob) | **Yes** (desc, materials, use, reasoning, provides-for) | Richer signal |
| Hierarchy-aware training | No (flat code) | **Yes** (chapter → code) | Teaches reasoning path |
| Semantic graph | No | **Yes** (~100k edges) | Decision boundary knowledge |
| Glossary context | No | **Yes** (~500-2000 terms) | Domain vocabulary |
| Model size | 70B (full SFT, 16× A100) | **8B** (LoRA, 1× GPU) | 10x more practical |

ATLAS derives reasoning traces through a label-conditioned transformation pipeline (GPT-4o-mini generates reasoning with the gold HTS code visible), which may make the generated reasoning more post-hoc than first-principles. Their dataset also contains 12% duplicate descriptions. Given our data volume, code coverage, structured extraction, and hierarchy-aware supervision, we expect to significantly exceed their 40% baseline.

## Staged Training Architecture

The most important architectural change from v1: **don't teach everything at once.**

An 8B LoRA adapter has limited capacity. Mixing hierarchical classification, reranking, glossary-conditioned reasoning, contrastive edge learning, and abstention in one run risks diluting the core behavior.

### Stage 1: Core Classification Adapter (this plan)

**One dominant objective**: Classify a product into the current HTS hierarchy and produce a concise, structured explanation.

| Task | Weight | Description |
|------|--------|-------------|
| `hierarchical_classify` | **90%** | Full hierarchy output with reasoning |
| `abstention` | **10%** | Structured "cannot classify" with explanation |

No rerank. No direct_classify. No contrastive edge examples. Just classification + knowing when to say no.

### Stage 2: Boundary Sharpening (future, after Stage 1 evaluation)

Continue training the Stage 1 adapter on:
- Semantic-edge contrastive cases (commonly confused code pairs)
- "Why this code and not that code" reasoning using `key_differentiator` from edges
- Ambiguous cases requiring explicit decision boundary reasoning

### Stage 3: Retrieval-Aware Rerank Adapter (future, if needed)

Reranking is a different task than classification — it depends on candidate sets and comparative judgment. Train separately if runtime evidence shows retrieval+rerank outperforms retrieval+classify.

## Data Sources

### Primary: CROSS Ruling Extractions (~317k product rows)

Extracted from ~185k CBP CROSS rulings via GPT-5.4-nano into `cross_ruling_extractions` table:

| Field | Coverage | Role |
|-------|----------|------|
| `description` | 100% | Product description (model input) |
| `hts_code` | 99.5% | Assigned HTS code (label) |
| `reasoning` | 99.4% | CBP classification reasoning (model output) |
| `hts_text` | 96.6% | Tariff schedule text — "provides for..." (model output) |
| `materials` | 91.5% | Product materials/composition (model input, optional) |
| `product_use` | 90.2% | Intended use/function (model input, optional) |
| `country` | 91.6% | Country of origin (metadata, not used in training) |

### Enrichment: HTS Hierarchy from hts-api Database

| Table | Records | Fields |
|-------|---------|--------|
| `hts_sections` | 22 | `section_number`, `description` |
| `hts_chapters` | 99 | `chapter`, `description`, `section_id` |
| `hts_headings` | ~1,600 | `heading` (4-digit), `description` |
| `hts_tariffs` | ~19,000 | `hts8` (8-digit), `description`, `mfn_rate` |
| `hts_ai_enrichments` | ~5,800 | `hts6`, `enriched_description`, `keywords` |

For each training example, join `hts_code` against this hierarchy to produce:
```
Chapter 62: Articles of apparel and clothing accessories, not knitted or crocheted
Heading 6206: Women's or girls' blouses, shirts and shirt-blouses
Subheading 6206.90: Of other textile materials
HTS Code: 6206.90.0040
```

### Glossary: Trade Term Definitions (~500-2000 terms)

The `hts_glossary` table contains multi-sense customs definitions. Customs terms often differ from common English meanings (e.g., "textile" has precise material composition rules in customs law).

**Training use**: Context injection into ~20% of training inputs when a product description mentions a customs term. Reduced from 30% (v1) to avoid model dependency on glossary presence.

### Semantic Graph: Code Relationships (~100k-500k edges)

The `hts_semantic_edges` table contains classified relationships: `commonly_confused`, `substitution`, `complementary`, `exception`, `prerequisite`.

**Training use**: Reserved for Stage 2 (boundary sharpening). Not used in Stage 1 to keep the core adapter focused.

### HTS6 Enrichments: AI-Generated Descriptions (~5,700)

**Training use**: Synthetic supplementation for thin codes. Each enrichment generates a training example where the enriched description is the input and the HTS6 code is the label.

### HTS Embeddings: Multi-Level Vectors (~38,000)

Pre-computed 1536-dim embeddings for all hierarchy levels with pgvector HNSW index.

**Training use**: Not used in LoRA training. Used at inference for retrieval-augmented candidate generation (see Production Architecture section).

## Data Preparation Pipeline

### Step 1: Export from Database

New script: `scripts/export_training_data.py`

```sql
SELECT
    e.ruling_number,
    e.description,
    e.hts_code,
    e.reasoning,
    e.hts_text,
    e.materials,
    e.product_use,
    cr.ruling_date,
    c.description AS chapter_description,
    h.description AS heading_description,
    t.description AS tariff_description
FROM cross_ruling_extractions e
JOIN cross_rulings cr USING (ruling_number)
LEFT JOIN hts_chapters c ON c.chapter = substring(
    regexp_replace(e.hts_code, '[^0-9]', '', 'g'), 1, 2)::int
LEFT JOIN hts_headings h ON h.heading = substring(
    regexp_replace(e.hts_code, '[^0-9]', '', 'g'), 1, 4)
LEFT JOIN hts_tariffs t ON t.hts8 = substring(
    regexp_replace(e.hts_code, '[^0-9]', '', 'g'), 1, 8)
WHERE e.description != '__EXTRACTION_FAILED__'
    AND e.hts_code IS NOT NULL
    AND e.reasoning IS NOT NULL
    AND length(e.description) >= 30
```

Output: `data/raw/cross_rulings_enriched.jsonl`

Additional exports for glossary (`data/raw/glossary.jsonl`) and HTS6 enrichments (`data/raw/hts6_enrichments.jsonl`).

### Step 2: Current-Valid HTS Code Filtering (NEW — HARD REQUIREMENT)

This is the highest-risk data quality issue. CROSS rulings span 30+ years. The HTS changes annually — codes get created, split, merged, and retired. Training on obsolete codes teaches the model to predict codes that don't exist.

**Process**:
1. Cross-reference every `hts_code` in the export against the current `hts_tariffs` table
2. **Keep**: Codes that exist in the current HTS (primary training set)
3. **Exclude**: Codes that no longer exist AND have no trustworthy successor mapping
4. **Map forward**: Codes with a clear successor (if a concordance is available)

**Expected impact**: ~15-20% of examples may reference obsolete codes. After filtering, the primary training set should contain only current-valid labels. This produces a smaller but cleaner label space — more valuable than a larger contaminated one.

### Step 3: Quality Filtering

- **Minimum description length**: 30 characters (not 10 as in v1)
- **Leakage detection**: Remove descriptions containing HTS codes, ruling numbers, or legal citations (extraction artifacts where the LLM copied legal text into the description)
- **Conflicting labels**: Remove exact-duplicate descriptions with different codes
- **MinHash LSH fuzzy dedup** (threshold 0.8, 128 perms): Catches near-identical extractions

### Step 4: Reasoning Normalization (NEW)

Raw CROSS reasoning varies wildly — from terse one-liners to multi-paragraph legal analyses with inconsistent style. Normalize to a consistent template structure:

1. **Identify product type** (what it is physically)
2. **Identify decisive characteristic** (material, function, feature that determines classification)
3. **Identify heading/subheading basis** (why this heading and not another)
4. **Note exclusion or differentiator** (if reasoning mentions why it's NOT another code)

**Explicit filtering policy**:
- **Keep**: Natural reasoning between 20-500 chars that references the classification basis (material, function, heading/subheading, GRI rule, chapter note)
- **Remove**: Pure legal boilerplate with no classification-relevant content
- **Remove**: Reasoning that only restates the answer without any basis (e.g., "Classified under 6109.10.0040")
- **Remove**: Reasoning that references facts absent from the description/materials/use fields — unless those facts are still present in the modeled input. This prevents the model from learning to "know" facts that were only in the original ruling text, not in what it will see at inference time
- **Truncate**: Reasoning over 500 chars — keep the first 500 chars (which typically contain the decisive logic; the tail is usually legal citation padding)

The goal is not to rewrite reasoning into a uniform template, but to ensure every surviving reasoning example contains a defensible classification basis grounded in the input the model actually receives.

### Step 5: Class Balance

| Strategy | Action |
|----------|--------|
| **Cap overrepresented codes** | Max 100 examples per code in training (uncapped in val/test) |
| **Supplement thin codes** | HTS6 enrichment examples for codes with 0-4 CROSS examples |
| **Chapter supplementation** | Minimum 50 examples per chapter from heading descriptions + enrichments (see guardrails below) |
| **Frequency-aware sampling** | During training, optional frequency-capped sampling per epoch |

**Supplementation guardrails**: Synthetic and enrichment-derived examples are not equivalent to CROSS-derived real examples. For chapters where supplementation makes up >50% of training examples, report performance separately. Keep all synthetic/enrichment examples out of val/test splits — held-out evaluation must be on real CROSS data only.

**Frequency cap audit**: When capping at 100 per code, produce an audit table showing:
- Total examples removed by cap
- Which chapters lose the most rows
- Whether the cap changes the chapter distribution materially
- Before/after code frequency histograms

This prevents the cap from fixing head-class skew while accidentally introducing chapter-level skew.

### Step 6: Abstention Examples (NEW — Improved)

Replace the v1 approach (corrupted descriptions → "I don't know") with three structured abstention categories:

**Insufficient detail**: Description lacks the material, construction, or use information needed to classify below chapter level.
```
Input: "plastic parts assorted 50kg"
Output: Cannot classify: The description does not specify the type of plastic article,
its function, or its form (sheet, tube, container, etc.). Classification requires
knowing whether these are household articles (Ch. 39), machine parts (Ch. 84),
or another category.
```

**Boundary ambiguity**: Product could fall under multiple headings depending on a missing decisive fact.
```
Input: "leather bag with textile lining"
Output: Cannot classify: Classification depends on the essential character of the bag.
If the outer surface is predominantly leather, heading 4202 applies. If textile
predominates, heading 4202 may still apply but under a different subheading. The
relative surface area of each material is needed.
```

**Missing legal discriminator**: Classification depends on a specific legal test (intended use, manufacturing method, essential character) that can't be determined from the description.
```
Input: "steel wire rope"
Output: Cannot classify: Classification depends on whether the wire rope is fitted
with fittings or made up into articles (heading 7312 vs 7326), and the specific
steel alloy composition (alloy vs non-alloy determines chapter 72 vs 73 treatment).
```

**Generation strategy**: Create abstention examples from real descriptions by systematically removing the decisive information. This produces realistic ambiguity rather than synthetic garbage.

### Step 7: Build Training Examples

Assign all examples as `hierarchical_classify` (90%) or `abstention` (10%).

**Input variation** (applied randomly to prevent the model from depending on any single format):

| Variant | Frequency | Description |
|---------|-----------|-------------|
| Rich | ~45% | Description + Materials + Use |
| Minimal | ~25% | Description only |
| Glossary-enriched | ~20% | Description + Materials + Use + Glossary definition(s) |
| Materials-only | ~10% | Description + Materials (no Use) |

### Step 8: Stratified Split

- 80/10/10 train/val/test
- Stratify by chapter
- Seed 42
- Ensure no ruling appears in both train and test (ruling-level split, not row-level)

### Step 9: Audit

- Task type distribution
- Chapter coverage (all 99 chapters must have examples in all splits)
- Code frequency distribution (head/mid/tail buckets)
- Description length distribution
- Sequence length distribution (flag any truncation at 2048)
- Current-valid code rate (should be 100% after Step 2)

## Label Granularity: 8-Digit vs 10-Digit

CROSS rulings assign codes at varying granularity — some are 8-digit, some 10-digit. Our hierarchy tables (`hts_tariffs`) are indexed at the 8-digit level (`hts8`). This creates an ambiguity that needs explicit resolution.

**Policy**: We retain the original CROSS extraction label as the supervision target, whether 8 or 10 digits. Hierarchy joins (chapter, heading, subheading) use the first 8 digits against `hts_tariffs`. Current-valid filtering also operates at 8 digits, since that is our authoritative tariff table.

**Evaluation**: We report metrics at multiple granularities:
- **8-digit exact match**: Primary metric — aligns with our tariff table and is comparable across all examples
- **10-digit exact match**: Reported separately on the subset of examples with 10-digit labels — acknowledges that statistical suffix prediction is harder and not always inferable from the description
- **6-digit, 4-digit, 2-digit**: Hierarchical metrics as specified in the evaluation section

This avoids conflating 8-digit and 10-digit performance. When we say "exact match" without qualification, we mean 8-digit.

## Training Format

### System Prompt

Lean. The model learns from examples, not from prompt bloat.

```
detailed thinking off

You are an HTS classification expert. Given a product description, classify the product under the Harmonized Tariff Schedule by identifying the correct chapter, heading, subheading, and HTS code. Provide a concise explanation based on the product's material, function, and distinguishing characteristics.

If the description does not contain enough information to classify reliably, respond with:
Cannot classify: [specific explanation of what information is missing]

Follow the output format exactly.
```

### User Message

```
Classify this product under the Harmonized Tariff Schedule:

A women's hip-length shirt made from 100% linen woven fabric, with long hemmed
sleeves, a notched collar, a full front opening secured by three buttons.

Materials: 100% linen woven fabric; sheer crinkle fabric inset on sleeves
Use: Women's casual wear
```

### Assistant Message

```
<think>
</think>

Chapter 62: Articles of apparel and clothing accessories, not knitted or crocheted
Heading 6206: Women's or girls' blouses, shirts and shirt-blouses
Subheading 6206.90: Of other textile materials
HTS Code: 6206.90.0040

Reasoning: The garment is a women's woven shirt classified under heading 6206. Because the fabric is linen (not cotton, silk, or man-made fibers), it falls under the residual "other textile materials" provision of subheading 6206.90.

Provides for: women's or girls' blouses, shirts and shirt-blouses: of other textile materials: other
```

The `<think>\n</think>\n` prefix is used to suppress chain-of-thought and force the model to go straight to the answer. NVIDIA's model documentation confirms that reasoning mode is controlled via the system prompt ("detailed thinking on/off") and that the prefilled `<think>` block prevents emergent reasoning. However, the exact interaction between system-prompt-level reasoning control and assistant-prefill during fine-tuning is not fully documented. **We will pilot two variants**: (A) reasoning-off via system prompt alone, and (B) reasoning-off with a prefilled `<think>\n</think>` assistant prefix, and retain whichever yields better format compliance and output stability.

### Abstention Output

```
<think>
</think>

Cannot classify: The description "plastic parts assorted 50kg" does not specify the type of plastic article, its function, or its form. Classification under Chapter 39 (plastics) requires knowing whether these are household articles (3924), builders' ware (3925), or other articles (3926), each with different duty rates.
```

### Chat Template (Nemotron Format)

```json
{
  "messages": [
    {"role": "system", "content": "detailed thinking off\n\n[system prompt]"},
    {"role": "user", "content": "Classify this product...\n\n[description]\n\nMaterials: [materials]\nUse: [use]"},
    {"role": "assistant", "content": "<think>\n</think>\n\nChapter XX: ...\nHeading XXXX: ...\nSubheading XXXX.XX: ...\nHTS Code: XXXX.XX.XXXX\n\nReasoning: ...\n\nProvides for: ..."}
  ]
}
```

## Training Configuration

### Pilot Run (MANDATORY before full training)

Train on 5-10k examples for 1-2 epochs to validate:
- Chat template and tokenizer alignment (no double-BOS, correct special tokens)
- Completion-only loss masking is correct (verify assistant span is non-empty, loss is zero on system/user tokens)
- `<think>\n</think>` wrapper behavior (model emits structured output, not reasoning)
- Output format parseability (>95% of outputs should parse correctly)
- Loss curve sanity (should decrease steadily)
- Sequence length distribution (confirm <1% truncation at 2048)

Also use the pilot for a **rank sweep**: train three runs with r ∈ {16, 32, 64}, holding everything else constant, and compare val loss + heading accuracy. Pick the winner for the full run.

### Model

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base model | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | 8B params, reasoning mode |
| Quantization | 4-bit NF4 (bitsandbytes) | ~5GB VRAM for base weights |
| Compute dtype | bfloat16 | |
| Attention | Flash Attention 2 | |
| Prep | `prepare_model_for_kbit_training()` | Required for QLoRA stability |

### LoRA

| Parameter | Value | Notes |
|-----------|-------|-------|
| Rank (r) | 32 (default) | Sweep {16, 32, 64} in pilot |
| Alpha | 2 × r | Convention; keep consistent during rank sweep |
| Dropout | 0.05 | |
| Target modules | All linear layers | q/k/v/o_proj, gate/up/down_proj |
| Bias | none | |
| Trainable params | ~50M at r=32 | ~0.6% of 8B |

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Epochs | 3 | May only need 1-2 given data volume |
| Batch size | 4 per device | |
| Gradient accumulation | 4 | Effective batch 16 |
| Learning rate | 2e-4 | Cosine schedule |
| Warmup | 5% of steps | |
| Min LR | 1e-5 | 5% of peak (prevents full decay) |
| Max sequence length | 2048 | Verify <1% truncation in pilot |
| Weight decay | 0.01 | |
| Optimizer | paged_adamw_8bit | Memory efficient |
| Gradient checkpointing | yes | |
| Loss masking | Completion-only | Only assistant tokens get gradient |
| Eval steps | 500 | Changed from 100 — 100 was excessive |
| Save steps | 500 | Aligned with eval |
| Logging steps | 10 | |

### Hardware

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 24GB | 48-80GB (faster, larger batch) |
| System RAM | 32GB | 64GB |
| Disk | 20GB | 50GB (with checkpoints) |
| Training time (est.) | ~8-12h on A100 80GB | ~4-6h on 4× A100 |

## Evaluation Strategy

### Accuracy Targets (Revised)

Based on the ATLAS paper (40% exact match from 18k examples, flat output, no hierarchy) and Tarifflo benchmark (89% with full retrieval pipeline):

**Model-only targets** (no retrieval at inference):

| Metric | Target | Stretch | ATLAS baseline |
|--------|--------|---------|----------------|
| Chapter match | >95% | >97% | not reported |
| Heading match | >85% | >90% | not reported |
| Subheading match | >70% | >80% | 57.5% (6-digit) |
| Exact match (10-digit) | >55% | >65% | 40.0% |
| Parse rate | >98% | >99.5% | — |
| Hierarchy consistency | >99% | >99.5% | — |

**Model + retrieval targets** (with candidate narrowing at inference):

| Metric | Target | Stretch | Tarifflo baseline |
|--------|--------|---------|-------------------|
| Exact match (10-digit) | >80% | >90% | 89.2% |

### Two-Tier Evaluation Benchmark (NEW)

**Tier 1: Rich benchmark** (1,000 items) — Full CROSS-quality descriptions with materials, use, and context. Comparable to the inputs used in published benchmarks. Stratified by chapter and code frequency.

**Tier 2: Thin benchmark** (500 items) — Short commercial descriptions like real importers write ("polyester t-shirt", "steel bolts M10", "plastic container 5L"). Manually curated to represent real-world classification difficulty. This is where commercial value lives.

### Baseline Comparisons (NEW)

Evaluate all four conditions to isolate what the adapter actually learns:

| Condition | Description |
|-----------|-------------|
| **A: Base model prompt-only** | Nemotron-8B with system prompt, no LoRA, no retrieval |
| **B: LoRA only** | Fine-tuned model, no retrieval augmentation |
| **C: Retrieval only** | Base model with top-K candidate codes in prompt |
| **D: Retrieval + LoRA** | Fine-tuned model with candidate codes in prompt |

### Metrics

| Metric | What it measures |
|--------|-----------------|
| **Chapter match** | Correct 2-digit chapter |
| **Heading match** | Correct 4-digit heading |
| **Subheading match** | Correct 6-digit subheading |
| **Exact match** | Correct full 8/10-digit code |
| **Hierarchy consistency** | Output chapter/heading/subheading matches the predicted code |
| **Parse rate** | Output successfully parsed into structured fields |
| **Hierarchical F1** | Partial credit based on ancestor overlap (captures near-misses) |
| **Head/mid/tail accuracy** | Performance by code frequency bucket |

### Error Analysis

- **Per-chapter breakdown**: Identify which chapters are hardest (expect textiles 50-63, machinery 84-85)
- **Heading confusion matrix**: Which headings get confused with each other
- **Failure mode taxonomy**: Wrong chapter (fundamental) vs wrong subheading (close) vs wrong suffix (minor)
- **Hierarchy consistency**: Does the model's emitted hierarchy match its code prediction
- **Reasoning quality**: Manual spot-check of 100 predictions — does reasoning match the classification decision
- **Head/mid/tail slices**: Report accuracy for codes with 100+, 20-99, 5-19, and 1-4 training examples

### Hard-Case Benchmark (NEW)

Beyond random held-out examples, build a curated set of ~200 hard cases:
- Commonly confused code pairs (from `hts_semantic_edges`)
- Ambiguous descriptions requiring GRI rule application
- Products from thin chapters (Ch. 1, 47, 26)
- Descriptions with deliberate vagueness (realistic commercial inputs)
- Boundary cases where material composition determines classification

This benchmark may matter more than average random test performance.

### Inferable-Exact Benchmark Slice (NEW)

Exact-match can unfairly punish the model for suffix-level distinctions that the input description never discloses. A product described as "men's cotton t-shirt" may have enough information to determine the heading (6109) and subheading (6109.10) but not the statistical suffix (.0040 vs .0065 depends on value brackets or import program).

**Policy**: Manually review a subset of ~200 test examples and label each as either:
- **Inferable**: The description + materials + use contain sufficient information to determine the exact code
- **Not inferable**: The exact code depends on facts not present in the input (value, import program, specific manufacturing method, etc.)

**Reporting**: Report exact-match accuracy separately for the inferable subset. This provides a more honest upper bound on what the model can achieve from the information it receives, and prevents the evaluation from being dominated by inherently unpredictable suffix variation.

### Validation Protocol

1. **Pilot validation**: 5-10k training run, verify format + loss + masking correctness
2. **Rank sweep**: Compare r={16, 32, 64} on heading accuracy
3. **Full training**: Monitor val metrics every 500 steps
4. **Early stopping**: Based on heading-match accuracy on val set (not just loss)
5. **Final evaluation**: Full metric suite on held-out test + hard-case benchmark
6. **Baseline comparison**: All four conditions (A/B/C/D) on both tier-1 and tier-2 benchmarks
7. **Manual review**: 100 random predictions + 50 hard-case predictions reviewed by human

## Production Architecture

Even if the fine-tune succeeds, the best commercial architecture is hybrid. The LoRA strengthens the knowledge system — it doesn't replace it.

```
┌──────────────────┐
│  User Input       │  "polyester t-shirt womens"
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 1. Normalize      │  Clean description, extract materials/use if present
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 2. Retrieve       │  Embedding similarity → top-K candidate codes
│    Candidates     │  + semantic graph expansion (related codes)
│                   │  Source: hts_embeddings + hts_semantic_edges
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 3. LoRA Classify  │  Nemotron-8B + LoRA adapter
│                   │  Input: description + materials + use + candidates
│                   │  Output: hierarchy + reasoning + provides-for
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 4. Validate       │  - Hierarchy consistency (code matches chapter/heading)
│                   │  - Code exists in current HTS
│                   │  - Output format parses correctly
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 5. Confidence     │  If model says "Cannot classify" → return with explanation
│    Gate           │  If confidence low → flag for human review
│                   │  If confident → return classification
└──────────────────┘
```

**Inference settings** (per Nemotron vendor guidance):
- Reasoning mode: OFF (prefill `<think>\n</think>`)
- Decoding: Greedy (`do_sample=False`) — deterministic output for classification
- Constrained decoding (optional): Enforce output template structure via vLLM guided decoding if format compliance drops below 98%

**LoRA serving options**:
- **Single adapter**: Merge into base model (`merge_and_unload`) — simplest, but validate quality parity with quantized base (known gotcha)
- **Adapter swapping**: Keep separate for multi-tenant serving or A/B testing — use vLLM multi-LoRA if needed

## Implementation Changes

### New Files

| File | Purpose |
|------|---------|
| `scripts/export_training_data.py` | Export enriched CROSS data + glossary + enrichments from DB |
| `scripts/build_eval_benchmarks.py` | Build tier-1 (rich) and tier-2 (thin) eval benchmarks |
| `scripts/run_pilot.py` | Pilot run with rank sweep and format validation |

### Modified Files

| File | Changes |
|------|---------|
| `configs/data.yaml` | New task weights (90/10), glossary rate 0.20, min_desc_length 30, add enriched source paths |
| `configs/train.yaml` | eval_steps 500, save_steps 500, add min_lr, add prepare_kbit flag |
| `configs/eval.yaml` | Add hierarchy_consistency, hierarchical_f1, head_mid_tail_accuracy metrics |
| `src/hts_lora/data/build_examples.py` | Add `hierarchical_classify` task type, structured abstention categories, glossary injection |
| `src/hts_lora/data/formatters.py` | Replace JSON output formatters with structured text hierarchy formatters |
| `src/hts_lora/data/ingest.py` | Add enriched JSONL loader with hierarchy fields |
| `src/hts_lora/evaluation/metrics.py` | Add hierarchy consistency, hierarchical F1, head/mid/tail slicing |

### Data Flow

```
┌──────────────────────────┐     ┌───────────────────────┐
│ cross_ruling_extractions │     │ hts_chapters/headings │
│  (Supabase PostgreSQL)   │     │ hts_tariffs           │
└───────────┬──────────────┘     └───────────┬───────────┘
            │                                │
            └──────────┬─────────────────────┘
                       ▼
              export_training_data.py
              (SQL join + current-HTS validation + JSONL export)
                       │
                       ▼
          data/raw/cross_rulings_enriched.jsonl
                       │
┌──────────────────┐   │   ┌──────────────────────┐
│ data/raw/         │   │   │ data/raw/             │
│  glossary.jsonl  │   │   │  hts6_enrichments.jsonl│
└────────┬─────────┘   │   └──────────┬────────────┘
         │             │              │
         └─────────────┼──────────────┘
                       ▼
              run_data_prep.py
              ┌────────┴────────┐
              │ ingest          │  Load JSONL
              │ filter          │  Current-valid codes, quality, dedup
              │ normalize       │  Reasoning normalization
              │ balance         │  Cap overrepresented, supplement thin
              │ build_examples  │  Assign task types, add hierarchy, abstention
              │ split           │  80/10/10 stratified by chapter (ruling-level)
              │ format          │  Chat template with hierarchy output
              │ audit           │  Statistics + quality checks
              └────────┬────────┘
                       │
                       ▼
          data/formatted/{train,val,test}.jsonl
                       │
              ┌────────┴────────┐
              ▼                 ▼
         run_pilot.py     run_train.py
         (5-10k, sweep)   (full ~200k)
              │                 │
              ▼                 ▼
         outputs/pilot/    outputs/{timestamp}/adapter/
                                │
                                ▼
                          run_eval.py
                          (hierarchical metrics, baselines, hard cases)
```

## Execution Plan

### Phase 1: Data Export and Curation (2-3 days)

1. ~~Finish CROSS extraction~~ **DONE** (~185k rulings, 317k product rows, 18.6k unique codes)
2. Clean up failed sentinel rows, retry if meaningful
3. Write `scripts/export_training_data.py`:
   - Join extractions with HTS hierarchy
   - **Current-valid HTS code filtering** (cross-reference against `hts_tariffs`)
   - Quality filtering (min 30 chars, leakage detection, dedup)
   - Reasoning length filtering (20-500 chars)
4. Export glossary and HTS6 enrichments
5. Build abstention examples (3 categories, ~10% of final dataset)
6. Run export, verify enriched JSONL quality
7. Run full data prep pipeline, review audit output

### Phase 2: Pipeline Modifications (2-3 days)

1. Update `configs/data.yaml` with new sources, task weights, parameters
2. Replace `build_examples.py` task assignment (hierarchical_classify 90%, abstention 10%)
3. Replace `formatters.py` output format (structured text hierarchy, not JSON)
4. Add structured abstention formatter
5. Add glossary injection (20% of examples)
6. Add frequency-capped sampling (cap 100 per code)
7. Update `metrics.py` (hierarchy consistency, hierarchical F1, head/mid/tail)
8. Build tier-1 and tier-2 eval benchmarks
9. Run data prep end-to-end, verify formatted output
10. Verify tokenized sequence lengths fit within 2048

### Phase 3: Pilot Training (2-3 days)

1. **Masking validation**: Train 1 step on 10 examples, verify loss is zero on system/user tokens and non-zero on assistant tokens
2. **Format validation**: Generate from untrained model with template, verify `<think>` wrapper behavior
3. **Rank sweep**: Train r={16, 32, 64} on 5-10k examples, 1-2 epochs each, compare val heading accuracy
4. **Sequence length audit**: Confirm <1% truncation at 2048
5. Pick winning rank, proceed to full run

### Phase 4: Full Training (1-2 days)

1. Train on full ~200k examples, 3 epochs (or early-stop on heading accuracy)
2. Monitor: loss curves, val metrics every 500 steps
3. Pick best checkpoint by heading-match accuracy on val set

### Phase 5: Evaluation (2-3 days)

1. Full metric suite on held-out test set
2. Four-condition baseline comparison (base, LoRA, retrieval, retrieval+LoRA)
3. Tier-1 (rich) and tier-2 (thin) benchmark evaluation
4. Hard-case benchmark evaluation
5. Per-chapter accuracy breakdown
6. Head/mid/tail performance analysis
7. Manual review of 100 random + 50 hard-case predictions
8. Error analysis and failure mode taxonomy

### Phase 6: Iteration (ongoing)

Based on eval results:
- If chapter <95%: investigate — this should be easy; check data quality or increase rank
- If heading <85%: add chapter notes as context for worst chapters, investigate confused headings
- If exact match <55%: expected for model-only; focus on retrieval pipeline (Condition D)
- If parse rate <98%: tighten output template, consider constrained decoding
- If hierarchy inconsistent: add post-processing validator, investigate training examples
- If tail codes fail badly: proceed to Stage 2 (boundary sharpening with semantic edges)

## Cost Estimate

| Item | Cost | Notes |
|------|------|-------|
| CROSS extraction | ~$150 | GPT-5.4-nano, 185k rulings (DONE) |
| Pilot training (3 rank-sweep runs) | ~$15-30 | Small runs on cloud GPU |
| Full training | ~$50-100 | Depending on GPU rental |
| Evaluation + baselines | ~$20-30 | Multiple inference runs |
| **Total** | **~$235-310** | |

## Open Questions

1. **Current-HTS concordance**: Do we have a mapping table for obsolete → current codes? If not, we exclude obsolete codes entirely rather than risk label noise. This may cut 15-20% of training data.

2. **Sequence length at 2048 vs 4096**: Pilot run will determine whether 2048 is sufficient. If >1% of examples truncate, consider 4096 (doubles memory, may need smaller batch).

3. **Reasoning-on vs reasoning-off**: The deep research report suggests ablating both modes. For Stage 1 we train reasoning-off (vendor-recommended for deterministic classification). If Stage 1 underperforms on hard cases, Stage 2 could experiment with reasoning-on.

4. **LoRA merge safety**: With 4-bit quantized base, merging can produce discrepancies. Must validate merged model output matches adapter output before deploying merged.

5. **Tier-2 benchmark sourcing**: Where do we get realistic commercial descriptions? Options: (a) manually write 500, (b) generate from product catalogs, (c) crowdsource. This is a meaningful effort but critical for honest evaluation.
