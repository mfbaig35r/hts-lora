# HTS LoRA v2 — Training Requirements

Requirements for the next iteration of the HTS classifier model, `hts-nemotron-8b-lora-v2`.

## Context

v1 (`mfbaig35r/hts-nemotron-8b-lora-v1`) is trained, serving, and working end-to-end
behind a local FastAPI wrapper. It demonstrates that LoRA fine-tuning of
`nvidia/Llama-3.1-Nemotron-Nano-8B-v1` on ~149k CROSS ruling examples produces a
model that reliably emits structured HTS classifications and runs on a MacBook.

However, four-sample live testing and the training data composition both point at
a clear ceiling. The v1 model:

- Recalls **topical** associations (product X → chapter Y) well
- Misses **structural** distinctions (parts vs finished, GRI, post-2022 splits)
- Hallucinates 10-digit statistical reporting numbers by zero-padding
- Fabricates CROSS ruling citations in reasoning prose
- Has no grounding — cannot cite or verify anything

v2's purpose is to move the model from *memorized pattern matching* to *grounded,
structurally-aware classification*. The highest-leverage changes are to the
training data mix, the output format, and the training signal. The base model,
LoRA architecture, and serving infrastructure do not need to change.

## Goals

### Quantitative targets (on a held-out test set of ≥2,000 examples)

| Metric | v1 (estimated) | v2 target |
|---|---|---|
| Chapter accuracy | 85–90% | ≥ 95% |
| Heading accuracy | 70% | ≥ 85% |
| Subheading accuracy | 50% | ≥ 70% |
| HTS code exact match (full) | 15–25% | ≥ 50% |
| Valid 10-digit rate (when emitted) | ~40% | ≥ 90% |
| Format compliance (parse_ok) | ~95% | ≥ 99% |
| Fabricated citation rate | high, unmeasured | 0% |
| Abstention precision | unmeasured | ≥ 95% |
| Abstention recall | unmeasured | ≥ 70% |
| HTS 2022-split slice accuracy | unknown, likely low | ≥ 75% |

### Qualitative goals

- Model can cite real rulings and chapter notes from retrieved context
- Model emits partial commitments (6-digit or 4-digit only) when stat suffix is
  genuinely indeterminate from the input, rather than fabricating zeros
- Model correctly handles post-2022 HTSUS schedule changes
- Model applies the General Rules of Interpretation (GRI) when classification
  is ambiguous between multiple competing headings

## Non-goals

The following are explicitly **out of scope** for v2:

- Production-grade customs broker replacement
- Real-time HTSUS schedule updates (annual snapshot is fine)
- Multi-language support
- End-user-facing explanation ("why did you pick this code" for laypeople)
- User feedback loops / RLHF from production traffic
- Swapping the base model — still Llama-3.1-Nemotron-Nano-8B-v1
- Full fine-tune instead of LoRA — still LoRA
- Moving serving off MLX — still local MLX + FastAPI wrapper
- Multi-modal (image) inputs

---

## Requirements

### R1 — Training Data Composition

**R1.1** The training data mix shall be approximately:
  - 60% CROSS rulings (curated, deduped, preference toward post-2022)
  - 20% HTSUS schedule text (chapter, heading, subheading definitions)
  - 10% Chapter notes and section notes, paired with worked examples
  - 5% GRI 1–6 worked examples
  - 5% Disambiguation negatives (correct vs common-wrong pairs)

**R1.2** Total training examples shall be between 100k and 200k after
deduplication.

**R1.3** All CROSS ruling examples shall be tagged with the HTSUS schedule
version that was in force when the ruling was issued. Rulings predating the
2022 WCO update shall be labeled `schedule_version: "pre-2022"`.

**R1.4** Deduplication shall use `(hts_code, normalized_description)` with an
embedding-similarity threshold of 0.95. Pure string-match dedup is insufficient.

**R1.5** HTSUS schedule text shall come from the authoritative current HTSUS
revision (target: 2026 revision). Source: USITC HTS database export.

**R1.6** Each chapter note and section note used in training shall be paired
with at least two worked product examples demonstrating how the note affects
classification (override of description, exclusion, scope limitation).

**R1.7** GRI examples shall cover at minimum:
  - GRI 1 (terms of heading and relative notes)
  - GRI 3(a) (most specific description)
  - GRI 3(b) (essential character)
  - GRI 3(c) (last in numerical order)
  - GRI 6 (subheading comparison rule)

**R1.8** Disambiguation negatives shall target the confusable clusters in R5.3.

**R1.9** Data preprocessing shall produce a `retrieval_context` field on at
least 80% of training examples (see R3).

### R2 — Output Format

**R2.1** The model shall emit the v1 structured text format:
```
Chapter NN: <description>
Heading NN.NN: <description>
Subheading NNNN.NN: <description>
HTS Code: NNNN.NN.NNNN
Reasoning: <text>
Provides for: <text>
```

**R2.2** The model shall emit a new `Confidence:` field immediately after the
`HTS Code:` line, with one of the following values:
  - `10-digit` — committed to the full statistical reporting number
  - `6-digit` — committed to subheading only
  - `4-digit` — committed to heading only
  - `2-digit` — committed to chapter only
  - `cannot-classify` — abstention

**R2.3** When `Confidence:` is less than `10-digit`, the `HTS Code:` field
shall contain only the committed portion. For example, a 6-digit commitment
shall emit `HTS Code: 6110.11` — not `6110.11.0000`.

**R2.4** When `Confidence:` is less than `10-digit`, a new field `Required for
full code:` shall appear after `Confidence:` specifying what information is
missing (e.g., "gender (men's/women's); article type (sweater/vest)").

**R2.5** Every citation in the `Reasoning:` field shall reference a specific
retrieved context entry by its identifier (CROSS ruling number, HTSUS section
reference, or chapter note reference). Free-floating citations like "CBP has
previously classified..." without a reference shall not appear.

**R2.6** The model shall not emit a citation for which there is no
corresponding entry in the retrieved context (enforced via training signal, see
R4.3).

**R2.7** The abstention format shall remain `Cannot classify: <reason>` for
backward compatibility, with an optional `Required for classification:` field
listing what's missing.

### R3 — Retrieval-Augmented Training

**R3.1** Each training example shall optionally carry a `retrieved_context`
field containing up to:
  - 5 CROSS rulings — `{ruling_number, excerpt, hts_code, schedule_version}`
  - 3 HTSUS schedule entries — `{code, description, level}`
  - 2 chapter/section notes — `{section, note_number, text}`

**R3.2** Retrieval shall use the existing `hts-api` MCP server's semantic
search tools. No new retrieval infrastructure required.

**R3.3** Retrieved context shall be rendered into the user prompt immediately
after the product information, as a structured section:
```
Product: <description>
Materials: ...
Use: ...

Relevant rulings:
- NY N123456 [8544.42.9000]: <excerpt>
- NY N234567 [8544.30.0000]: <excerpt>

Relevant HTSUS text:
- 85.44: Insulated wire, cables, and other insulated electric conductors...
- 8544.42: Other, fitted with connectors...

Relevant notes:
- Section XVI note 1(m): This section does not cover...
```

**R3.4** 80% of training examples shall include retrieval context. The
remaining 20% shall not, to preserve the model's ability to classify without
retrieval when the retrieval pipeline is unavailable at inference time.

**R3.5** The retrieval corpus (CROSS rulings, HTSUS text, notes) shall be a
frozen snapshot taken at training start. All training examples shall use this
snapshot to avoid retrieval drift during data prep.

**R3.6** At inference time, the serving layer shall invoke the same retrieval
pipeline against the *current* (non-frozen) corpus to augment classify
requests.

**R3.7** Retrieval during training data prep shall be cached per-example to
avoid rerunning queries across training epochs.

### R4 — Loss and Training Signal

**R4.1** Completion-only loss shall be used: loss is computed only over the
assistant response tokens, not the system/user prompt tokens. (Same as v1.)

**R4.2** The `HTS Code:`, `Heading:`, and `Subheading:` lines shall receive a
2x loss weight relative to reasoning/provides-for text. Exact numerical content
of these lines matters more than stylistic fluency of reasoning.

**R4.3** A format-compliance auxiliary loss shall penalize responses missing
any of: Chapter, Heading, Subheading, HTS Code, Confidence, Reasoning, Provides
for. Weight to be tuned; starting value 0.1 of the main loss.

**R4.4** A citation-grounding loss shall penalize reasoning text that
introduces citation references not present in the retrieved context of the
same example. Implementation: regex-extract citation tokens (`NY N\d+`,
`§ \d+\.\d+`) from the reasoning and verify they appear in the retrieved
context; penalize mismatches. Weight: 0.1 of the main loss.

**R4.5** LoRA configuration shall start from v1 hyperparameters (rank 32, alpha
64, all attention + FFN layers) and tune based on v2 eval results.

**R4.6** The training run shall use the same base model, quantization, and
tokenizer as v1.

### R5 — DPO Second-Stage Fine-Tune

**R5.1** After the main supervised fine-tune (SFT) completes, a Direct
Preference Optimization (DPO) second stage shall fine-tune the adapter on
preference pairs.

**R5.2** The DPO dataset shall contain between 2,000 and 5,000 preference pairs.

**R5.3** Preference pairs shall target the following failure clusters observed
in v1:
  - **Parts vs finished goods** (e.g., solar panel 8541.43 vs 8541.90; bike
    frame 8714.91 vs 8712.00)
  - **Material-based vs end-use classification** (e.g., aluminum bike frame
    → Ch 87 not Ch 76)
  - **HTS 2022 subheading splits** (especially 8541.40 family)
  - **Knitted vs woven apparel** (Ch 61 vs Ch 62)
  - **Residual "nesoi" overuse** (model picks "other" basket when a specific
    subheading applies)
  - **Fabricated vs grounded citations** (chosen = grounded in context,
    rejected = fabricated)

**R5.4** Rejected ("losing") responses in preference pairs shall be sourced
from v1 model predictions that were wrong on a held-out set of rulings.
Chosen ("winning") responses shall be the correct classification with grounded
reasoning.

**R5.5** DPO training shall not require more than 10% of the compute budget of
the main SFT run.

### R6 — Evaluation

**R6.1** A held-out test set of ≥2,000 examples shall be curated for final
evaluation. The test set shall not share any `ruling_number` with the training
set.

**R6.2** At least 10% of the test set shall be drawn from post-2022 rulings to
evaluate the HTS 2022 split-handling.

**R6.3** The evaluation shall compute the following per-example metrics:
  - Chapter exact match
  - Heading exact match
  - Subheading exact match
  - Full HTS code exact match (only when Confidence == 10-digit)
  - Hierarchy consistency (chapter ⊂ heading ⊂ subheading ⊂ code)
  - Parse rate (all required structured fields present)
  - Valid stat-suffix rate (when 10-digit committed)
  - Citation grounding rate (fraction of emitted citations present in context)
  - Abstention precision / recall
  - Confidence calibration (correlation between claimed confidence level and
    actual correctness at that level)

**R6.4** Per-chapter accuracy breakdowns shall be reported to identify
systematic weak spots.

**R6.5** A regression test set shall be maintained containing every v1 failure
observed during development, including at minimum:
  - Insulated copper wire (correct: 8544.42)
  - Aluminum bicycle frame (correct: 8714.91)
  - Silicon photovoltaic panel (correct: 8541.43)
  - Wool sweater (correct: 6110.11 with 10-digit clarification needed)

  v2 must not regress on any item in the regression set.

**R6.6** A slice metric shall be reported for the HTS 2022 split products,
specifically:
  - 8541.40 → 8541.41 / 8541.42 / 8541.43 / 8541.49
  - 8542.39 (various)
  - 8524 (flat panel display modules, newly added)

**R6.7** Evaluation shall run both with and without retrieval context to
measure the retrieval contribution. Reported as `score_with_retrieval` and
`score_without_retrieval`.

### R7 — HTS Code Validation (Post-Processing)

**R7.1** A validation post-processor shall check every emitted 10-digit HTS
code against the authoritative HTSUS stat suffix list.

**R7.2** Invalid 10-digit codes shall be flagged `valid_stat_suffix: false` in
the API response. The code is *not* auto-corrected — the model is held
accountable.

**R7.3** The validator shall load the full stat suffix list at server startup,
not per-request.

**R7.4** The stat suffix list source shall be the same HTSUS 2026 export used
for training (R1.5), ensuring consistency.

**R7.5** If the model commits to a 10-digit code that isn't in the list, the
response shall also populate `valid_codes_under_subheading: [list]` to help the
caller find the right stat suffix.

### R8 — Serving Integration

**R8.1** The v2 serving layer shall be backward-compatible with the v1 API.
New fields shall be additive only (`confidence`, `required_for_full_code`,
`valid_stat_suffix`, `retrieval_used`, `retrieved_context`).

**R8.2** The classify endpoint shall invoke the retrieval pipeline before the
LLM call and inject the results into the prompt.

**R8.3** Retrieval latency budget: 200ms p95 (excluding LLM time).

**R8.4** Total `/classify` latency budget: 5s p95 on the M4 Pro (retrieval +
inference + validation + parse).

**R8.5** If retrieval fails or times out, the endpoint shall fall back to
non-retrieval inference and set `retrieval_used: false` in the response. The
classify call shall never fail solely because retrieval failed.

**R8.6** The FastAPI response schema shall expose the retrieved context used
for the classification in a `retrieved_context` field, so consumers can audit
the reasoning grounding.

---

## Dependencies

| ID | Dependency | Status |
|---|---|---|
| D1 | HTSUS 2026 schedule data (authoritative export) | Not yet sourced |
| D2 | Chapter / section notes corpus | Partially available via hts-api |
| D3 | GRI 1–6 text | Need to transcribe or source |
| D4 | hts-api retrieval infrastructure | Exists |
| D5 | Post-2022 CROSS rulings (≥2k) | Need to verify volume |
| D6 | DPO training code | Need to add (not in v1 training stack) |
| D7 | HTSUS stat suffix list for validation | Derivable from D1 |
| D8 | GPU for SFT training | Same as v1 (rented H100) |
| D9 | GPU for DPO training | Same as v1 |
| D10 | Regression test set | Needs curation |

---

## Open Questions

These need decisions before detailed implementation planning:

**Q1** — Do we have access to a clean GRI corpus, or does it need to be
transcribed from HTSUS? (Affects R1.7, D3)

**Q2** — Are there enough post-2022 CROSS rulings to meaningfully shift the
training distribution, or do we need to down-weight pre-2022 rulings instead of
replacing them? (Affects R1.1, R1.3)

**Q3** — Should retrieved context be cached to a local database during data
prep to avoid hammering hts-api for 100k+ examples across multiple epochs?
(Affects R3.7)

**Q4** — Should DPO preference pairs be generated synthetically (hard to get
right) or exclusively from v1 wrong-prediction mining? Hybrid? (Affects R5.4)

**Q5** — Test set: gold-standard hand labels, or ruling-derived labels with
known noise? The former is more expensive but gives cleaner metrics. (Affects
R6.1)

**Q6** — Is the citation-grounding loss (R4.4) worth the implementation
complexity, or is the retrieval-augmented training (R3) alone sufficient to
eliminate fabricated citations?

**Q7** — Do we want to maintain v1 in production alongside v2 during a
transition, or is a hard cutover acceptable? (Affects R8.1 backward
compatibility strictness)

**Q8** — Should the confidence field (R2.2) also allow `uncertain-at-heading`
(2-digit chapter commit with heading doubt), or is the 5-level enum
sufficient?

---

## Success Criteria

v2 shall be considered complete and successful when **all** of the following
are true:

1. All R1–R8 requirements are met.
2. All quantitative targets in the Goals section are met on the held-out test
   set.
3. The regression test set (R6.5) passes with zero failures.
4. The HTS 2022 slice (R6.6) meets its accuracy target.
5. The citation grounding rate (R4.4 metric) is ≥ 95% on the test set.
6. The v2 model runs on the same local MLX + FastAPI serving stack as v1 with
   no new infrastructure dependencies (retrieval is the only addition, and it
   already exists).
7. A comparison report is published showing v1 vs v2 on matched prompts,
   including at minimum the four regression cases.

---

## Prioritization (suggested implementation order)

This is not an implementation plan, but a suggested ordering for when the plan
is written:

1. **Infrastructure: data + retrieval caching** — R1, R3, D1–D3, D7
2. **Format + training signal changes** — R2, R4 (SFT)
3. **Evaluation infrastructure** — R6, regression set (do this early so you
   can measure v1 baseline properly before training v2)
4. **SFT training run** — R4, R5.1 (SFT portion)
5. **DPO second stage** — R5
6. **HTS code validator** — R7
7. **Serving integration** — R8

Rationale for ordering:

- Eval (step 3) intentionally comes before training (step 4) so that v1 gets
  properly benchmarked on the same test set, giving v2 a real apples-to-apples
  comparison rather than the ~4-sample vibes we have today.
- The HTS code validator (step 6) is independent of everything else and could
  move earlier, but putting it after training means we can tell whether to
  validate all codes or only the ones the model is confident about.

---

## Explicit non-changes from v1

The following components of the v1 stack shall **not** change in v2:

- Base model: `nvidia/Llama-3.1-Nemotron-Nano-8B-v1`
- Chat format: Llama 3.1 with manual `<think>\n</think>\n\n` prefix
- System prompt: `detailed thinking off ...` (imported from formatters)
- Fine-tuning method: LoRA (not full fine-tune, not QLoRA)
- Quantization at serving: MLX q4 (4.5 bits per weight)
- Serving runtime: `mlx_lm.server` on the M4 Pro
- API wrapper: FastAPI + Pydantic, `POST /classify`, `GET /health`, `GET /`
- Response parser: `parse_output.py` with hierarchy backfill
- Output structure (core fields): Chapter / Heading / Subheading / HTS Code /
  Reasoning / Provides for

Changes to these components are explicitly out of scope. If any of them turn
out to be blockers for the v2 targets, that's a signal we need a v3 scope
discussion, not a scope creep on v2.
