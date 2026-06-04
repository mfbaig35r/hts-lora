# HTS LoRA Training Plan — Ministral 3 8B Reasoning

Requirements and implementation plan for training a second LoRA adapter on top
of `mistralai/Ministral-3-8B-Reasoning-2512`, as a comparison and potential
upgrade vs the existing `mfbaig35r/hts-nemotron-8b-lora-v1`.

## Overview

We've shipped v1 — a LoRA adapter on `nvidia/Llama-3.1-Nemotron-Nano-8B-v1`
trained on 119k HTS classification examples. It works (final loss 0.13, eval
loss 0.27 at best). This document plans a **second training run** on a
different base model to test a specific hypothesis:

> **Hypothesis**: A base model that's already been post-trained for chain-of-thought
> reasoning will outperform a vanilla instruct-tuned model on HTS classification,
> because HTS classification *is* a reasoning task — our output format has an
> explicit "Reasoning:" field where the model walks through chapter notes,
> material composition, and intended use to arrive at a code.

We're betting that `Ministral-3-8B-Reasoning-2512` will produce better
reasoning chains and therefore higher accuracy than Nemotron, despite mixing
two experimental variables (different model family + different post-training).

## Goals

1. Train a LoRA adapter on `mistralai/Ministral-3-8B-Reasoning-2512` using the same 119,602 training examples
2. Apply all the lessons learned from the Nemotron run (shorter training, best-checkpoint loading, more save slots)
3. Refactor the data pipeline to be model-agnostic so we never have to do this dance again for a third base model
4. Keep cost under ~$50 for the run (single H100 SXM, ~14 hours)
5. Produce a v1 adapter we can compare against Nemotron v1 head-to-head once the v2 evaluation pipeline is ready

## Non-goals

- Beating published benchmarks — we don't have v2 eval yet so there's nothing to compare formally
- Multi-modal training (vision encoder will be frozen, not trained)
- Training the 14B variant — out of scope, single GPU capacity concern
- Training all three Ministral variants (Base/Instruct/Reasoning) — Reasoning only for this run
- Hyperparameter sweeps — single run with the same hyperparameters as Nemotron, modulo lessons learned

## Decision rationale

### Why Reasoning-2512 (not Instruct-BF16 or Base)?

| Variant | Pro | Con |
|---|---|---|
| **Reasoning-2512** ✓ | Already trained for chain-of-thought; HTS classification IS a reasoning task; BF16 (LoRA-friendly) | Mixes two variables vs Nemotron (family + post-training); harder to attribute wins |
| Instruct-2512-BF16 | Cleanest apples-to-apples comparison vs Nemotron | Less interesting hypothesis; "another instruct base" |
| Base-2512 | Pure SFT target; no instruct bias | 119k examples may be too small to teach instruction-following + the task simultaneously |

We're picking **upside over experimental cleanliness**. If Reasoning crushes
Nemotron, we ship it. If it disappoints or we want a clean control, we can
run Instruct-BF16 as a follow-up.

### Why H100 SXM 80GB (not A100 80GB)?

- We know the H100 setup works end-to-end from the Nemotron run
- ~$5-8 savings on A100 isn't worth re-validating the GPU stack
- H100 finishes ~30% faster — less wall time babysitting

### Why apply lessons learned now (not save them for later)?

The Nemotron run showed:
- Eval loss bottomed at step 7000 (epoch 1.87)
- Mild overfitting through epochs 2-3
- We lost the best checkpoint because `save_total_limit` was too low

These are free wins. Apply them.

## What's different from the Nemotron run

| Dimension | Nemotron run | Ministral run | Notes |
|---|---|---|---|
| **Base model** | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | `mistralai/Ministral-3-8B-Reasoning-2512` | Different family entirely |
| **Architecture** | Llama 3.1 (text-only) | Mistral + 0.4B vision encoder | Vision encoder must be frozen |
| **Tokenizer** | Standard `AutoTokenizer` | Possibly `MistralCommonBackend` | **Risk** — see pre-flight |
| **Chat template** | Llama 3 with `"detailed thinking off"` + `<think>\n</think>\n` | Mistral `[INST]...[/INST]` style, no thinking block | Big format difference |
| **Vocabulary size** | 128,256 | TBD (likely ~131k) | Affects logits memory |
| **Context window** | 8k effective | 256k native | Doesn't matter for us (we use 1536) |
| **Epochs** | 3 (overfit) | **2** | Lesson from Nemotron |
| **save_total_limit** | 3 (lost best ckpt) | **5** | Lesson from Nemotron |
| **load_best_model_at_end** | False | **True** | Lesson from Nemotron |
| **eval_steps** | 1000 | **500** | Catch the bottom more precisely |
| **Data file format** | Nemotron-specific (system has `"detailed thinking off"`, assistant has `<think>` block) | **Generic messages, model-specific templating at training time** | Requires data pipeline refactor |

## Architecture concerns

### Vision encoder

Ministral 3 8B = 8.4B language model + 0.4B vision encoder = ~9B total. The
vision encoder is part of the model checkpoint and will load by default.

**Strategy**: Freeze it. Don't strip it. Specifically:
- After `from_pretrained`, iterate `model.named_parameters()` and set
  `requires_grad=False` for any parameter whose name contains `vision`,
  `vit`, `image`, or `mm_projector` (or whatever Mistral names them — verify
  in pre-flight)
- LoRA target_modules already exclude vision layers (we target
  `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj` which are LLM
  attention/MLP names)
- Verify in pre-flight that LoRA application doesn't accidentally wrap
  vision encoder layers

### Tokenizer compatibility

Mistral's official path uses `MistralCommonBackend` (requires
`mistral-common>=1.8.6`). Our training code uses standard `AutoTokenizer`.

**Strategy**:
1. **Try `AutoTokenizer.from_pretrained` first.** Mistral models historically
   have a `tokenizer.json` that the standard fast tokenizer can consume.
2. If `AutoTokenizer` works (returns sensible tokens, round-trips correctly),
   no code changes needed.
3. If it fails or produces wrong tokens, add `mistral-common>=1.8.6` as a
   training dep and use `MistralCommonBackend.from_pretrained()`.

This is a **gating risk** for the pre-flight smoke test — if the tokenizer
doesn't work, we can't proceed without code surgery.

### `target_modules` for Mistral

Mistral architecture uses the same module names as Llama:
- `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention)
- `gate_proj`, `up_proj`, `down_proj` (MLP)

Our existing target_modules list should work as-is. **Verify in pre-flight**
by inspecting `model.named_modules()` after load and confirming PEFT picks
up the right targets.

## Data pipeline refactor

**Current state**: `data/formatted/{train,valid,test}.jsonl` files contain
messages with Nemotron-specific prefixes baked in:

```json
{
  "messages": [
    {"role": "system", "content": "detailed thinking off\n\nYou are an expert..."},
    {"role": "user", "content": "Product: ..."},
    {"role": "assistant", "content": "<think>\n</think>\n\nChapter 19: ..."}
  ]
}
```

This is fine for Nemotron but unusable for Ministral, which doesn't have a
"thinking off" mode and uses a totally different chat template.

**Target state**: Store generic messages with no model-specific quirks, then
apply model-specific templating at training time:

```json
{
  "messages": [
    {"role": "system", "content": "You are an expert..."},
    {"role": "user", "content": "Product: ..."},
    {"role": "assistant", "content": "Chapter 19: ..."}
  ]
}
```

The model-specific stuff (`"detailed thinking off"` prefix, `<think>\n</think>\n`
block, etc.) becomes a function applied at training time based on a config
field.

### Refactor steps

1. **Add `model_format` to training config schema**:
   ```yaml
   training:
     model_format: "nemotron"  # or "mistral" or "llama3"
   ```

2. **Add `apply_model_format(messages, model_format)` function** in
   `src/hts_lora/data/formatters.py`:
   - `nemotron`: prepends `"detailed thinking off\n\n"` to system content,
     prepends `"<think>\n</think>\n\n"` to assistant content
   - `mistral`: pass-through (no quirks)
   - `llama3`: pass-through
   - Default: pass-through with a warning

3. **Update `scripts/run_data_prep.py`** to write generic messages (no
   Nemotron quirks) to `data/formatted/{train,valid,test}.jsonl`. Re-run data
   prep once.

4. **Update `src/hts_lora/training/train_lora.py`** to call
   `apply_model_format(messages, config.training.model_format)` on each
   example before tokenization.

5. **Backwards compatibility**: For the existing Nemotron training to keep
   working, set `model_format: "nemotron"` in
   `configs/train_h100.yaml`. The result is byte-identical to what we
   trained on before.

6. **Tests**: Add `tests/test_apply_model_format.py` covering all three
   formatters and the round-trip (generic → format-specific → tokenized).

7. **Verify**: After regenerating `data/formatted/`, do a diff against
   `data/formatted_nemotron_v1/` (a backup of the old files) to confirm
   the only differences are the system/assistant content prefixes.

### Data backup

Before regenerating `data/formatted/`, **back up the existing files**:

```bash
cp -r data/formatted data/formatted_nemotron_v1
```

So we can compare and revert if anything goes sideways.

## Training config

New file: `configs/train_h100_ministral.yaml`

```yaml
# Training config for Ministral 3 8B Reasoning on 1x H100 SXM 80GB (RunPod)
# Effective batch size: 16 * 1 GPU * 2 grad_accum = 32

model:
  model_id: "mistralai/Ministral-3-8B-Reasoning-2512"
  torch_dtype: "bfloat16"
  attn_implementation: "sdpa"
  freeze_vision_encoder: true   # NEW — Ministral has a 0.4B vision encoder
  vision_param_patterns:        # NEW — names to match for freezing
    - "vision"
    - "vit"
    - "mm_projector"
    - "image_encoder"

quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_use_double_quant: true

lora:
  r: 32
  lora_alpha: 64
  lora_dropout: 0.05
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
  bias: "none"
  task_type: "CAUSAL_LM"

training:
  model_format: "mistral"          # NEW — see refactor above
  num_train_epochs: 2              # CHANGED from 3 (Nemotron overfit in epoch 3)
  per_device_train_batch_size: 16
  per_device_eval_batch_size: 16
  gradient_accumulation_steps: 2
  learning_rate: 2.0e-4
  weight_decay: 0.01
  warmup_ratio: 0.05
  lr_scheduler_type: "cosine"
  max_seq_length: 1536
  logging_steps: 10
  eval_steps: 500                  # CHANGED from 1000 (catch the bottom)
  save_steps: 500                  # CHANGED from 1000
  save_total_limit: 5              # CHANGED from default (3) — keep best ckpt
  load_best_model_at_end: true     # NEW
  metric_for_best_model: "eval_loss" # NEW
  greater_is_better: false         # NEW
  seed: 42
  bf16: true
  fp16: false
  gradient_checkpointing: true
  optim: "paged_adamw_8bit"

data_dir: "data/formatted"
output_dir: "outputs"
```

### Expected runtime

| | Nemotron | Ministral (estimated) |
|---|---|---|
| Steps per epoch | 3,738 | ~3,738 (same data, same batch) |
| Total steps | 11,214 (3 epochs) | **7,476 (2 epochs)** |
| Step time | ~3.85s/step | ~3.5–4.5s/step (depends on Mistral throughput) |
| Eval time | ~510s × 11 = 5,610s | ~510s × 16 = 8,160s (more frequent eval) |
| **Total wall time** | ~13h 44m | **~9–11 hours** |
| **Cost @ $2.69/hr** | $43 | **~$25–30** |

We win on cost (~$15 savings) by training fewer epochs, even after spending
more on more frequent evals.

## Pre-flight smoke test (BEFORE renting the H100)

This is the critical de-risking step. We need to verify the model + tokenizer
+ LoRA pipeline works *before* paying for a long-running rental.

**Cheap path**: Rent a small GPU (RTX 4090 or A40 — $0.30-0.50/hr) for 30
minutes. Total cost ~$0.25.

**Cheaper path**: Use the M4 Pro with BF16 inference (no bitsandbytes — Mac
has no CUDA). Validates tokenizer + chat template + forward pass, but NOT the
4-bit quantization path or gradient flow with grad checkpointing.

**Recommendation**: Cheap GPU rental. Worth the $0.25 to validate the full
4-bit + LoRA + Trainer pipeline before committing to a $25-30 run.

### Smoke test script: `scripts/smoke_test_ministral.py`

```
1. Load the tokenizer:
   - Try AutoTokenizer.from_pretrained("mistralai/Ministral-3-8B-Reasoning-2512")
   - If that fails, try MistralCommonBackend.from_pretrained(...)
   - Verify a round-trip: tokenize a sample message, decode, compare strings

2. Load the model with 4-bit quantization (bitsandbytes NF4)
   - Verify total params ≈ 8.4B + 0.4B (vision) = ~8.8B
   - Verify trainable params before LoRA = some subset (vision frozen, LM in 4-bit not trainable)

3. Apply LoRA adapter:
   - peft.get_peft_model with our target_modules
   - Verify trainable params ≈ 80-100M (similar ratio to Nemotron)
   - Verify NO vision encoder layers got LoRA wrapped

4. Build a sample message in Mistral chat format:
   - Use the existing data formatter (apply_model_format with model_format='mistral')
   - Apply the chat template
   - Tokenize, verify length is reasonable (< 512 tokens for a small example)

5. Run a single forward pass:
   - Compute loss
   - Verify loss is in the expected range (~2-4 for an untrained model on this format)
   - Verify gradients flow (loss.backward(), check param.grad is not None on at least one LoRA param)

6. Print success summary or fail loudly with a clear error
```

If this script passes on a small rented GPU in 30 minutes, we know the run
will work on the H100. If it fails, we have time to fix it without paying for
hours of an idle H100.

### Smoke test acceptance criteria

- [ ] Tokenizer loads (either path)
- [ ] Sample message round-trips through tokenize → decode
- [ ] Model loads with 4-bit quantization
- [ ] Total parameter count is in the expected range
- [ ] LoRA applies cleanly with our target_modules
- [ ] LoRA trainable params ≈ 80-100M (matches Nemotron ratio)
- [ ] Vision encoder layers are NOT in the LoRA wrapped set
- [ ] Single forward pass returns a sensible loss
- [ ] `loss.backward()` populates gradients on LoRA params

## H100 setup (this weekend)

### Pre-rental checklist

- [ ] Pre-flight smoke test passed on a cheap GPU
- [ ] Data pipeline refactor merged + tests pass
- [ ] `configs/train_h100_ministral.yaml` written and validated
- [ ] `data/formatted/` regenerated with generic messages
- [ ] `data/formatted_nemotron_v1/` backup exists
- [ ] H100 SXM 80GB pod created on RunPod, SSH access verified
- [ ] Disk space sanity-checked (move HF cache to /workspace from day 1 — see Nemotron lessons)

### Pod setup steps (mostly mirror Nemotron)

1. SSH into pod (direct TCP, not the gateway)
2. Move HF cache to `/workspace/.cache/huggingface` with symlink (avoid root disk filling)
3. `git clone` hts-lora repo into `/workspace/`
4. `cd /workspace/hts-lora && uv sync`
5. Pin compatible versions:
   ```
   uv pip install transformers==4.46.3 peft==0.13.2 accelerate==0.34.2
   ```
   (Same versions that worked for Nemotron — don't experiment.)
6. **Add Mistral-specific deps if smoke test required them**:
   ```
   uv pip install "mistral-common>=1.8.6"
   ```
7. Upload the generic-format `data/formatted/` files to `/workspace/hts-lora/data/formatted/` (use the same `upload_server.py` helper from the Nemotron run)
8. Run smoke test on the H100 itself (`scripts/smoke_test_ministral.py`) to confirm
9. Launch training with `nohup` + `PYTHONUNBUFFERED=1` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + `>train.log 2>&1`
10. Start the monitor.sh script (~30 min interval VRAM + step + loss check)

### Things that bit us last time (don't forget)

| Issue | Fix |
|---|---|
| Root disk filled with HF cache | Move cache to /workspace day 1 |
| Flash Attention 2 ABI mismatch | Use SDPA — don't try to install flash-attn |
| transformers 5.x set_submodule incompat | Pin transformers 4.46.3 |
| MCP read_timeout too tight | Not relevant here (no MCP), but training run will be ~10h |
| Lost best checkpoint (save_total_limit=3) | save_total_limit=5 + load_best_model_at_end |

## Training execution

Same approach as Nemotron:
1. Launch with `nohup` so SSH disconnect doesn't kill it
2. Background `monitor.sh` writing every 30 min to `monitor.log`
3. Periodic status checks throughout the day
4. Spot-check eval losses to catch unexpected behavior early

### Things to watch for

- **Eval loss should bottom around step 3,000-4,000** (~half of 7,476 total) if our hypothesis about overfitting timing holds. With `load_best_model_at_end`, the best checkpoint is automatically restored at the end, so this is just informational.
- **Loss should converge faster than Nemotron** if Reasoning's pre-existing reasoning capability transfers well. Initial loss ~2.5, expect to hit ~0.4 by step 1000 (Nemotron took ~1500).
- **VRAM should be similar** (~50-65GB peak during eval) — both are 8B + 4-bit + LoRA on the same data.

## Post-training (mirror the Nemotron flow exactly)

1. **Verify final adapter works** with an inference test on a known product
   (insulated copper wire, expect chapter 85). Use the Mistral chat template
   this time — no `<think>` block prefix.

2. **Backup adapter to local Mac**:
   ```
   scp -P <port> -r root@<host>:/workspace/hts-lora/outputs/<run-dir>/adapter \
       /Users/fbaig/Projects/hts-lora/outputs/train_h100_ministral_<date>/adapter
   ```

3. **Backup training artifacts** (train.log, metrics.jsonl, sample_predictions.jsonl,
   config_snapshot.yaml) into the same local directory, mirroring the
   Nemotron layout.

4. **Push to HuggingFace** (private repo):
   ```
   .venv/bin/hf upload mfbaig35r/hts-ministral-3-8b-reasoning-lora-v1 \
     outputs/train_h100_ministral_<date>/adapter \
     . --repo-type=model --private \
     --commit-message "Initial v1 upload: H100 ~10h, Ministral 3 8B Reasoning + HTS LoRA"
   ```

5. **Write a real README** for the HF repo (don't ship the empty PEFT stub).
   Include the Mistral chat template instead of Nemotron's `<think>` block.

6. **Verify SHA256** of local adapter matches HF LFS — same belt-and-suspenders
   check we did for Nemotron.

7. **Kill the pod** once everything is verified saved.

## Evaluation (deferred)

We can't run real evaluation until the v2 inference/eval pipeline is built
(per `/Users/fbaig/.claude/plans/memoized-hatching-teapot.md`). For this
training run, success criteria are:

1. Training completes without crashes
2. Final eval loss is in a sensible range (< 0.30)
3. Spot-check inference on 5-10 known products produces correctly-formatted
   structured output (Chapter/Heading/Subheading/HTS Code/Reasoning/Provides for)
4. Outputs are at least as coherent as Nemotron on the same products

Real comparison (parse rate, exact match, hierarchy consistency, etc.) waits
for the v2 eval rewrite.

## Open questions

| ID | Question | Decision |
|---|---|---|
| Q1 | Variant: Reasoning, Instruct-BF16, Base, or all three? | **Reasoning-2512** (decided) |
| Q2 | GPU: H100 SXM, A100 80GB, or cheaper? | **H100 SXM 80GB** (decided) |
| Q3 | When? | **Plan now, train this weekend** (decided) |
| Q4 | Tokenizer: AutoTokenizer or MistralCommonBackend? | **AutoTokenizer first; fall back to MistralCommonBackend if needed** — decided in pre-flight |
| Q5 | Vision encoder: freeze or strip? | **Freeze** — cleaner, less invasive |
| Q6 | Data pipeline: refactor now or hack at training time? | **Refactor** — cleaner architecture, enables future model swaps |
| Q7 | Run pre-flight on M4 Pro or rent a cheap GPU? | **Rent a cheap GPU** — validates the full 4-bit + grad path |

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `AutoTokenizer` doesn't work for Ministral; need `MistralCommonBackend` | Medium | Low (fallback exists) | Try AutoTokenizer first in smoke test, fall back if needed |
| `MistralCommonBackend` doesn't integrate cleanly with HF Trainer | Low | High (would block training) | Validate in smoke test before renting H100 |
| Vision encoder accidentally gets LoRA-wrapped | Low | Medium (wastes params, may break) | Verify trainable param breakdown in smoke test |
| Vision encoder breaks 4-bit quantization | Low | High | Validate in smoke test; if so, manually `del model.vision_encoder` after load |
| Mistral chat template differs from training data format → garbage outputs | Medium | High | Validate end-to-end in smoke test (tokenize, run forward, decode) |
| Reasoning model produces verbose chain-of-thought outside our format | Medium | Medium | Train with our exact target format; greedy decoding at inference; adjust eval to handle leading reasoning if needed |
| New `target_modules` names for Mistral | Low | Low | Verify with `model.named_modules()` in smoke test |
| Flash Attention not needed but PEFT version incompatible | Low | Medium | Pin same versions as Nemotron run (transformers==4.46.3, peft==0.13.2, accelerate==0.34.2) |
| Cost overrun (run takes longer than expected) | Low | Low | Hard cap: kill pod if loss isn't converging by step 2000 |
| Reasoning model is just *worse* than Nemotron on this task | Medium | Low (we have v1 already) | Ship v1 anyway as a comparison point; document the negative result |

## Implementation order

Strict ordering — each step depends on the previous.

### Phase A: Data pipeline refactor (local, no GPU)

1. **Backup current data**: `cp -r data/formatted data/formatted_nemotron_v1`
2. **Add `apply_model_format()` function** to `src/hts_lora/data/formatters.py`
3. **Update `scripts/run_data_prep.py`** to write generic messages (no Nemotron quirks)
4. **Update `src/hts_lora/training/train_lora.py`** to call `apply_model_format` based on `config.training.model_format`
5. **Add `model_format` field** to the training config schema (with default `"none"` for safety)
6. **Update `configs/train_h100.yaml`** (Nemotron config) to set `model_format: "nemotron"`
7. **Write tests** (`tests/test_apply_model_format.py`)
8. **Re-run data prep**: `uv run python scripts/run_data_prep.py` → produces new `data/formatted/`
9. **Diff verify**: confirm new `data/formatted/` matches old after applying `nemotron` format
10. **Run all tests**: `uv run python -m pytest tests/ -v --tb=short`

### Phase B: Training scaffolding (local, no GPU)

11. **Create `configs/train_h100_ministral.yaml`** (the new config above)
12. **Create `scripts/smoke_test_ministral.py`** (the pre-flight script above)
13. **Update model loading code** (`src/hts_lora/training/model_factory.py`) to:
    - Honor `freeze_vision_encoder` config
    - Match `vision_param_patterns` against parameter names
    - Set `requires_grad=False` for matching params
14. **Test locally what's testable** (config parses, smoke test imports cleanly)

### Phase C: Pre-flight (cheap rented GPU, ~$0.25)

15. **Rent a 4090 or A40** for 30 minutes
16. **Run `scripts/smoke_test_ministral.py`** on the rented GPU
17. **Verify all 9 acceptance criteria pass** (see Smoke test acceptance criteria above)
18. **Kill the cheap GPU**
19. **Iterate on smoke test issues** locally if anything failed

### Phase D: Real training (H100 SXM, ~$30, ~10h)

20. **Rent an H100 SXM 80GB** on RunPod
21. **Set up the pod** (move HF cache, clone repo, install deps, upload data)
22. **Run smoke test on the H100** to triple-confirm setup
23. **Launch training** with `nohup` + `PYTHONUNBUFFERED=1` + monitoring
24. **Periodic status checks** (every 1-2h)
25. **Wait for training to complete** (~10h)
26. **Verify inference works** with a test product
27. **Backup adapter + artifacts** to local Mac
28. **Push to HuggingFace** (private)
29. **Write real README** for the HF repo
30. **SHA256 verify** local vs HF
31. **Kill the H100 pod**

### Phase E: Wrap-up

32. **Update `docs/training-plan.md`** with a "v1.5 (Ministral)" section pointing at the new HF repo
33. **Mark this plan doc as complete**
34. **If results look promising**, queue up the v2 evaluation pipeline work to do a real comparison

## Acceptance criteria (end of Phase D)

- [ ] Data pipeline refactor merged, all tests pass
- [ ] Smoke test passes on a cheap GPU
- [ ] H100 training completes 2 epochs (~7,476 steps) without crashes
- [ ] Final eval loss < 0.30
- [ ] `load_best_model_at_end` restored a checkpoint with eval_loss < final eval_loss
- [ ] Spot-check inference: 5/5 test products produce correctly-formatted structured output
- [ ] Adapter pushed to `mfbaig35r/hts-ministral-3-8b-reasoning-lora-v1` (private)
- [ ] Local backup at `outputs/train_h100_ministral_<date>/`
- [ ] SHA256 of local adapter matches HF LFS
- [ ] Total cost < $50

## Future work

- **v1.1**: Train Instruct-BF16 as the experimental control, if Reasoning-2512 results are promising
- **v2 evaluation**: Build the inference/eval rewrite (per the existing plan file) and run formal head-to-head: Nemotron vs Ministral-Reasoning vs Ministral-Instruct
- **Vision augmentation**: Explore fine-tuning the vision encoder on product images for HTS-from-photo classification (would be a huge differentiator vs published benchmarks)
- **14B comparison**: If we want to push accuracy harder, train `Ministral-3-14B-Instruct-2512` on the same data — still single-GPU feasible with QLoRA on H100
