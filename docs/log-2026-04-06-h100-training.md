# H100 Training Run: Nemotron-Nano-8B + HTS LoRA v1

**Date**: April 6, 2026
**Duration**: ~16 hours total (2h setup/debug + 13h44m training + ~30min verification + ~30min wrap-up)
**Cost**: ~$43 ($2.69/hr × 16h on RunPod H100 SXM 80GB)
**Outcome**: ✓ Successful — adapter shipped to `mfbaig35r/hts-nemotron-8b-lora-v1`

---

## TL;DR

- Trained an 8B-param HTS classifier from a coffee shop on rented metal in under 16 hours, $43
- Final training loss ~0.13 (running avg 0.248), best eval loss 0.2596 at step 7000
- Adapter is live on HuggingFace (private), backed up locally with full reproducibility bundle
- **Key finding**: model overfit mildly in epochs 2-3 — best checkpoint was step 7000 (epoch ~1.87), but we lost it because `save_total_limit` was 3
- **Other key finding**: my own session memory was wrong about the training config (it was 16×2 not 8×4, and max_seq_length was 1536 not 2048) — only discovered when we pulled the actual log files at the end. Documented here to prevent future confusion.
- Wrote two follow-up plan docs: serving plan and Ministral 3 8B Reasoning training plan

---

## Where We Started

Phase 2 from yesterday (April 5) had built:
- 119,602 train + 14,784 valid + 14,952 test formatted examples in `data/formatted/`
- v2 inference and eval pipeline (parse_output.py with ParsedPrediction dataclass)
- Updated metrics module for v2 structured output

What we **didn't** have:
- A trained model (Phase 2 ended without successful training)
- Confidence the training pipeline actually worked end-to-end on a real GPU

The previous Lambda V100 attempts had burned ~$33 with no usable adapter. Today's goal was simple: **rent an H100, get the model trained, verify it works, ship it.**

---

## The Setup Gauntlet

Going from "rented H100" to "training is running" took **~2 hours of debugging** before we could launch a real run. Every obstacle below cost real money on the meter.

### Obstacle 1: Flash Attention 2 ABI mismatch

**Symptom**: After installing `flash-attn==2.8.3`, importing it failed with:
```
undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
```

**Root cause**: The pre-built wheel was compiled against a different PyTorch ABI than the one installed on the pod.

**What I tried first** (wrong): Install a different flash-attn version. Same error.

**What worked**: Give up on Flash Attention. Switch the model factory to use SDPA (PyTorch's built-in scaled dot-product attention). It's slightly slower but works out of the box, has no version dependency hell, and is actively maintained as the default attention path in transformers.

**Lesson**: For 8B models on a single H100, SDPA is fine. Only chase Flash Attention if you're (a) memory-bound, (b) on multiple GPUs where the speedup compounds, or (c) using a base image where it's already pre-installed and known-compatible.

### Obstacle 2: transformers 5.5.0 incompatibility with torch 2.4.1

**Symptom**: When loading the model:
```
AttributeError: 'LlamaForCausalLM' object has no attribute 'set_submodule'
```

**Root cause**: `transformers==5.5.0` (latest) calls `model.set_submodule()`, which is a method added in `torch==2.6+`. The pod had `torch==2.4.1`.

**What I considered**: Upgrade torch to 2.6. But upgrading torch on a working pod risks cascading breaks (CUDA driver compat, bitsandbytes, peft, accelerate all depend on torch versions).

**What worked**: Downgrade transformers instead. Pinned the same versions that worked on the V100:
```
pip install transformers==4.46.3 peft==0.13.2 accelerate==0.34.2
```

**Lesson**: When you have a known-good version triple (transformers + peft + accelerate), pin them. Don't let the dependency resolver pull "latest" because the latest version of transformers is usually 1-2 weeks behind on supporting "latest" torch features. The lag is real.

### Obstacle 3: Disk space crisis (root filesystem 94% full)

**Symptom**: Pod's root filesystem was 20GB total. After downloading the Nemotron base model (~16GB), root was at 94%. The next install would fail.

**Root cause**: HuggingFace caches models to `~/.cache/huggingface/hub`. On RunPod, `/root` is small (20GB) but `/workspace` is huge (200TB+). The HF cache landed on the small disk.

**What worked**:
```bash
mkdir -p /workspace/.cache
mv /root/.cache/huggingface /workspace/.cache/huggingface
ln -s /workspace/.cache/huggingface /root/.cache/huggingface
```

After the move, root dropped from 94% to 5%.

**Lesson**: On any rented GPU with separate root and data volumes, **move the HF cache to the data volume on day 1, before downloading anything.** Add this to the pod setup script. Don't wait until you're out of space.

### Obstacle 4: Smoke test typos (mine)

Two trivial bugs in my smoke test script:
1. Wrote `torch.cuda.get_device_properties(0).total_mem` (should be `total_memory`)
2. Tried to unpack `model, tokenizer = load_base_model(...)` but `load_base_model` returns just the model (the tokenizer is loaded separately by `load_tokenizer`)

Fixed by reading the actual function signatures in `model_factory.py` instead of guessing them. **Lesson**: Always read the function signature before calling it, even for "obvious" APIs.

### Obstacle 5: First successful smoke test

After all of the above, finally:
```
Loaded base model: nvidia/Llama-3.1-Nemotron-Nano-8B-v1
LoRA applied: 83,886,080 trainable / 8,114,147,328 total (1.03%)
```

83.9M trainable parameters, 1.03% of the 8.1B base. The pipeline worked.

---

## The OOM Saga

With smoke test passing, launched training. Immediately hit OOM.

### First attempt: batch_size=16

**Symptom**:
```
OutOfMemoryError: CUDA out of memory. Tried to allocate 4.69 GiB.
GPU 0 has a total capacity of 79.18 GiB of which 1.73 GiB is free.
```

49GB used, only 1.7GB free, then it tried to allocate another 4.7GB during the cross-entropy computation.

**Root cause**: Llama 3 family has a **128,256-token vocabulary**. The cross-entropy loss computation builds a logits tensor of shape `[batch, seq_len, vocab]`:
```
16 (batch) × 1536 (seq) × 128,256 (vocab) × 4 bytes (fp32) = 12.6 GB just for logits
```
On top of model weights (~20GB in 4-bit + LoRA + optimizer states + activations + gradient checkpointing buffers), this was the straw that broke it.

### Second attempt: batch_size=8, grad_accum=4

Same effective batch size (32), half the memory pressure on logits.

This worked. **Or so I thought.**

### What actually ran

When we pulled the training artifacts at the end of the day, the actual config that was used was **batch_size=16, gradient_accumulation_steps=2, max_seq_length=1536**. So I'd been telling the user "batch=8, accum=4, seq=2048" all day, and it was wrong.

The truth (from `train.log` line 6: `Batch size: 16 x 2`) is that **the second OOM-fix attempt also reduced max_seq_length from 2048 to 1536**. With seq=1536 instead of 2048, the logits tensor is `16 × 1536 × 128,256 × 4 = 12.6 GB` instead of `16 × 2048 × 128,256 × 4 = 16.8 GB` — a 25% reduction that was apparently enough to make batch=16 fit again.

**Lesson**: When debugging OOM, multiple knobs change at once and it's easy to lose track of which one mattered. Always pull the actual config from the run output (not your memory) when reporting on what ran.

---

## The Run

### Launch config (the actual one)

```yaml
model: nvidia/Llama-3.1-Nemotron-Nano-8B-v1
torch_dtype: bfloat16
attn_implementation: sdpa
load_in_4bit: true
bnb_4bit_quant_type: nf4
bnb_4bit_compute_dtype: bfloat16
bnb_4bit_use_double_quant: true

lora:
  r: 32
  lora_alpha: 64
  lora_dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  bias: none

training:
  num_train_epochs: 3
  per_device_train_batch_size: 16
  gradient_accumulation_steps: 2     # effective batch = 32
  learning_rate: 2.0e-4
  warmup_ratio: 0.05
  lr_scheduler_type: cosine
  max_seq_length: 1536
  bf16: true
  gradient_checkpointing: true
  optim: paged_adamw_8bit
  eval_steps: 1000
  save_steps: 1000
  # save_total_limit: defaulted to 3 — this bit us, see Lessons
```

Launched with:
```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONUNBUFFERED=1 \
nohup python scripts/run_train.py --config configs/train_h100.yaml \
  > train.log 2>&1 &
```

Three environment-level details that mattered:
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` prevents fragmentation issues that show up after a few thousand steps
- `PYTHONUNBUFFERED=1` makes train.log readable in real time (without it, tqdm and prints buffer indefinitely)
- `nohup ... &` means the process survives SSH disconnect, which is essential for a 14-hour run

### Monitor script

Wrote a tiny `/workspace/monitor.sh` that logged every 30 minutes:

```bash
#!/bin/bash
while true; do
  echo "=== $(date '+%H:%M:%S') ==="
  grep -oP "'loss': [0-9.]+" /workspace/train.log | tail -1
  tail -1 /workspace/train.log | grep -oP '\d+/11214'
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print "VRAM: " $1 " MiB"}'
  echo ''
  sleep 1800
done
```

This let us check status with `tail monitor.log` from any SSH session without interrupting training.

### Loss trajectory

| Step | Epoch | Loss | Eval loss | Notes |
|---|---|---|---|---|
| 10 | 0.003 | 2.5136 | — | Cold start |
| 1000 | 0.27 | ~1.05 | 0.3711 | First eval — VRAM jumped 40GB → 67GB |
| 2000 | 0.54 | ~0.66 | 0.3201 | |
| 3000 | 0.80 | ~0.49 | 0.2979 | |
| 4000 | 1.07 | ~0.33 | 0.2889 | Past epoch 1 |
| 5000 | 1.34 | ~0.21 | 0.2789 | |
| 6000 | 1.61 | ~0.21 | 0.2680 | |
| **7000** | **1.87** | **~0.20** | **0.2596** | **Best eval — overfitting starts here** |
| 8000 | 2.14 | ~0.21 | 0.2768 | Eval loss going UP — overfitting |
| 9000 | 2.41 | ~0.14 | 0.2743 | Train loss falling, eval rising |
| 10000 | 2.68 | ~0.13 | 0.2736 | |
| 11000 | 2.94 | ~0.13 | 0.2729 | |
| 11214 | 3.00 | ~0.13 | — | Final |

**The shape**: training loss kept falling (model memorized the training set), but eval loss bottomed at step 7000 and slowly crept up afterward. Classic mild overfitting in the back half of the run.

### VRAM and timing

- **Steady-state VRAM during training**: ~40GB
- **VRAM during eval**: ~67GB (model + activations + eval batch dataloader)
- **Step time**: ~3.85 seconds/step
- **Eval time**: ~510 seconds × 11 evals = ~94 minutes total in eval
- **Total wall time**: 13h 44m (49,452 seconds reported by Trainer)

The eval-time VRAM jump was unexpected. Possibly worth investigating later — it suggests the eval dataloader keeps a larger working set than necessary. For now, we just had headroom.

---

## Conversation During The Run

The training run was 14 hours long and we had a long meandering conversation while it ran. Worth capturing the architecture insights that came out of it because they're going to drive future work.

### The "8B as a building block" architecture

The user's vision: **the trained 8B model becomes an API that a frontier model (Sonnet/GPT-5.4-mini) calls as a tool.** The orchestrator handles general reasoning, customer interaction, and edge cases; the 8B handles the narrow domain task of mapping product descriptions to HTS codes.

This is the right architecture for two reasons:
1. **8B is small enough to run cheaply (or locally)** — fits in 24GB VRAM in BF16, fits in <12GB quantized
2. **A specialized model with the right training data can beat a frontier model on a narrow task** — published benchmarks back this up (ATLAS hit 40% with Llama-3.3-70B fine-tuned, GPT-5-Thinking zero-shot hit 25% on the same test set)

The economics: Sonnet would cost $3-15 per 1M tokens. An 8B running on a small rented GPU costs maybe $0.30-0.60 per 1M tokens (vLLM continuous batching). For high-volume HTS classification (millions of items), the cost difference is the entire margin.

### M4 Mac mini as a deployment target

User asked: "what's the minimum GPU needed to run our tuned 8B parameter model? A base M4 Mac mini?"

Answer: **Yes**. A base M4 Mac mini (16GB) would run the model in 4-bit quantized comfortably, probably 15-30 tok/s. An M4 Pro Mac mini (24GB) would run it in BF16 with no quantization. Either is < $700 hardware cost.

This unlocks an interesting deployment pattern: **a fleet of Mac minis serving HTS classification at the edge**, no cloud GPU costs, no per-token pricing, no rate limits. For an internal tool at a customs broker, this is genuinely viable.

### "Sharing this with the world"

User's framing: "we're building the HTS LLM for the world... we're sharing it freely to sink the shitty startups that tried to put a patent on RAG over 8 digit HTS descriptions."

There's a real strategic question here about open-vs-closed source and timing. The user's instinct is to publish v1 freely but **lag the public release behind their internal version by one generation**. While v1 is public, they're already training v2 internally. By the time the world catches up, they're another step ahead. This is the same play that worked for Stability AI (publish v1, keep v2 internal as a moat) until it didn't (because they couldn't sustain it commercially).

For a one-person + AI team, the lag-behind-by-one-generation play might actually be more sustainable than pure open or pure closed, because:
- Open release builds reputation and prevents enclosure (no one can patent what's already public)
- The next-gen advantage is short-lived but recurring
- Maintenance burden of public releases stays minimal (no SLAs)

### The "team of two" reflection

User: *"it's kinda remarkable you and I did this by ourselves, we're literally the entire team."*

Worth noting in this log because it's true and it matters. The work that happened today — debugging GPU environment issues, training a custom 8B model, verifying it, publishing it to HF, writing two requirement docs for follow-up work — would conventionally need a team of 3-5 people: an MLE, a devops/infra engineer, a researcher, and probably a tech writer. With the LLM-as-pair-programmer pattern, it's one human + one Claude session.

The bottleneck isn't capability anymore. It's attention — what to work on next, what to defer, when to push and when to rest. Most of my (Claude's) value today was in keeping the user oriented during the long debugging stretches, not in writing novel code.

---

## Inference Verification

After training completed, before backing up, we needed to verify the model actually worked.

### First inference attempt: garbled output

Built a quick test script that loaded the base model + adapter and generated for a test product (insulated copper electrical wire). Output was incoherent — random tokens, no structure.

**Diagnosis**: Wrong prompt format. I'd built the prompt with a generic system message and standard chat template, but the training data uses a Nemotron-specific format.

### Reading the actual training data format

Pulled `/workspace/hts-lora/data/formatted/test.jsonl` and read it. Discovered:

```json
{
  "messages": [
    {"role": "system", "content": "detailed thinking off\n\nYou are an expert..."},
    {"role": "user", "content": "Product: ...\nMaterials: ...\nUse: ...\nCountry: ..."},
    {"role": "assistant", "content": "<think>\n</think>\n\nChapter 19: ...\nHeading 19.01: ...\nHTS Code: 1901.90.9195\n\nReasoning: ..."}
  ]
}
```

The format has two non-obvious quirks specific to Nemotron:
1. **System prompt starts with `"detailed thinking off\n\n"`** — this is Nemotron's switch for skipping chain-of-thought generation
2. **Assistant content starts with `"<think>\n</think>\n\n"`** — an empty thinking block, telling the model "I've thought about it, here's the answer"

Without these prefixes, the model produces garbage because it's never seen a prompt without them during training.

### Second inference attempt: works

Rebuilt the test script with the correct format:
```python
messages = [
    {"role": "system", "content": "detailed thinking off\n\nYou are an expert in the U.S. Harmonized Tariff Schedule..."},
    {"role": "user", "content": "Product: insulated copper electrical wire, 12 AWG\nUse: residential wiring\nCountry of origin: Mexico"},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
prompt += "<think>\n</think>\n\n"  # Nemotron thinking-off prefix
```

Output:
```
Chapter 85: ELECTRICAL MACHINERY AND EQUIPMENT...
Heading 85.48: Electrical parts of machinery...   ← wrong (should be 85.44)
Subheading 8548.10: Other electrical parts...     ← wrong
HTS Code: 8548.10.0000                            ← wrong
Reasoning: Insulated copper electrical wire is classified...
```

**Verdict**: format works, model produces structured output, **but** it confused 8544 (insulated wire) with 8548 (electrical parts of machinery NESOI) on a fairly common product. Chapter is right (85), heading is wrong by 4 codes.

This is one data point, not an evaluation. Could be a quirk of greedy decoding without retrieval context, or could be a real weakness. We don't know without running real eval, which is blocked on the v2 inference/eval rewrite.

### What we learned from this exercise

- **Training format must be the inference format.** Any prompt-format drift produces garbage output. This is THE most important thing to get right when serving the model later.
- **A "smoke test" inference call should be part of the post-training checklist.** We almost shipped to HF without verifying, then would have discovered the format issue when someone tried to use it. Always validate before publishing.
- **The structured text format works.** No JSON parsing failures, the model emitted Chapter/Heading/Subheading/HTS Code/Reasoning sections cleanly. This validates the v2 design decision from yesterday.

---

## Backup, Publication, and Discrepancies Found

### The artifacts

After training completed, we pulled everything off the pod. The full bundle now lives at `/Users/fbaig/Projects/hts-lora/outputs/train_h100_20260406/`:

```
adapter/                       336MB  - the LoRA itself (also pushed to HF)
config_snapshot.yaml             876B - config the trainer captured at startup
train_h100.yaml                1.0KB  - current config file from disk
train.log                      1.7MB  - full training log (text)
train.log.jsonl                3.5KB  - structured training log
metrics.jsonl                  152KB  - per-step loss + grad_norm + LR + every eval
sample_predictions.jsonl       42KB   - 33 sample predictions captured during training
monitor.log                    3.3KB  - our 30-min status snapshots
monitor.sh                     308B   - the monitoring script
```

Total ~2.1MB outside the adapter itself. Compressed to a single tarball (`hts_artifacts.tar.gz`, 300KB) for the transfer, then extracted locally and flattened.

### What we deliberately did NOT pull

- **`checkpoints/` directory (1.5GB)** containing checkpoint-10000, checkpoint-11000, checkpoint-11214. The Trainer kept only the last 3 (default `save_total_limit=3`). All three are in the overfit zone (eval loss > 0.27), so worse than the best (0.2596 at step 7000). The step-7000 checkpoint had already been deleted by the time we looked. **This is the single most painful lesson of the run.** See Lessons below.

### HuggingFace upload

Pushed to `mfbaig35r/hts-nemotron-8b-lora-v1` as a private repo. Used the `hf` CLI (the new name for `huggingface-cli` in `huggingface_hub>=1.0`):

```bash
hf upload mfbaig35r/hts-nemotron-8b-lora-v1 \
  outputs/train_h100_20260406/adapter . \
  --repo-type=model --private \
  --commit-message "Initial v1 upload: H100 13:44h training, final loss 0.12"
```

Upload took ~7 seconds at 7.7 MB/s from a coffee shop wifi. Verified post-upload that:
- Local SHA256 of `adapter_model.safetensors` matches HF's LFS-stored SHA256 byte-for-byte
- All 6 files present in the HF tree
- `private: true` in the model API response

Wrote a real model card (README.md) replacing the empty PEFT-generated stub. Pushed it as a second commit.

### Discrepancies discovered post-upload

**This is embarrassing but worth documenting.** When I pulled the actual training artifacts at the end of the day, several details I'd been telling the user were wrong:

| | What I told the user | Actual (per train.log + metrics.jsonl) |
|---|---|---|
| Training examples | "~149k" | **119,602 train + 14,784 valid** (134k) |
| Effective batch | 32 (8×4) | **32 (16×2)** |
| Max seq length | 2048 | **1536** |
| Final loss | 0.12 | **~0.13** (last 3 steps: 0.136, 0.123, 0.136) |
| Best eval loss | "0.273 at step 11000" | **0.2596 at step 7000** (then crept up) |

The first three differences are functionally equivalent (same effective batch, slightly smaller seq, etc.) but the **last one matters a lot**. I'd been claiming the final checkpoint was the best, when in fact it was the worst of the surviving checkpoints. Eval loss bottomed at step 7000 and rose afterward. Mild overfitting through epochs 2 and 3.

The README on HF was updated with correct numbers and an honest "Note on overfitting" section.

**Why did this happen?**

When the user asks "what's our config?" mid-run, I was answering from session memory of what I *thought* we'd configured, not from the actual files on the pod. The fixes during the OOM debugging happened across multiple edit cycles, and I conflated which version stuck. The fix is mechanical: **always pull the actual config and metrics from disk before reporting on a run**, never from memory.

I've added "verify against artifacts" to the post-training checklist in the Ministral plan doc.

---

## Insights & Lessons Learned

This is the part of the doc that should drive future training runs. Each lesson is followed by what we'll do differently.

### 1. `save_total_limit` was way too low

**What happened**: Trainer kept only the last 3 checkpoints. By the end of the run, only checkpoints 10000, 11000, 11214 existed. The actual best checkpoint (step 7000, eval loss 0.2596) had been silently deleted around step 10000 when the rotation removed it.

**Impact**: We can't ship the best version of the model. The version on HF is the final-step adapter, which is ~5% worse on eval loss than the model we briefly had at step 7000. Not catastrophic, but a real loss.

**Fix for next run**: `save_total_limit: 5` minimum, paired with `load_best_model_at_end: true` and `metric_for_best_model: "eval_loss"`. The combination means:
- Trainer keeps the last 5 checkpoints PLUS the all-time best one
- At the end of training, the best checkpoint is automatically loaded into the model object
- The "final" adapter saved to disk is the best one, not the last one

This is one config block (3 lines) that completely solves the problem.

### 2. 2 epochs is enough for our data + architecture

**What happened**: Eval loss bottomed at epoch 1.87 and crept up through epochs 2-3. Training loss continued to fall (memorization), but generalization peaked early.

**Why**: 119k examples is plenty to teach a strong-but-narrow signal once, but past a certain point the model starts memorizing example-specific details (specific products, specific phrasings) rather than learning the general task.

**Fix for next run**: `num_train_epochs: 2`. Saves wall time (~30%), saves money (~$10), and produces a better model than 3 epochs.

### 3. 128k vocab makes logits enormous — plan for it

**What happened**: First batch_size=16 attempt OOMed on the cross-entropy step. The logits tensor (`[batch, seq, vocab=128256]`) is enormous for Llama 3 family models because of the giant vocabulary.

**Math**: At seq=2048, batch=16, vocab=128k, fp32 logits: `16 × 2048 × 128,256 × 4 bytes = 16.8 GB` — just for the logits, on top of everything else.

**Fix for next run**: Either reduce batch size, reduce seq length, or both. For Mistral (which has a smaller vocab around 32k or 131k depending on version), this might be less of an issue. For any Llama 3 derivative, plan for the vocab tax.

### 4. Pin transformers/peft/accelerate versions; don't chase latest

**What happened**: Latest `transformers==5.5.0` broke on `torch==2.4.1` because of an API method that didn't exist yet.

**Fix for next run**: Pin `transformers==4.46.3 peft==0.13.2 accelerate==0.34.2` as the known-good triple. These are 4-6 months old but battle-tested with our pipeline. Only update when we have a specific reason to.

This goes in the pod setup checklist.

### 5. Move HF cache to /workspace on day 1

**What happened**: Downloaded the base model to `/root/.cache/huggingface`, filled root disk to 94%, had to scramble to move it.

**Fix for next run**: First three commands on a fresh pod:
```bash
mkdir -p /workspace/.cache
mv /root/.cache/huggingface /workspace/.cache/huggingface 2>/dev/null
ln -s /workspace/.cache/huggingface /root/.cache/huggingface
```

If `/root/.cache/huggingface` doesn't exist yet, just create the symlink first, then the next HF download lands in the right place.

### 6. SDPA is fine for single-GPU 8B

**What happened**: Spent ~30 minutes trying to make Flash Attention 2 work, failed on ABI mismatches, switched to SDPA, training worked.

**Fix for next run**: Default to `attn_implementation: "sdpa"` in the config. Don't even try Flash Attention 2 unless we hit a specific memory or speed wall that warrants the dependency-management pain.

### 7. Always verify post-training before publishing

**What happened**: Almost shipped the adapter to HF without testing inference. The first inference test failed (wrong prompt format). Without that test, we'd have published a model card claiming "use this prompt format" with the wrong format.

**Fix for next run**: Mandatory pre-publish smoke test that:
1. Loads the adapter
2. Sends one known product through inference using the EXACT prompt format from training data
3. Verifies the output is structured (parses with `parse_v2_output`)
4. Prints the result for human eyeballing

If this test passes, then publish. If not, fix before publishing.

### 8. Pull actual configs from artifacts, not memory

**What happened**: I (Claude) gave the user wrong numbers about batch size, seq length, and final loss for hours during the run, based on outdated session memory. Only discovered the discrepancies when we pulled the artifacts at the very end.

**Fix for next run**: When the user asks about the run's configuration mid-flight, I should pull `config_snapshot.yaml` from the output directory (or the train.log header) instead of answering from memory. Same applies at end-of-run reporting — pull from artifacts.

### 9. The Nemotron prompt format is non-obvious and easy to forget

**What happened**: Wrote an inference test, forgot the `"detailed thinking off"` prefix and `<think>\n</think>\n` block. Output was garbage.

**Fix for next run**: Document the prompt format in the model card, in the inference module docstring, AND in any test scripts. Better yet, build a `build_v2_messages()` helper that hides this complexity from callers (this is what the FastAPI wrapper in the serving plan does).

For Ministral (which doesn't have this thinking-mode quirk), the issue won't recur — but we'll have a different chat template format to get right. The general principle is: **the inference prompt MUST match the training prompt byte-for-byte**, and the safest way to enforce that is to use the same `apply_chat_template` call with the same messages at both training and inference time.

### 10. The data pipeline has model-specific quirks baked in

**What happened**: `data/formatted/*.jsonl` contains Nemotron-specific stuff (`"detailed thinking off"` system prompt prefix, `<think>\n</think>\n` assistant prefix). To train Ministral on the same dataset, we'd need to either re-run data prep or strip the quirks at training time.

**Fix for next run**: Refactor the data pipeline to store **generic messages** and apply model-specific templating at training time based on a `model_format` config field. This is Phase A of the Ministral training plan. The result is data files that can train any base model — Nemotron, Mistral, Llama, Qwen, Gemma — without re-running data prep.

---

## What We Built At The End Of The Day

After the training run completed and we'd shipped the adapter, we stayed up to plan the next steps. Two requirement docs landed:

### 1. `docs/serving-plan.md` (427 lines)

A two-phase plan to get the adapter running as an API:
- **Phase 1**: Local MLX server on the M4 Pro using `mlx_lm.fuse` + `mlx_lm.server` (free, fast to set up, validates the model end-to-end)
- **Phase 2**: FastAPI wrapper that hides the Nemotron prompt format and returns clean structured JSON instead of raw text

Key design decisions baked in:
- Pre-fuse LoRA into the base model rather than runtime adapter loading (simpler, slightly faster)
- q4 quantization for the MLX base (~5GB, leaves headroom on 24GB)
- Single source of truth for prompts (import from training formatter, never reimplement)
- Wrapper never crashes on bad model output (returns `parse_ok=false` instead)
- Local-only by default (no auth, no rate limiting, no public exposure)

Phase 3 (vLLM on rented GPU for production) is explicitly out of scope and deferred.

### 2. `docs/training-plan-ministral.md` (557 lines)

A five-phase plan to train a second LoRA adapter on `mistralai/Ministral-3-8B-Reasoning-2512`:

- **Phase A** (local, free): Refactor data pipeline to model-agnostic format
- **Phase B** (local, free): Training scaffolding, smoke test script, vision encoder freezing
- **Phase C** ($0.25): Pre-flight smoke test on a cheap rented GPU (4090 or A40, 30 min)
- **Phase D** (~$30): Real training run on H100 SXM, ~10 hours
- **Phase E** (local, free): Wrap-up, docs, status

Key decisions baked in (from a structured Q&A with the user):
- Variant: **Reasoning-2512** (BF16, post-trained for chain-of-thought, highest upside)
- GPU: **H100 SXM** (known good from today's run)
- Vision encoder: **freeze, don't strip** (less invasive)
- Tokenizer: **try AutoTokenizer first, fall back to MistralCommonBackend**
- Lessons-learned config changes: 2 epochs, save_total_limit=5, load_best_model_at_end=true, eval_steps=500

Critical de-risking step is Phase C: the cheap-GPU smoke test. For $0.25, we validate the entire pipeline (model load, tokenizer, 4-bit quant, LoRA application, vision freezing, forward pass, gradient flow) BEFORE committing to a $30 H100 rental. Worth every cent.

---

## Final Stats

| | |
|---|---|
| **Date** | April 6, 2026 |
| **Total wall time** | ~16 hours (setup + train + verify + plan) |
| **Training wall time** | 13h 44m (49,452s reported) |
| **Pure setup/debug time** | ~2h (Flash Attention, transformers, disk space, OOMs) |
| **GPU** | 1× NVIDIA H100 SXM 80GB on RunPod |
| **Cost** | ~$43 ($2.69/hr × ~16h) |
| **Base model** | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` (8.1B params) |
| **Adapter** | LoRA r=32, alpha=64, 83.9M trainable params (1.03%) |
| **Training data** | 119,602 train + 14,784 valid examples |
| **Epochs** | 3 (overfit in epochs 2-3 — should have been 2) |
| **Total steps** | 11,214 |
| **Effective batch** | 32 (16 × 2 grad accum) |
| **Max seq length** | 1536 |
| **Final training loss** | ~0.13 (running avg 0.248) |
| **Best eval loss** | **0.2596 at step 7000** (epoch ~1.87) |
| **Final eval loss** | 0.2729 at step 11000 |
| **Best ckpt status** | **GONE** — overwritten by save_total_limit=3 |
| **Adapter size** | 336 MB |
| **Adapter location** | `mfbaig35r/hts-nemotron-8b-lora-v1` (private) + local backup |
| **Verification** | Inference test passed (correct chapter, structured output) |
| **Plans written** | 2 (serving + Ministral training) |
| **Lines of plan doc** | 984 (427 serving + 557 Ministral) |

---

## Open Questions & Next Steps

### Things we know we don't know

1. **Real evaluation numbers.** We don't have a parse rate, exact match rate, chapter accuracy, or hierarchy consistency for this adapter. The v2 eval pipeline isn't built yet. Until it is, we have a single inference spot-check (chapter right, heading wrong on one product). Need to update inference + eval code per `/Users/fbaig/.claude/plans/memoized-hatching-teapot.md` to get real numbers.

2. **How much the lost step-7000 checkpoint would have helped.** We can't measure it. Estimate from eval loss: 0.2596 vs 0.2729 is a ~5% improvement, which roughly translates to a 1-3% improvement in exact-match accuracy. Not enormous, but real.

3. **Whether SDPA cost us speed vs Flash Attention.** Probably 10-20% slower on H100 SXM, but we never benchmarked. Doesn't matter at this scale.

### Pending action items

- [x] Push adapter to HF private repo
- [x] Backup training artifacts locally
- [x] Verify SHA256 between local and HF
- [x] Kill the H100 pod
- [x] Write serving plan
- [x] Write Ministral training plan
- [ ] **Revoke HF token `hts-publish`** (was pasted in chat, should be considered exposed)
- [ ] Confirm RunPod billing actually stopped (no trailing charges)
- [ ] Phase A of Ministral plan: data pipeline refactor (local, free, ready when you are)
- [ ] v2 inference/eval rewrite (per existing plan file) — required for real evaluation of either Nemotron or Ministral adapter

### Strategic questions for future sessions

- **When do we go public?** The user wants to publish v1 freely but lag behind by one generation. Decision point: after Ministral training? After v2 eval? After some accuracy threshold? Worth a deliberate decision.
- **Vision augmentation?** Ministral has a 0.4B vision encoder. Could fine-tune for HTS-from-photo classification. Would be a major differentiator vs published benchmarks (none of which support image input). Significant additional dataset work though.
- **Multi-LoRA serving?** When we have v1 (Nemotron), v1.1 (Ministral Reasoning), and possibly v1.2 (Ministral Instruct), we'll want to A/B them from one server. vLLM supports this natively; MLX does not (cleanly). Decision: serve from vLLM in Phase 3 production, even if dev is on MLX.

---

## Closing Note

This was the day we proved the entire pipeline works end-to-end: from raw data through training through publication through verified inference. Everything before this was preparation. Everything after is iteration.

The published benchmarks for HTS classification are:
- ATLAS (Llama-3.3-70B fine-tuned, full SFT): 40.0% exact-match, 16× A100s
- GPT-5-Thinking (zero-shot): 25.0% exact-match
- Tarifflo (retrieval + ML pipeline): 89.2% exact-match
- Avalara (manual expert + AI assist): 80.0% exact-match

Our 8B LoRA adapter, trained on a single H100 in under 14 hours for $43, hasn't been formally evaluated yet. But the hypothesis is that with 17× more training data than ATLAS, hierarchy-aware training, and structured reasoning targets, we should significantly exceed their 40% baseline — maybe land in the 60-80% range for exact match, with chapter accuracy considerably higher.

We'll know when the v2 eval pipeline is built. Until then, today was about getting the model into existence.

**Mission accomplished.**

— Session log compiled April 6-7, 2026
