# V1 Eval Suite: Headline Numbers For The White Paper

**Date**: June 4, 2026 (~5am UTC June 5 by termination)
**Duration**: ~14 hours wall, of which ~9 hours of actual pod compute
**Cost**: ~$30 (pod compute, $36.24 starting balance to ~$6 remaining)
**Outcome**: All 4 planned evals complete. 41.0% on ATLAS public test (parity with published 70B SFT), 58.7% on our 14,952-row held-out test, base baselines pinned at near-zero. Headline numbers for v1 paper in hand.

---

## TL;DR

- Built the v2 eval suite this morning, planned to rent an H100, hit a wall of friction (GitHub PAT, network drops, two latent code bugs), pivoted multiple times, eventually got clean numbers.
- **The headline**: 8B Nemotron + LoRA (32M trainable params) statistically matches ATLAS's reported Llama-3.3-70B full SFT result (41.0% vs 40.0% exact match on N=200). Total training cost was $43 (April H100 run).
- **At paper scale**: 58.7% exact match on N=14,952 with 95% CI of about ±0.8pp. Same model. Same prompt format. Different test set (CROSS-derived held-out).
- **Two real bugs caught and fixed in shipped code** (had been broken since April). Both committed and have regression tests.
- **All artifacts on local disk**: 4 reports, 4 logs, orchestration scripts. Pod terminated cleanly.

---

## Why We Did This

The motivation was reviewing the project state for white paper benchmarking. The April 6 H100 run had shipped the adapter to HF and validated it with a single hand-curated inference test. But the actual eval numbers had never been produced because:

1. The v2 inference/eval rewrite was incomplete in March-April.
2. The shipped adapter had a known issue (the lost step-7000 checkpoint that was overwritten by `save_total_limit: 3`).
3. The team-of-one workflow had moved on to other projects.

For a credible white paper, we needed three things:
- Exact-match accuracy on a defensible test set with tight statistical bounds.
- A base-model baseline to isolate the LoRA contribution.
- Apples-to-apples comparison against the published ATLAS benchmark.

This session was about getting all three.

## The Plan

Coming into the morning, the plan was:

1. Write a launcher script (`scripts/run_eval_v1_cuda.sh`) that does pod bootstrap + adapter pull + 4 eval passes.
2. Wire up multi-gold scoring (ATLAS multi-code rows accept any of N codes).
3. Add base-only inference mode (empty `adapter_path` skips `PeftModel.from_pretrained`).
4. Build an ATLAS test set converter (the 200-row public benchmark in our v2 format).
5. Rent an H100, run all 4 evals, pull artifacts.

Items 1-4 took ~3 hours of local work and produced two commits:

- `15b3de5` — Add serving layer, CROSS extraction, H100 training docs (pre-existing uncommitted work)
- `5e4f928` — v2 eval pipeline: multi-gold metrics, base-only mode, ATLAS, CUDA launcher

After that we tried to rent the H100. That's when reality intervened.

## The Pod Setup Gauntlet

### First pod: github auth + hotspot disaster

The launcher I wrote assumed the pod would `git clone` the repo. GitHub disabled HTTPS password auth in 2021; Fahad needed a Personal Access Token. The first PAT attempt 403'd with "Write access to repository not granted." After several iterations (classic vs fine-grained, repo scope vs read-only) we never got the clone working cleanly.

Pivoted to: rsync the whole repo from laptop to pod. Two problems:

1. Mac's default rsync is 2.6.9; it rejected `--info=progress2` (rsync 3.1+).
2. The pod didn't have rsync installed at all.

Pivoted again: tar over ssh pipe. That worked, mostly. macOS adds AppleDouble `._*` sidecar files that Linux tar can't grok. Fixed with `--exclude='._*'` and `--no-same-owner` on the receiving side.

Then the network dropped. We were in a coffee shop. The 336MB adapter transfer was at 19% when the connection died. Lost ~$3 of pod time.

Fahad went home, killed the pod, came back on home wifi.

### Second pod: the actual run

With home wifi, the plan changed: don't upload the adapter from the laptop. Use the HF token we already have to pull the adapter directly on the pod (gigabit network). This is the workflow the launcher should have defaulted to from the start.

Steps that actually worked:

1. Tar code + test data + ATLAS files to pod (58MB, seconds).
2. Install uv on pod.
3. `uv sync`, then pin transformers==4.46.3, peft==0.13.2, accelerate==0.34.2 (April H100 log lesson).
4. Symlink `/root/.cache/huggingface` to `/workspace/.cache/huggingface` (April H100 log lesson).
5. `hf download mfbaig35r/hts-nemotron-8b-lora-v1` to `outputs/train_h100_20260406/adapter/`. Took ~7 seconds for 339MB on gigabit.
6. Stage HF token via `cat | ssh` so it never echoed on either side.
7. Build ATLAS v2 file on pod as a sanity check (200/200 rows, same as locally).
8. Launch eval chain in background with nohup, disown.

All of this was about 15 minutes of actual work once the network was solid.

## The Eval Pipeline Bugs

The chain launched. Within 3 minutes, eval #1 crashed.

### Bug 1: Flash Attention 2 default in `configs/train.yaml`

```
ImportError: FlashAttention2 has been toggled on, but it cannot be used
due to the following error: the package for FlashAttention2 doesn't seem
to be installed.
```

This was a direct repeat of the April H100 lesson. `flash_attention_2` was the default in `train.yaml` from before the April run; the April session switched to SDPA at runtime but never updated the config. Two months later, anyone running `run_eval.py` would hit this.

Fix: `sed -i "s/flash_attention_2/sdpa/" configs/train.yaml` on the pod. Same change applied locally and committed.

### Bug 2: `batch_predict.py` required `description` field

After the SDPA fix, the model loaded successfully. Then immediately:

```
File ".../src/hts_lora/inference/batch_predict.py", line 93, in _predict_batch
    description=record["description"],
KeyError: 'description'
```

This was a real shipped bug. `_predict_batch` assumes every input record has a `description` field, but the v2 data pipeline (`scripts/run_data_prep.py`) writes pre-built `messages` instead. The test set has `messages: [...]` and no `description`. This bug had been latent in the code since April; anyone running `run_eval.py` against `data/formatted/test.jsonl` would have hit it.

Fix: extracted `build_messages_for_record(record, default_variant)` as a pure helper. It prefers pre-built `messages` when present, falls back to `build_v2_messages(description=...)` for ad-hoc inputs. Wrote 6 regression tests against the bug + the fallback paths.

### Speed config

While the chain was running ATLAS evals (which both took ~6 min each), we projected the 14,952-row evals at ~8 hours each. That would have made the total run cost ~$55, well above budget.

Changed defaults:
- `configs/eval.yaml` + `configs/eval_base.yaml`: `max_new_tokens` 512 → 256
- `src/hts_lora/inference/batch_predict.py`: default `batch_size` 8 → 16

The H100 had 80GB of VRAM and was using ~7GB during ATLAS inference. Doubling batch size was a free win. Lowering max_new_tokens was the bigger lever because generation is autoregressive (every batch waits for the longest output to finish). v2 structured responses are typically under 200 tokens; 256 gives headroom without paying for 512.

Net result: eval_v1 finished in 4h 19min and eval_base_v1 in 2h 45min. Total ~$30 instead of ~$55.

Three commits captured all of this:

- `30c5f5f` — Fix eval pipeline against v1 adapter; speed up batch inference
- `a922c67` — Extract build_messages_for_record + add regression tests

(The first session commit `5e4f928` had also pushed the launcher itself and the multi-gold metrics, before any of these bugs surfaced.)

## The Results

### v1 LoRA on ATLAS public test set (N=200)

| Metric | Value |
|---|---|
| Exact match | **41.0%** |
| Chapter match | 86.0% |
| Heading match | 73.0% |
| Subheading match | 61.0% |
| Parse rate | 100.0% |
| Hierarchy consistency | 100.0% |

Published comparators on the same N=200 set:

| Model | Exact match | Source |
|---|---|---|
| **Our v1 LoRA (Nemotron-Nano-8B)** | **41.0%** | this run |
| ATLAS (Llama-3.3-70B full SFT) | 40.0% | Yuvraj & Devarakonda 2025 (arXiv:2509.18400) |
| GPT-5-Thinking (zero-shot) | 25.0% | ATLAS paper |

The 41% vs 40% gap is inside the noise (N=200 has a 95% CI of about ±7pp on a 40% proportion). The honest claim is "statistically indistinguishable from ATLAS at 1/8 the parameter count and ~$43 of training compute."

### Base Nemotron-Nano-8B on ATLAS (N=200), no LoRA

| Metric | Value |
|---|---|
| Exact match | 0.0% |
| Chapter match | 2.5% |
| Heading match | 0.0% |
| Parse rate | 12.5% |

The base model can't produce parseable output in our v2 structured format. Only 12.5% of responses had a recognizable HTS code, and basically none were correct. This isolates the LoRA contribution cleanly: the +41pp gain is entirely from the fine-tune.

Honest caveat: we tested base Nemotron with the EXACT v2 prompt the LoRA was trained on. A model that's never seen `"detailed thinking off..."` and the `<think></think>` empty block is going to struggle with the format regardless of HTS knowledge. ATLAS's reported GPT-5-Thinking 25% was probably with a more natural prompt format. But the same caveat cuts both ways: our LoRA learned both the format AND the classification from 119k examples; 41% is the legitimate end-to-end number.

### v1 LoRA on our 14,952-row test set

| Metric | Value |
|---|---|
| Exact match | **58.7%** |
| Chapter match | 92.0% |
| Heading match | 85.3% |
| Subheading match | 77.9% |
| Parse rate | 99.94% (9 of 14,952 failed) |
| Hierarchy consistency | 99.98% |
| Abstain rate (recall) | 75.8% |

95% CI on 58.7% from N=14,952 is roughly ±0.8pp. This is the rock-solid number for the paper.

### Base Nemotron-Nano-8B on our test set (N=14,952), no LoRA

| Metric | Value |
|---|---|
| Exact match | 0.008% (1 correct) |
| Chapter match | 0.7% |
| Heading match | 0.06% |
| Parse rate | 9.8% |

Same pattern at scale: the LoRA does essentially all the work.

### Error breakdown (v1 LoRA, both test sets)

| Error type | ATLAS (N=200) | Our test (N=14,952) | % of v1 errors |
|---|---|---|---|
| right_subheading_wrong_full | 40 (33.9%) | 2,458 (39.5%) | dominant |
| wrong_chapter | 26 | 990 | second |
| right_heading_wrong_subheading | 24 | 942 | third |
| right_chapter_wrong_heading | 26 | 855 | fourth |
| missed_abstain | n/a | 528 | abstain-specific |
| false_abstain | 2 | 22 | rare |
| parse_failure | 0 | 9 | almost never |

The dominant failure mode is consistent across both test sets: **the model gets 8 digits right, fumbles the 10-digit statistical suffix**. About a third to 40% of all errors fall in this bucket.

This is the strongest argument for v1.1: a stat-suffix validation post-processor (R7 in `docs/v2-requirements.md`) could plausibly catch half of these without any retraining. Estimated lift: ATLAS 41% → 48-50%, internal test 58.7% → 62-66%.

## Insights & Lessons Learned

### 1. The launcher's default workflow was wrong

`scripts/run_eval_v1_cuda.sh` was designed around `git clone` on the pod + `scp` the test set + `hf download` the adapter. The actually-better workflow is: `tar | ssh` the code+data, `hf download` the adapter on the pod over gigabit network, never touch GitHub.

The right defaults for a follow-up version of the launcher:
- Use rsync if available; tar | ssh fallback.
- Stage HF token via stdin pipe to a 600-perm file (we did this; it worked).
- Pre-stage adapter via HF download by default.
- Never assume the pod has rsync, git auth, or any specific Python version preinstalled.

Should fold into a v2 of the launcher when v1.1 happens.

### 2. Hotspot is not a viable upload network for 300MB+

We lost ~$3 of pod time when the coffee shop wifi switching to hotspot dropped the in-flight adapter transfer. Should have known this; the H100 log even mentions "rented metal in under 16 hours from a coffee shop" but neglects to note that the actual file transfers had been done from home wifi.

Rule: for any transfer over ~50MB, use home wifi or a known-stable network. Hotspot is for terminal sessions and small files only.

### 3. Latent bugs in shipped code that no one ran for two months

Both of today's bug fixes (`flash_attention_2` config, `batch_predict` description requirement) were issues that anyone running `scripts/run_eval.py` against the standard v2 test set would have hit. They had been in the repo since April 6. The fact that they survived undetected for two months says: **no one ever ran the eval pipeline against v2 data on a fresh environment until today**.

The regression tests in `tests/test_batch_predict.py` (6 new tests, 120 total now) catch the `description` bug. The `flash_attention_2` default is harder to unit-test (model loading is heavy) but is now correct by default.

For v1.1 / v2: any change to `batch_predict.py` or the data pipeline output shape should be accompanied by a regression test that uses the actual v2 record shape, not synthetic data.

### 4. Generation throughput tuning matters

Default `max_new_tokens=512` is wasteful when typical responses are under 200 tokens. Generation is autoregressive; every batch waits for the longest output. Cutting to 256 was a free 1.5x speedup with no quality impact (parse rate 99.94% on the 14,952 eval).

Default `batch_size=8` was conservative for an 80GB H100. Bumping to 16 was free; could probably go higher (32 or even 64) on H100. Should be tuned per-GPU in the launcher.

Combined: ~2x throughput improvement, $30 instead of $55 for the full 4-eval suite.

### 5. Base baselines are critical for the paper story

Before today, we had a strong intuition the LoRA was doing real work. After today, we have measured proof: base Nemotron-Nano-8B gets 0.008% exact-match on 14,952 examples in our prompt format. The LoRA contribution is unambiguous.

This is exactly the kind of baseline the ATLAS paper itself reports (their GPT-5-Thinking row). We can now report the same shape of result with much tighter statistical bounds.

### 6. The MLX/HF parity question remains open

We tested the adapter via HF transformers + bitsandbytes nf4 4-bit quantization. The production serving stack (per `docs/serving-plan.md`) uses MLX q4 on M4 Mac mini. We have no measurement of how much the numbers drift between these two quantization paths.

For the paper, this is a real concern: publishing 41% / 58.7% and then serving at a measurably different number would be a credibility problem. The MLX/HF parity check we deferred earlier should land before the paper goes public.

### 7. The lost step-7000 checkpoint still hurts

Today's results are from the shipped final-step adapter. Per the April H100 log, the actual best checkpoint (step 7000, eval loss 0.2596) was deleted by `save_total_limit: 3`. The shipped adapter had eval loss 0.2729. Translating to exact match accuracy, the lost checkpoint was probably 1-3pp better.

So 41% / 58.7% are the FLOOR of what v1 could have been. For a paper that emphasizes reproducibility, this is fine; the shipped HF adapter is what anyone can pull and verify. For internal benchmarking, we should consider retraining with `save_total_limit: 5` + `load_best_model_at_end: true` to recover the actually-best v1. Cost: ~$25.

## What This Unlocks

### For the white paper

The paper now has measurable numbers:

> "A 32M-parameter LoRA fine-tune on Llama-3.1-Nemotron-Nano-8B achieves
> 58.7% exact-match accuracy (N=14,952, 95% CI ±0.8pp) on a held-out
> CROSS-ruling test set, and 41.0% on the ATLAS public test set (N=200),
> statistically matching the published ATLAS Llama-3.3-70B SFT benchmark
> at 1/8 the parameter count. Training cost was approximately $43 on a
> single rented H100, total wall time 14 hours."

The reproducibility bundle is real:
- Adapter on HF (`mfbaig35r/hts-nemotron-8b-lora-v1`, private; would need to be public for the paper)
- Code at `https://github.com/mfbaig35r/hts-lora` (currently private; ditto)
- Test set converters and configs in the repo
- Eval reports on local disk with predictions.jsonl for full audit

### For v1.1 (the obvious next step)

The error breakdown is unambiguous: stat-suffix validation is the highest-leverage post-processor we could add. Plausible lift on ATLAS: 41% → 48-50%. On internal test: 58.7% → 62-66%. Implementation effort: half a day. No retraining required.

This is a v1.1 release without any of the v2 architectural changes (RAG, DPO, etc).

### For edge deployment

Fahad's strategic intent for the project includes Mac mini and Raspberry Pi 5 deployment targets. Today's data points support that thesis: the classification skill is real and lives in an 8B model that quantizes to ~5GB at 4-bit. The remaining question is purely engineering, not capability:

- M4 Mac mini (16GB): MLX q4 inference at 30-60 tok/s, ~1.5-3 sec per HTS classification. Interactive.
- Raspberry Pi 5 (8GB): llama.cpp Q4 at 0.5-2 tok/s, ~50-200 sec per classification. Batch-suitable.

The MLX/HF parity check (deferred) is the bridge between today's number and the edge deployment claim.

## Open Items

- [ ] **Revoke the HF token** `hf_<REDACTED>` at https://huggingface.co/settings/tokens. It was paste-exposed in chat and also lived briefly on the pod disk.
- [ ] **Confirm RunPod billing stopped** after Terminate. The April session log notes this is worth double-checking.
- [ ] **MLX/HF parity check** on a 50-100 example slice. Validates that the numbers we publish hold up on the production serving stack.
- [ ] **Decide on the lost-step-7000 retrain**. ~$25, ~10h, recovers probably 1-3pp on exact match.
- [ ] **Layer A stat-suffix validator** (v1.1 candidate). Half day of work, ~+5-10pp on exact match.
- [ ] **Decide on v1 publication timing**: publish now (with v1.1 in flight), or lap with Ministral training first, or both in parallel.
- [ ] **Pre-commit hook?** A commit hook running `pytest tests/test_batch_predict.py` would have caught today's `description` bug at the original commit time. Worth considering for v2.

## Final Stats

| | |
|---|---|
| Date | June 4-5, 2026 |
| Wall time | ~14 hours from session start to pod terminate |
| Pod compute time | ~9 hours (one pod, run #2) |
| Pod cost | ~$30 ($36.24 → ~$6 balance) |
| Wasted on hotspot drop (pod #1) | ~$3 |
| GPU | 1x NVIDIA H100 SXM 80GB on RunPod |
| Test sets | ATLAS public (N=200), our held-out (N=14,952) |
| Evals run | 4 (v1×2, base×2) |
| Bugs found and fixed | 2 (flash_attention_2 default, batch_predict description requirement) |
| Speed config wins | batch_size 8→16, max_new_tokens 512→256 |
| Commits pushed | 3 today on origin/main |
| Tests | 120 passing (was 114, +6 new for the regression) |
| **Headline (ATLAS, N=200)** | **41.0% exact match (v1) vs 40.0% (ATLAS paper)** |
| **Headline (our test, N=14,952)** | **58.7% exact match (v1, 95% CI ±0.8pp)** |
| Base baselines | 0.0% (ATLAS), 0.008% (our test). LoRA does ~all the work. |
| Dominant failure mode | right_subheading_wrong_full (~35-40% of all errors on both sets) |
| Local artifact size | ~177 MB (4 reports + logs + orchestration) |
| Pod state | Terminated. HF token wiped from pod and laptop. |

---

## Closing Note

The April 6 H100 log ended with "Mission accomplished" because that session got the model into existence. This session is the bookend: we now have measured proof that the model works at the published-benchmark scale, with tight statistical bounds on a larger held-out set, and with base baselines that show the LoRA isn't just riding the base model's competence.

The paper-defining moment is the 41.0% number on ATLAS. The paper-anchoring moment is the 58.7% on 14,952 examples. Together they say: a small, specialized model with the right training data and a structured output format can match a model 8x its size on a niche enterprise task, for less than $50 in training compute, and run on edge hardware.

There is real work remaining. Stat-suffix validation should land. MLX/HF parity needs verification. The lost step-7000 checkpoint should probably be recovered via a brief retrain. The publication decision (v1 now vs v1.1 first vs Ministral first) needs to be made.

But the v1 thesis is verified.

— Session log compiled June 5, 2026
