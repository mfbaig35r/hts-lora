# HTS LoRA Serving Plan

Requirements and implementation plan for serving the trained HTS LoRA adapter
(`mfbaig35r/hts-nemotron-8b-lora-v1`) as a usable API.

## Overview

We have a trained LoRA adapter on top of `nvidia/Llama-3.1-Nemotron-Nano-8B-v1`
that classifies products into the U.S. Harmonized Tariff Schedule. To make it
useful — for orchestration with larger frontier models, internal tools, batch
classification, or end-user demos — we need it behind an API.

This document specifies a **two-phase approach**:

| Phase | What | Why |
|---|---|---|
| **Phase 1** | Local MLX server on the M4 Pro | Free, fast to set up, validates the model end-to-end, perfect for dev/MVP |
| **Phase 2** | FastAPI wrapper with structured I/O | Hides the Nemotron prompt format, returns clean JSON, makes the API actually usable by other code |

Phase 3 (production / rented GPU / vLLM) is **out of scope** for this document
and is referenced only as future work.

## Goals

1. Run the trained adapter locally on the M4 Pro (24GB) with no recurring cost
2. Expose an HTTP API that other code can call to classify products
3. Hide the Nemotron prompt-format quirks (system prefix, `<think>` block) from callers
4. Return structured JSON (chapter, heading, subheading, HTS code, reasoning) — not raw text
5. Be usable as a building block for the orchestrator architecture (8B-as-tool for a frontier model)

## Non-goals

- Production-grade serving (concurrency, autoscaling, multi-tenancy) — that's Phase 3
- Public internet exposure — local-only by default
- Authentication / rate limiting — local-only, single user
- Web UI / chat interface — API only
- Real-time streaming — request/response is fine for v1
- Multiple LoRA adapters served from one base model — single adapter for now

## Architecture

```
┌─────────────────┐      HTTP/JSON       ┌────────────────────┐      OpenAI-compat      ┌──────────────────┐
│  Caller         │ ─────────────────►   │  FastAPI wrapper   │  ────────────────────►  │  mlx_lm.server   │
│  (orchestrator, │                       │  (Phase 2)         │                          │  (Phase 1)       │
│  CLI, notebook) │ ◄─────────────────    │                    │  ◄────────────────────  │                  │
└─────────────────┘   {hts_code, ...}    │  - prompt format   │     raw text response    │  - fused MLX     │
                                          │  - parse output    │                          │    model         │
                                          │  - validation      │                          │  - localhost     │
                                          └────────────────────┘                          │    :8080         │
                                                                                           └──────────────────┘
                                                                                                    ▲
                                                                                                    │
                                                                              ┌─────────────────────┴──────────┐
                                                                              │  models/nemotron-hts-fused/    │
                                                                              │  (base + LoRA merged → MLX)    │
                                                                              └────────────────────────────────┘
```

**Caller never sees**:
- The `"detailed thinking off"` system prompt
- The `<think>\n</think>\n` assistant prefix
- The structured-text output format (Chapter NN: ..., HTS Code: ..., etc.)

**Caller sees**:
```json
POST /classify
{
  "description": "insulated copper electrical wire, 12 AWG",
  "materials": "copper, PVC insulation",
  "use": "residential wiring",
  "country_of_origin": "Mexico"
}

Response:
{
  "hts_code": "8544.42.9000",
  "chapter": {"code": "85", "description": "ELECTRICAL MACHINERY..."},
  "heading": {"code": "85.44", "description": "Insulated wire, cable..."},
  "subheading": {"code": "8544.42", "description": "Other electric conductors..."},
  "reasoning": "Classified under 8544 because...",
  "provides_for": "Insulated electric conductors...",
  "is_abstention": false,
  "abstention_reason": null,
  "parse_ok": true,
  "model": "hts-nemotron-8b-lora-v1",
  "latency_ms": 1842
}
```

---

## Phase 1: Local MLX Server

### Goal

Run the trained adapter on the M4 Pro as an OpenAI-compatible HTTP server with
zero ongoing cost, reachable at `http://localhost:8080/v1/chat/completions`.

### Approach

Use [mlx-lm](https://github.com/ml-explore/mlx-lm)'s built-in `mlx_lm.server`,
which speaks the OpenAI Chat Completions protocol natively. Pre-fuse the LoRA
adapter into the base model so we serve a single artifact (simpler than
adapter-on-the-fly loading, and slightly faster).

### Functional requirements

| ID | Requirement |
|---|---|
| F1.1 | The base model (`nvidia/Llama-3.1-Nemotron-Nano-8B-v1`) must be converted to MLX format and quantized to 4-bit (q4) to fit comfortably in 24GB of unified memory |
| F1.2 | The LoRA adapter from `mfbaig35r/hts-nemotron-8b-lora-v1` must be fused into the converted MLX base model, producing a single `models/nemotron-hts-fused/` directory |
| F1.3 | An MLX server must be runnable via `uv run mlx_lm.server --model models/nemotron-hts-fused --port 8080` (or equivalent) |
| F1.4 | The server must respond to standard OpenAI Chat Completions requests at `POST /v1/chat/completions` |
| F1.5 | Inference latency for a typical HTS classification request (~200 input tokens, ~200 output tokens, greedy decoding) must be < 10s on the M4 Pro |
| F1.6 | A smoke-test script must verify the server returns a parseable structured response for a known test product |

### Non-functional requirements

| ID | Requirement |
|---|---|
| NF1.1 | All MLX dependencies live under the existing `mlx` optional dep group in `pyproject.toml` (already present) |
| NF1.2 | The fused model must NOT be checked into git (add `models/` to `.gitignore`) |
| NF1.3 | Conversion + fusion must be reproducible from a single CLI command |
| NF1.4 | The fused model directory must be < 6GB (q4-quantized 8B) |
| NF1.5 | The server must run as a foreground process — no daemonization, no PID files, no systemd |

### Implementation steps

1. **Add `.gitignore` entry** for `models/` (if not already ignored)

2. **Create `scripts/run_serve_mlx.py`** — a Typer CLI with three subcommands:

   ```
   uv run python scripts/run_serve_mlx.py convert
       # Downloads base model from HF, converts to MLX q4, writes to models/nemotron-base-mlx/
       # Skips if already exists.

   uv run python scripts/run_serve_mlx.py fuse \
       --adapter mfbaig35r/hts-nemotron-8b-lora-v1
       # Downloads adapter from HF (or uses local path),
       # calls mlx_lm.fuse to merge into base,
       # writes to models/nemotron-hts-fused/

   uv run python scripts/run_serve_mlx.py serve --port 8080
       # Runs mlx_lm.server pointed at the fused model
   ```

   Each subcommand wraps the underlying `mlx_lm` CLIs (`mlx_lm.convert`,
   `mlx_lm.fuse`, `mlx_lm.server`) with friendly defaults and the right
   arguments for our use case.

3. **Create `scripts/smoke_test_serve.py`** — sends one OpenAI-formatted
   chat completion request to `localhost:8080/v1/chat/completions` with the
   correct Nemotron prompt format and prints the raw response. Used to
   verify the server is working before building the FastAPI wrapper.

4. **Document in `docs/serving-plan.md`** (this doc) and add a "Serving"
   section to the project README.

### Acceptance criteria

- [ ] `uv run python scripts/run_serve_mlx.py convert` produces `models/nemotron-base-mlx/` (~5GB, q4-quantized)
- [ ] `uv run python scripts/run_serve_mlx.py fuse --adapter mfbaig35r/hts-nemotron-8b-lora-v1` produces `models/nemotron-hts-fused/`
- [ ] `uv run python scripts/run_serve_mlx.py serve` starts a server on port 8080 within 30s
- [ ] `curl http://localhost:8080/v1/models` returns the model name
- [ ] `python scripts/smoke_test_serve.py` returns a structured HTS classification (Chapter/Heading/Subheading/HTS Code/Reasoning/Provides for) for a test product like "insulated copper electrical wire"
- [ ] End-to-end latency for the smoke test is < 10s
- [ ] Memory usage during inference is < 18GB (leaving headroom on 24GB)

### Open questions

- **Q1.1**: Should we fuse from the HF Hub adapter or from the local copy at `outputs/train_h100_20260406/adapter/`? **Decision**: Use the local copy by default, fall back to HF Hub via a `--adapter` flag. Faster and avoids needing HF auth in the convert step.
- **Q1.2**: q4 vs q8 quantization for the base model? **Decision**: Start with q4 (~5GB). If output quality regresses noticeably from the un-quantized adapter inference, retry with q8 (~9GB). Both fit in 24GB.
- **Q1.3**: Should we expose a `--share` mode that starts a Tailscale or ngrok tunnel? **Decision**: No, out of scope for Phase 1. Local-only.

### Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| MLX doesn't support the Nemotron architecture cleanly | Low | Llama 3.1 family is well-supported; Nemotron is a Llama 3.1 derivative. Validate during convert step. |
| Quality degradation from fusing q4 base + LoRA | Medium | Spot-check vs the H100 inference results we already verified. If bad, retry at q8. |
| `mlx_lm.server` doesn't apply chat templates the same way as transformers | Medium | Test early; if needed, send pre-templated prompts via the `prompt` field instead of `messages` |
| 24GB OOM during inference with long sequences | Low | Cap `max_new_tokens` at 512; we already trained with `max_seq_length=1536` |

---

## Phase 2: FastAPI Wrapper

### Goal

Wrap the MLX server with a thin FastAPI service that gives callers a clean
domain-specific interface (`POST /classify`) and hides all the LLM mechanics.

### Approach

A small FastAPI app that:
1. Accepts a structured HTS classification request (description + optional materials/use/country)
2. Builds the v2 prompt format with the Nemotron-specific quirks
3. Calls the MLX server's OpenAI-compatible endpoint
4. Parses the structured text response using the existing `parse_output.py` module
5. Returns clean typed JSON to the caller

### Functional requirements

| ID | Requirement |
|---|---|
| F2.1 | Expose `POST /classify` accepting JSON `{description, materials?, use?, country_of_origin?}` |
| F2.2 | Build the prompt using the v2 system prompt + Nemotron `"detailed thinking off"` prefix and `<think>\n</think>\n` assistant prefix |
| F2.3 | Call the MLX server via the OpenAI Python client (`openai>=1.30.0` is already a project dep) |
| F2.4 | Parse the raw response using `hts_lora.inference.parse_output.parse_v2_output` (already implemented) |
| F2.5 | Return a `ClassificationResponse` Pydantic model with all parsed fields plus `latency_ms`, `model`, `parse_ok` |
| F2.6 | Expose `GET /health` returning `{"status": "ok", "upstream": "<mlx_server_url>", "model": "..."}` |
| F2.7 | Expose `GET /` returning a minimal HTML page with curl examples and API docs link |
| F2.8 | If parsing fails (`parse_ok=False`), return HTTP 200 with the raw text in a `raw` field — never crash on bad model output |
| F2.9 | If the upstream MLX server is unreachable, return HTTP 503 with a clear error message |
| F2.10 | Configurable upstream URL via env var `HTS_MLX_URL` (default `http://localhost:8080/v1`) |

### Non-functional requirements

| ID | Requirement |
|---|---|
| NF2.1 | New code lives under `src/hts_lora/serving/` (new module) |
| NF2.2 | Add `fastapi>=0.110` and `uvicorn>=0.27` as a new optional dep group `serving` in `pyproject.toml` |
| NF2.3 | The FastAPI app must be runnable via `uv run python scripts/run_serve_api.py` |
| NF2.4 | Auto-generated OpenAPI schema at `/docs` (FastAPI default) must accurately reflect the request/response models |
| NF2.5 | All public types are Pydantic v2 BaseModels |
| NF2.6 | Single-file app under 250 lines — this is glue code, not a framework |

### API contract

#### `POST /classify`

**Request body**:
```json
{
  "description": "insulated copper electrical wire, 12 AWG, stranded",
  "materials": "copper conductor, PVC insulation",
  "use": "residential and commercial building wiring",
  "country_of_origin": "Mexico"
}
```

Only `description` is required. All other fields are optional and influence
which of the 4 input variants the prompt builder uses (matching the training
distribution).

**Response body (success)**:
```json
{
  "hts_code": "8544.42.9000",
  "chapter": {"code": "85", "description": "ELECTRICAL MACHINERY..."},
  "heading": {"code": "85.44", "description": "Insulated wire, cable..."},
  "subheading": {"code": "8544.42", "description": "Other electric conductors..."},
  "reasoning": "The product is an insulated electrical conductor...",
  "provides_for": "Insulated electric conductors fitted with connectors...",
  "is_abstention": false,
  "abstention_reason": null,
  "parse_ok": true,
  "raw": null,
  "model": "hts-nemotron-8b-lora-v1",
  "latency_ms": 1842
}
```

**Response body (abstention)**:
```json
{
  "hts_code": null,
  "chapter": null,
  "heading": null,
  "subheading": null,
  "reasoning": null,
  "provides_for": null,
  "is_abstention": true,
  "abstention_reason": "The product description is too vague to determine a specific HTS classification. Please provide additional details about materials and intended use.",
  "parse_ok": true,
  "raw": null,
  "model": "hts-nemotron-8b-lora-v1",
  "latency_ms": 1124
}
```

**Response body (parse failure — model output didn't match the expected format)**:
```json
{
  "hts_code": null,
  "...": null,
  "is_abstention": false,
  "parse_ok": false,
  "raw": "<the raw model output>",
  "model": "hts-nemotron-8b-lora-v1",
  "latency_ms": 1530
}
```

#### `GET /health`

```json
{
  "status": "ok",
  "upstream": "http://localhost:8080/v1",
  "upstream_reachable": true,
  "model": "hts-nemotron-8b-lora-v1"
}
```

Returns HTTP 503 if the upstream MLX server is unreachable.

### Implementation steps

1. **Add `serving` optional dep group** to `pyproject.toml`:
   ```toml
   serving = [
       "fastapi>=0.110",
       "uvicorn[standard]>=0.27",
   ]
   ```

2. **Create `src/hts_lora/serving/` module**:
   - `__init__.py` — empty
   - `models.py` — Pydantic request/response models (`ClassifyRequest`, `ClassificationResponse`, `HealthResponse`, `HierarchyLevel`)
   - `prompts.py` — `build_v2_messages(description, materials, use, country)` — single source of truth for the v2 prompt format. Reuses or imports from the existing formatter code if possible.
   - `client.py` — Thin OpenAI client wrapper that talks to the MLX server, handles the `<think>\n</think>\n` prefix injection, and times the call.
   - `app.py` — FastAPI app definition with the three routes (`/classify`, `/health`, `/`)

3. **Create `scripts/run_serve_api.py`** — Typer CLI that runs uvicorn:
   ```
   uv run python scripts/run_serve_api.py --port 8000 --upstream http://localhost:8080/v1
   ```

4. **Add tests** in `tests/test_serving.py`:
   - Test prompt builder against all 4 input variants (matching training)
   - Test `parse_output` integration: feed a known good response, verify it parses
   - Test the FastAPI app with `httpx.AsyncClient` and a mocked OpenAI client (no real MLX server in CI)
   - Test abstention parsing path
   - Test parse-failure path returns HTTP 200 with `parse_ok=false`
   - Test `/health` returns 503 when upstream unreachable

5. **Update README** with a "Quick start: serving" section showing the two-step
   sequence (start MLX server, start FastAPI wrapper, curl `/classify`).

### Acceptance criteria

- [ ] `uv pip install -e ".[mlx,serving]"` installs cleanly
- [ ] With the Phase 1 MLX server running on port 8080, `uv run python scripts/run_serve_api.py --port 8000` starts the FastAPI app
- [ ] `curl -X POST http://localhost:8000/classify -H "Content-Type: application/json" -d '{"description": "insulated copper wire"}'` returns a structured JSON response with a valid HTS code in chapter 85
- [ ] `curl http://localhost:8000/health` returns `{"status": "ok", ...}` when MLX is running
- [ ] `curl http://localhost:8000/health` returns HTTP 503 when MLX is stopped
- [ ] FastAPI auto-generated docs at `http://localhost:8000/docs` show the request/response schema correctly
- [ ] All `tests/test_serving.py` tests pass: `uv run python -m pytest tests/test_serving.py -v`
- [ ] End-to-end latency (caller → FastAPI → MLX → response) is < 11s for a typical request (Phase 1 MLX latency + ~50ms wrapper overhead)
- [ ] Wrapper code is < 250 lines total across `serving/`

### Open questions

- **Q2.1**: Should the FastAPI wrapper also expose a streaming endpoint (`POST /classify/stream` with SSE)? **Decision**: No for v1. Add later if a caller actually needs it.
- **Q2.2**: Should we implement request batching (caller sends a list of descriptions)? **Decision**: No for v1. Single request only. Batch is a Phase 3 concern when we have vLLM with continuous batching.
- **Q2.3**: Should the wrapper retry on parse failures with a higher temperature? **Decision**: No. If the model output doesn't parse, return `parse_ok=false` and let the caller decide. Silent retries hide model quality issues.
- **Q2.4**: Should the prompt builder live in `serving/prompts.py` or be imported from the existing data formatter code (`src/hts_lora/data/formatters.py` or wherever the training prompts are built)? **Decision**: Import from the existing formatter module. Single source of truth — the serving prompt MUST match the training prompt exactly, and duplicating it risks drift.
- **Q2.5**: Configuration via env vars or a YAML config file? **Decision**: Env vars for runtime config (`HTS_MLX_URL`, `HTS_MODEL_NAME`), hardcoded defaults for everything else. Don't introduce a config file for ~3 settings.

### Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| MLX server's chat template doesn't match training format exactly | Medium | Test early in Phase 1; if it differs, send pre-templated `prompt` instead of structured `messages` |
| The existing `parse_output.parse_v2_output` doesn't handle some real model outputs | Medium | Add fixtures from real Phase 1 outputs to `tests/test_parse_output.py`; iterate the parser if needed. The function already exists at `src/hts_lora/inference/parse_output.py` so this is incremental work, not greenfield. |
| The wrapper adds noticeable latency | Low | Pure Python overhead for prompt building + parsing should be < 10ms; FastAPI/uvicorn add < 5ms; total wrapper overhead < 50ms is achievable |
| Prompt template drift between training and serving | High if not addressed | Single-source-of-truth: import `build_messages` from the existing formatter module, don't reimplement |

---

## Implementation order

Strict ordering — each step depends on the previous being verified working.

1. **Phase 1 step 1**: `scripts/run_serve_mlx.py convert` — get base model in MLX format
2. **Phase 1 step 2**: `scripts/run_serve_mlx.py fuse` — merge LoRA into base
3. **Phase 1 step 3**: `scripts/run_serve_mlx.py serve` + `scripts/smoke_test_serve.py` — verify server returns sensible output for a known product
4. **Phase 1 spot check**: Compare 5-10 outputs against the H100-verified outputs we already have in `outputs/train_h100_20260406/sample_predictions.jsonl` — quality should be roughly equivalent (no major regressions from MLX q4 quantization)
5. **Phase 2 step 1**: Add `serving` dep group, create `serving/` module skeleton
6. **Phase 2 step 2**: Implement `prompts.py` (reuse training formatter), `client.py`, `models.py`
7. **Phase 2 step 3**: Implement `app.py` with `/classify`, `/health`, `/` routes
8. **Phase 2 step 4**: Write `tests/test_serving.py` and verify all pass
9. **Phase 2 step 5**: End-to-end test: start MLX server, start FastAPI, hit `/classify` from a real Python client
10. **Documentation**: Update README with serving quick-start

## Out of scope (Phase 3+)

These are explicitly **not** part of this plan, but are worth listing so we
know what we're deferring:

- **vLLM on rented GPU**: For real production. Will get a separate `serving-prod-plan.md` when needed.
- **Authentication / API keys**: Local-only means no auth. Add when going off-laptop.
- **Rate limiting**: Same — single user, no need.
- **Multi-LoRA serving**: When v1.1 / Ministral / etc. exist and we want to A/B them from one server. vLLM supports this natively; MLX does not (cleanly).
- **Streaming responses**: Add when a UI needs it.
- **Batch endpoint**: Add when a real batch workload exists.
- **Tailscale / ngrok / public exposure**: Out of scope for Phase 1-2. If you want it remotely accessible, run Tailscale yourself; the server doesn't need to know.
- **Containerization (Docker)**: Premature for local dev. Add when deploying.
- **Observability (Prometheus, traces)**: Premature. `print()` and `latency_ms` in the response are enough for v1.
- **Monitoring / alerting**: N/A for local serving.
- **Caching of identical requests**: Defer until there's a measured win.
- **A/B comparison UI**: Defer.
- **Confidence scores**: The v2 model produces structured text, not logprobs through the wrapper. Logprobs would require either using the raw MLX inference path (not the OpenAI server) or computing them in the wrapper. Skip for v1.

## Verification checklist (end of Phase 2)

- [ ] MLX server runs locally and serves the fused model
- [ ] FastAPI wrapper translates clean JSON requests to LLM calls and back
- [ ] Smoke test passes: known electrical wire test → returns chapter 85, valid HTS code, parseable response
- [ ] All unit tests in `tests/test_serving.py` pass
- [ ] End-to-end latency < 11s
- [ ] Memory usage < 18GB during inference
- [ ] No crashes on bad model output (parse_ok=false path)
- [ ] No crashes on upstream unreachable (503 path)
- [ ] README "Quick start: serving" works for a fresh clone
- [ ] Can be called from the orchestrator architecture as a black-box tool

## Future work (referenced, not specified here)

- **Phase 3**: vLLM on a rented L4 (24GB) for production-grade serving
- **Phase 3.1**: Multi-adapter serving (v1 + v1.1 + Ministral side-by-side)
- **Phase 3.2**: Batch API endpoint for bulk classification jobs
- **Phase 3.3**: Streaming `/classify/stream` SSE endpoint
- **Phase 4**: Orchestrator integration — wire the FastAPI service into a frontier-model agent (Sonnet or GPT) as a tool for "what is the HTS classification of this product?"
