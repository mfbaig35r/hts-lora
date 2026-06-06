"""FastAPI app for CUDA inference.

Used by the hts-classifier:cuda Docker image. Loads the base Nemotron-Nano-8B
in 4-bit nf4 quantization via bitsandbytes, applies the v1 LoRA adapter via
PEFT, and exposes the same POST /classify interface as the Mac mini wrapper.

This is the same inference path that produced the 41.0% exact-match number
on the ATLAS public test set in the v1 paper. For Apple Silicon / MLX
inference (36.0% on the same test, 4.7s latency), see serving/app.py.

The model loads at startup, not per-request. First /classify after a fresh
boot pays no additional load cost.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from hts_lora.inference.parse_output import parse_prediction
from hts_lora.inference.predict import build_v2_messages
from hts_lora.serving.models import (
    ClassificationResponse,
    ClassifyRequest,
    HealthResponse,
    HierarchyLevel,
)

log = logging.getLogger("hts_lora.serving.cuda_app")

# Loaded once at startup, reused across requests.
_state: dict[str, Any] = {}


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Load the model + tokenizer once at startup."""
    base_model_id = os.getenv("HTS_BASE_MODEL", "nvidia/Llama-3.1-Nemotron-Nano-8B-v1")
    adapter_dir = os.getenv("HTS_ADAPTER_DIR", "outputs/train_h100_20260406/adapter")

    log.info("Loading base model %s in 4-bit nf4...", base_model_id)
    # Local imports so module import doesn't fail in test environments
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    log.info("Applying LoRA adapter from %s", adapter_dir)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    _state["model"] = model
    _state["tokenizer"] = tokenizer
    log.info("Model ready.")
    yield
    _state.clear()


app = FastAPI(
    title="HTS Classifier (CUDA)",
    description=(
        "Local HTS classification via Llama-3.1-Nemotron-Nano-8B + v1 LoRA, "
        "running on NVIDIA GPU with HF transformers + bnb-nf4 quantization."
    ),
    lifespan=_lifespan,
)

_INDEX_HTML = """<!doctype html>
<html><head><title>HTS Classifier (CUDA)</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; max-width: 720px; margin: 2em auto; padding: 0 1em; }
  pre { background: #f4f4f4; padding: 1em; border-radius: 6px; }
  code { background: #f4f4f4; padding: 0.1em 0.3em; border-radius: 3px; }
  h1 { border-bottom: 1px solid #ddd; padding-bottom: 0.3em; }
</style></head>
<body>
  <h1>HTS Classifier API (CUDA)</h1>
  <p>Linux + NVIDIA GPU edition. Loads the v1 LoRA adapter on Llama-3.1-Nemotron-Nano-8B
  via HF transformers + bitsandbytes nf4 4-bit quantization.</p>
  <h2>Endpoints</h2>
  <ul>
    <li><code>POST /classify</code> &mdash; classify a product</li>
    <li><code>GET /health</code> &mdash; check liveness</li>
    <li><code>GET /docs</code> &mdash; OpenAPI / Swagger UI</li>
  </ul>
  <h2>Example</h2>
  <pre>curl -X POST localhost:8000/classify -H 'Content-Type: application/json' \\
  -d '{"description":"wool sweater, knitted, mens, size large","country_of_origin":"Peru"}'</pre>
</body></html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(_INDEX_HTML)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if _state.get("model") is not None else "loading",
        upstream="local (in-process)",
        upstream_reachable=_state.get("model") is not None,
        model=os.getenv("HTS_MODEL_NAME", "hts-nemotron-8b-lora-v1"),
    )


def _level(code: str | None, description: str | None) -> HierarchyLevel | None:
    if not code:
        return None
    return HierarchyLevel(code=code, description=description or "")


@app.post("/classify", response_model=ClassificationResponse)
def classify(req: ClassifyRequest) -> ClassificationResponse:
    model = _state.get("model")
    tokenizer = _state.get("tokenizer")
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model still loading")

    # Build the v2 prompt
    variant = "rich" if (req.materials or req.use or req.country_of_origin) else "minimal"
    messages = build_v2_messages(
        description=req.description,
        variant=variant,  # type: ignore[arg-type]
        materials=req.materials,
        product_use=req.use,
        country=req.country_of_origin,
    )
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    text += "<think>\n</think>\n\n"

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    started = time.perf_counter()
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens or 512,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.05,
        )
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    generated_ids = out_ids[0][inputs["input_ids"].shape[1] :]
    raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    parsed = parse_prediction(raw_text)

    return ClassificationResponse(
        hts_code=parsed.hts_code,
        chapter=_level(parsed.chapter_code, parsed.chapter_desc),
        heading=_level(parsed.heading_code, parsed.heading_desc),
        subheading=_level(parsed.subheading_code, parsed.subheading_desc),
        reasoning=parsed.reasoning,
        provides_for=parsed.provides_for,
        is_abstention=parsed.is_abstention,
        abstention_reason=parsed.abstention_reason,
        parse_ok=parsed.parse_ok,
        raw=raw_text if not parsed.parse_ok else None,
        model=os.getenv("HTS_MODEL_NAME", "hts-nemotron-8b-lora-v1"),
        latency_ms=elapsed_ms,
    )


# Hint for any tooling that introspects via asdict(parsed)
__all__ = ["app", "asdict"]
