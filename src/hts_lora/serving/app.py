"""FastAPI app exposing the HTS classifier as a clean POST /classify endpoint.

The wrapper hides the Nemotron prompt format quirks (system prefix, <think>
prefix) and the structured-text output format. Callers see plain typed JSON.

Configuration via environment variables:
    HTS_MLX_URL    Upstream MLX OpenAI base URL (default http://localhost:8080/v1)
    HTS_MLX_MODEL  Upstream model name        (default nemotron-hts-fused)
    HTS_MODEL_NAME Friendly model name in the response payload
                                              (default hts-nemotron-8b-lora-v1)
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from hts_lora.inference.parse_output import parse_prediction
from hts_lora.serving.client import MLXClient
from hts_lora.serving.models import (
    ClassificationResponse,
    ClassifyRequest,
    HealthResponse,
    HierarchyLevel,
)
from hts_lora.serving.prompts import build_messages, render_prompt


def _get_client() -> MLXClient:
    """Build a fresh MLXClient from current env vars."""
    return MLXClient(
        base_url=os.getenv("HTS_MLX_URL", "http://localhost:8080/v1"),
        model=os.getenv("HTS_MLX_MODEL", "nemotron-hts-fused"),
    )


def _model_name() -> str:
    return os.getenv("HTS_MODEL_NAME", "hts-nemotron-8b-lora-v1")


def create_app(client: MLXClient | None = None) -> FastAPI:
    """Build the FastAPI app. Pass an MLXClient for testing; default reads env."""
    app = FastAPI(
        title="HTS Classification API",
        description=(
            "Classify products into the U.S. Harmonized Tariff Schedule using a "
            "LoRA-fine-tuned Llama-3.1-Nemotron-Nano-8B model served via MLX."
        ),
        version="0.1.0",
    )

    # Allow tests to inject a client; otherwise build per-request from env
    injected_client = client

    def get_client() -> MLXClient:
        return injected_client or _get_client()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _INDEX_HTML

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        c = get_client()
        reachable = c.is_reachable()
        resp = HealthResponse(
            status="ok" if reachable else "degraded",
            upstream=c.base_url,
            upstream_reachable=reachable,
            model=_model_name(),
        )
        if not reachable:
            return JSONResponse(  # type: ignore[return-value]
                status_code=503,
                content=resp.model_dump(),
            )
        return resp

    @app.post("/classify", response_model=ClassificationResponse)
    def classify(req: ClassifyRequest) -> ClassificationResponse:
        c = get_client()
        messages = build_messages(
            description=req.description,
            materials=req.materials,
            use=req.use,
            country_of_origin=req.country_of_origin,
        )
        prompt = render_prompt(messages)
        try:
            upstream = c.complete_prompt(prompt, max_tokens=req.max_tokens)
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Upstream MLX server unreachable: {e}",
            )

        parsed = parse_prediction(upstream.text)

        chapter = (
            HierarchyLevel(code=parsed.chapter_code, description=parsed.chapter_desc)
            if parsed.chapter_code
            else None
        )
        heading = (
            HierarchyLevel(code=parsed.heading_code, description=parsed.heading_desc)
            if parsed.heading_code
            else None
        )
        subheading = (
            HierarchyLevel(code=parsed.subheading_code, description=parsed.subheading_desc)
            if parsed.subheading_code
            else None
        )

        return ClassificationResponse(
            hts_code=parsed.hts_code,
            chapter=chapter,
            heading=heading,
            subheading=subheading,
            reasoning=parsed.reasoning,
            provides_for=parsed.provides_for,
            is_abstention=parsed.is_abstention,
            abstention_reason=parsed.abstention_reason,
            parse_ok=parsed.parse_ok,
            raw=None if parsed.parse_ok else upstream.text,
            model=_model_name(),
            latency_ms=upstream.latency_ms,
        )

    return app


_INDEX_HTML = """\
<!doctype html>
<html>
<head><title>HTS Classification API</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; max-width: 720px; margin: 2em auto; padding: 0 1em; }
  pre { background: #f4f4f4; padding: 1em; border-radius: 6px; overflow-x: auto; }
  h1 { border-bottom: 1px solid #ddd; padding-bottom: 0.3em; }
  code { background: #f4f4f4; padding: 0.1em 0.3em; border-radius: 3px; }
</style>
</head>
<body>
  <h1>HTS Classification API</h1>
  <p>Classify products into the U.S. Harmonized Tariff Schedule using a
  LoRA-fine-tuned Llama-3.1-Nemotron-Nano-8B model served locally via MLX.</p>

  <h2>Endpoints</h2>
  <ul>
    <li><code>POST /classify</code> &mdash; classify a product</li>
    <li><code>GET /health</code> &mdash; check upstream MLX server health</li>
    <li><code>GET /docs</code> &mdash; OpenAPI / Swagger UI</li>
  </ul>

  <h2>Example</h2>
  <pre>curl -X POST http://localhost:8000/classify \\
  -H "Content-Type: application/json" \\
  -d '{
    "description": "insulated copper electrical wire, 12 AWG",
    "materials": "copper, PVC",
    "use": "residential wiring",
    "country_of_origin": "Mexico"
  }'</pre>
</body>
</html>
"""


# Default app for `uvicorn hts_lora.serving.app:app`
app = create_app()
