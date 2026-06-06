#!/usr/bin/env bash
# Entrypoint for the hts-classifier:cuda image.
# Ensures the v1 adapter is downloaded to HTS_ADAPTER_DIR, then starts the
# CUDA FastAPI app on HTS_API_HOST:HTS_API_PORT.

set -euo pipefail

ADAPTER_DIR="${HTS_ADAPTER_DIR:-/app/outputs/train_h100_20260406/adapter}"
ADAPTER_REPO="${HTS_ADAPTER_REPO:-mfbaig35r/hts-nemotron-8b-lora-v1}"
HOST="${HTS_API_HOST:-0.0.0.0}"
PORT="${HTS_API_PORT:-8000}"

echo "[hts-classifier:cuda] Starting on ${HOST}:${PORT}"
echo "[hts-classifier:cuda] Adapter dir : ${ADAPTER_DIR}"
echo "[hts-classifier:cuda] Adapter repo: ${ADAPTER_REPO}"

if [ ! -f "${ADAPTER_DIR}/adapter_model.safetensors" ]; then
    echo "[hts-classifier:cuda] Adapter not found locally, downloading from HF..."
    mkdir -p "${ADAPTER_DIR}"
    uv run hf download "${ADAPTER_REPO}" --local-dir "${ADAPTER_DIR}"
    echo "[hts-classifier:cuda] Adapter cached."
else
    echo "[hts-classifier:cuda] Adapter already present, skipping download."
fi

# Pre-flight: CUDA visible?
uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available in container'; print(f'[hts-classifier:cuda] CUDA visible: {torch.cuda.get_device_name(0)}')"

echo "[hts-classifier:cuda] Launching FastAPI..."
exec uv run uvicorn hts_lora.serving.cuda_app:app \
    --host "${HOST}" \
    --port "${PORT}" \
    --log-level info
