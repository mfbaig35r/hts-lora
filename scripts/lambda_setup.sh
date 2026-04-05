#!/bin/bash
# Lambda Cloud A100 setup script for HTS LoRA training
# Run this ONCE after SSH-ing into the instance
set -euo pipefail

echo "=== HTS LoRA GPU Training Setup ==="

# 1. Install uv
echo "[1/5] Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 2. Clone repo
echo "[2/5] Cloning repo..."
if [ ! -d "hts-lora" ]; then
    gh repo clone mfbaig35r/hts-lora
fi
cd hts-lora

# 3. Install dependencies (CUDA, no MLX)
echo "[3/5] Installing dependencies..."
uv sync

# 4. Set up HuggingFace token (needed for Nemotron model download)
echo "[4/5] Setting up HuggingFace..."
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: Set HF_TOKEN first: export HF_TOKEN=hf_xxx"
    exit 1
fi
echo "HF_TOKEN=${HF_TOKEN}" > .env
echo "CUDA_VISIBLE_DEVICES=0" >> .env

# 5. Verify data exists
echo "[5/5] Checking data..."
if [ ! -f "data/formatted/train.jsonl" ]; then
    echo "ERROR: Training data not found at data/formatted/"
    echo "Upload it with: scp -r data/formatted/ lambda:~/hts-lora/data/formatted/"
    exit 1
fi

echo ""
echo "=== Setup complete! ==="
echo "Training data: $(wc -l < data/formatted/train.jsonl) train examples"
echo ""
echo "To start training:"
echo "  uv run python scripts/run_train.py"
echo ""
echo "Training will take ~2 hours on A100 40GB."
echo "Adapter will be saved to outputs/train_*/adapter/"
