#!/usr/bin/env bash
# Run the v1 HTS LoRA evaluation suite on a rented CUDA box (tested on RunPod H100 SXM 80GB).
#
# 4 eval passes:
#   #1  v1 LoRA adapter on our test set (data/formatted/test.jsonl, ~14,952)
#   #2  Base Nemotron-Nano-8B, no adapter, on our test set
#   #3  v1 LoRA adapter on ATLAS test set (data/external/atlas_test_v2.jsonl, 200)
#   #4  Base Nemotron-Nano-8B, no adapter, on ATLAS test set
#
# Prerequisites on the pod:
#   - cd into the hts-lora repo (cloned/synced beforehand)
#   - data/formatted/test.jsonl present (gitignored; scp from local before
#     running, e.g. `scp -P $POD_SSH_PORT data/formatted/test.jsonl
#     root@$POD_HOST:hts-lora/data/formatted/test.jsonl`)
#   - export HF_TOKEN=hf_...   (private repo download; create a fresh one,
#                              the prior `hts-publish` token was paste-exposed)
#   - uv installed (curl -LsSf https://astral.sh/uv/install.sh | sh)
#   - GPU visible (nvidia-smi works)
#
# The script pauses for confirmation between phases so a bug never burns
# more than one phase of meter time. Full run takes 2-4 hours on H100,
# roughly $5-15.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"
WORKSPACE_CACHE="${WORKSPACE_CACHE:-/workspace/.cache/huggingface}"
ADAPTER_REPO="${ADAPTER_REPO:-mfbaig35r/hts-nemotron-8b-lora-v1}"
ADAPTER_DIR="${ADAPTER_DIR:-${REPO_DIR}/outputs/train_h100_20260406/adapter}"
ARTIFACTS_TAR="${ARTIFACTS_TAR:-/workspace/hts_v1_eval_artifacts.tar.gz}"

# Pinned versions known good with torch 2.4 + bnb (per docs/log-2026-04-06-h100-training.md).
PIN_TRANSFORMERS="4.46.3"
PIN_PEFT="0.13.2"
PIN_ACCELERATE="0.34.2"

YELLOW="\033[33m"
GREEN="\033[32m"
RED="\033[31m"
NC="\033[0m"

confirm() {
    echo
    echo -e "${YELLOW}>>> $1${NC}"
    read -r -p "    Continue? [y/N] " ans
    case "$ans" in
        y|Y|yes|YES) ;;
        *) echo "Aborted by user."; exit 1 ;;
    esac
}

heading() {
    echo
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  $1${NC}"
    echo -e "${GREEN}========================================${NC}"
}

# -----------------------------------------------------------------------------
# Phase 0: pod sanity
# -----------------------------------------------------------------------------
heading "Phase 0  Pod sanity"
echo "REPO_DIR    : ${REPO_DIR}"
echo "ADAPTER_DIR : ${ADAPTER_DIR}"
echo
nvidia-smi || { echo -e "${RED}nvidia-smi failed - no CUDA?${NC}"; exit 1; }
echo
df -h / /workspace 2>/dev/null || df -h /
echo
if [ -z "${HF_TOKEN:-}" ]; then
    echo -e "${RED}HF_TOKEN env var not set. Set it before re-running (private adapter repo).${NC}"
    exit 1
fi
echo "HF_TOKEN    : set (length ${#HF_TOKEN})"
if [ ! -f "${REPO_DIR}/data/formatted/test.jsonl" ]; then
    echo -e "${RED}data/formatted/test.jsonl missing (it is gitignored). scp it from local before continuing.${NC}"
    exit 1
fi
echo "test set    : $(wc -l < "${REPO_DIR}/data/formatted/test.jsonl") rows"

# -----------------------------------------------------------------------------
# Phase 1: bootstrap
# -----------------------------------------------------------------------------
confirm "Phase 1  Bootstrap: move HF cache, sync deps, install version pins"

mkdir -p /workspace/.cache
if [ -d "/root/.cache/huggingface" ] && [ ! -L "/root/.cache/huggingface" ]; then
    mv /root/.cache/huggingface "${WORKSPACE_CACHE}" 2>/dev/null || true
fi
mkdir -p "${WORKSPACE_CACHE}"
[ -L "/root/.cache/huggingface" ] || ln -sf "${WORKSPACE_CACHE}" /root/.cache/huggingface

export HF_HOME="${WORKSPACE_CACHE}"
export TRANSFORMERS_CACHE="${WORKSPACE_CACHE}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1

cd "${REPO_DIR}"
uv sync
# Override with the known-good pins (avoid the transformers 5.x / torch 2.4 set_submodule bug)
uv pip install \
    "transformers==${PIN_TRANSFORMERS}" \
    "peft==${PIN_PEFT}" \
    "accelerate==${PIN_ACCELERATE}"

echo
echo "Pinned: transformers==${PIN_TRANSFORMERS} peft==${PIN_PEFT} accelerate==${PIN_ACCELERATE}"

# -----------------------------------------------------------------------------
# Phase 2: pull adapter + convert ATLAS
# -----------------------------------------------------------------------------
confirm "Phase 2  Pull v1 adapter from ${ADAPTER_REPO} and build ATLAS v2 file"

if [ ! -f "${ADAPTER_DIR}/adapter_model.safetensors" ]; then
    mkdir -p "${ADAPTER_DIR}"
    HF_TOKEN="${HF_TOKEN}" uv run hf download "${ADAPTER_REPO}" \
        --local-dir "${ADAPTER_DIR}"
else
    echo "Adapter already present at ${ADAPTER_DIR}, skipping download."
fi

uv run python scripts/build_atlas_eval.py
test -f data/external/atlas_test_v2.jsonl || \
    { echo -e "${RED}ATLAS conversion produced no output file${NC}"; exit 1; }

# -----------------------------------------------------------------------------
# Phase 3: eval #1 - v1 adapter on our test set
# -----------------------------------------------------------------------------
confirm "Phase 3  Eval #1: v1 adapter on data/formatted/test.jsonl (14,952 ex, ~30-60 min)"
uv run python scripts/run_eval.py --config configs/eval.yaml

# -----------------------------------------------------------------------------
# Phase 4: eval #2 - base Nemotron on our test set
# -----------------------------------------------------------------------------
confirm "Phase 4  Eval #2: base Nemotron on data/formatted/test.jsonl (14,952 ex, ~30-60 min)"
uv run python scripts/run_eval.py --config configs/eval_base.yaml

# -----------------------------------------------------------------------------
# Phase 5: eval #3 - v1 adapter on ATLAS
# -----------------------------------------------------------------------------
confirm "Phase 5  Eval #3: v1 adapter on ATLAS (200 ex, ~5 min)"
uv run python scripts/run_eval.py --config configs/eval_atlas_v1.yaml

# -----------------------------------------------------------------------------
# Phase 6: eval #4 - base Nemotron on ATLAS
# -----------------------------------------------------------------------------
confirm "Phase 6  Eval #4: base Nemotron on ATLAS (200 ex, ~5 min)"
uv run python scripts/run_eval.py --config configs/eval_atlas_base.yaml

# -----------------------------------------------------------------------------
# Phase 7: artifacts
# -----------------------------------------------------------------------------
heading "Phase 7  Bundle artifacts"
tar -czvf "${ARTIFACTS_TAR}" \
    outputs/eval_v1 \
    outputs/eval_base_v1 \
    outputs/eval_atlas_v1 \
    outputs/eval_atlas_base \
    2>&1 | tail -10

echo
echo -e "${GREEN}Done. Artifacts at: ${ARTIFACTS_TAR}${NC}"
echo
echo "To pull to local machine:"
echo "  scp -P <pod_ssh_port> root@<pod_host>:${ARTIFACTS_TAR} ."
echo
echo "Remember to stop the pod when done to avoid trailing charges."
