# hts-classifier:cuda
#
# Linux + NVIDIA GPU inference image. Bundles the FastAPI wrapper and the
# HF transformers + bitsandbytes nf4 inference path used at training time.
# Same code path that produced the 41.0% exact-match number on the ATLAS
# public test set in the v1 paper.
#
# Not for Mac. MLX is Apple Silicon only and does not run in Linux containers.
# See the /install page on canonical.agency for the Mac mini native path.
#
# Build:
#   docker build -t hts-classifier:cuda -f docker/cuda/Dockerfile .
#
# Run:
#   docker run --gpus all -p 8000:8000 \
#       -e HF_TOKEN=hf_xxx   # only needed if the adapter repo is private
#       -v $(pwd)/cache:/root/.cache/huggingface  # persist model cache
#       hts-classifier:cuda
#
# First start downloads the adapter (~336 MB) and base model (~16 GB).
# Subsequent starts reuse the cache volume.

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv via the published binary; it's faster than pip for our deps
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app

# Copy lockfile + project metadata first for layer-cache friendliness
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-dev

# Then the application source
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs

# The adapter is pulled at runtime, not baked into the image. This keeps the
# image free of the (publicly-redistributed) LoRA weights and lets the entry-
# point cache them to a mounted volume.
COPY docker/cuda/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000

# Healthcheck pings the wrapper's GET / (returns the index HTML)
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl --fail --silent http://localhost:8000/health || exit 1

ENV HTS_MODEL_NAME=hts-nemotron-8b-lora-v1 \
    HTS_ADAPTER_REPO=mfbaig35r/hts-nemotron-8b-lora-v1 \
    HTS_ADAPTER_DIR=/app/outputs/train_h100_20260406/adapter \
    HTS_API_HOST=0.0.0.0 \
    HTS_API_PORT=8000

ENTRYPOINT ["/entrypoint.sh"]
