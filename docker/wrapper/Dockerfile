# hts-classifier:wrapper
#
# Lightweight image (~150 MB) that runs ONLY the FastAPI wrapper. Talks
# upstream to a user-provided MLX server (typically running natively on
# an Apple Silicon host).
#
# Useful when:
#   - You're running mlx_lm.server natively on a Mac mini for the GPU access,
#     and you want the API layer in a container for deploy / restart hygiene.
#   - You're routing through nginx / a load balancer that prefers a normal
#     container as the backend.
#   - You want to A/B test FastAPI versions without restarting the heavyweight
#     MLX server.
#
# Build:
#   docker build -t hts-classifier:wrapper -f docker/wrapper/Dockerfile .
#
# Run, pointing at the host's MLX server on port 8080:
#   docker run -p 8000:8000 \
#       -e HTS_MLX_URL=http://host.docker.internal:8080/v1 \
#       -e HTS_MLX_MODEL=/path/to/models/nemotron-hts-fused \
#       hts-classifier:wrapper
#
# (host.docker.internal works on Docker Desktop for Mac. On Linux use the
#  host's actual IP or --network=host.)

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install only the runtime deps needed by the FastAPI wrapper. Skip the
# inference stack (torch, transformers, peft, etc) entirely.
RUN pip install --no-cache-dir \
    "fastapi>=0.110" \
    "uvicorn[standard]>=0.27" \
    "openai>=1.0" \
    "pydantic>=2.0" \
    "typer>=0.9"

# Copy just the serving + inference parsing code (no training, no eval).
# inference/predict.py has TYPE_CHECKING-guarded imports of torch and
# transformers so it loads in this slim image without those deps. The
# top-level inference function is not used here (we proxy to the upstream
# MLX server), only build_v2_messages.
COPY src/hts_lora/__init__.py ./src/hts_lora/__init__.py
COPY src/hts_lora/serving ./src/hts_lora/serving
COPY src/hts_lora/inference/__init__.py ./src/hts_lora/inference/__init__.py
COPY src/hts_lora/inference/parse_output.py ./src/hts_lora/inference/parse_output.py
COPY src/hts_lora/inference/predict.py ./src/hts_lora/inference/predict.py
COPY src/hts_lora/utils ./src/hts_lora/utils
COPY src/hts_lora/data/__init__.py ./src/hts_lora/data/__init__.py
COPY src/hts_lora/data/formatters.py ./src/hts_lora/data/formatters.py

ENV PYTHONPATH=/app/src \
    HTS_MLX_URL=http://host.docker.internal:8080/v1 \
    HTS_MODEL_NAME=hts-nemotron-8b-lora-v1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail --silent http://localhost:8000/health || exit 1

CMD ["uvicorn", "hts_lora.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
