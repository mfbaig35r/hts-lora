# Docker images

Two images, two audiences.

| Image | Target | What it does | Size |
|---|---|---|---|
| `hts-classifier:cuda` | Linux + NVIDIA GPU | Bundled FastAPI app loading HF transformers + bitsandbytes nf4. Same code path that produced the 41.0% ATLAS exact-match number in the paper. | ~4.5 GB |
| `hts-classifier:wrapper` | Any | FastAPI wrapper only. Talks upstream to a user-provided OpenAI-compatible inference server (typically `mlx_lm.server` running natively on a Mac). | ~340 MB |

Mac mini native deployment does **not** use Docker. MLX needs Apple Silicon Metal access, which Docker containers on macOS don't have. See [canonical.agency/install](https://canonical.agency/install) for the native walkthrough.

---

## Build

```bash
# From the repo root
docker build -t hts-classifier:cuda    -f docker/cuda/Dockerfile    .
docker build -t hts-classifier:wrapper -f docker/wrapper/Dockerfile .
```

On Apple Silicon, the CUDA image needs `--platform linux/amd64`. It builds but cannot run locally without an NVIDIA GPU.

## Run: `hts-classifier:cuda`

```bash
docker run --rm \
    --gpus all \
    -p 8000:8000 \
    -v $(pwd)/hf-cache:/root/.cache/huggingface \
    hts-classifier:cuda
```

First start downloads:
1. The v1 adapter (~336 MB) from `mfbaig35r/hts-nemotron-8b-lora-v1`
2. The base model `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` (~16 GB)

Cache to a host volume to avoid re-downloading on container restart. Hit `localhost:8000/classify` once the `/health` endpoint reports ready.

### Environment variables

| Var | Default | What it controls |
|---|---|---|
| `HTS_API_HOST` | `0.0.0.0` | FastAPI bind host |
| `HTS_API_PORT` | `8000` | FastAPI bind port |
| `HTS_BASE_MODEL` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | Base model HF repo |
| `HTS_ADAPTER_REPO` | `mfbaig35r/hts-nemotron-8b-lora-v1` | LoRA adapter HF repo |
| `HTS_ADAPTER_DIR` | `/app/outputs/train_h100_20260406/adapter` | Where the adapter is staged inside the container |
| `HTS_MODEL_NAME` | `hts-nemotron-8b-lora-v1` | Friendly name in the response payload |
| `HF_TOKEN` | unset | Only needed if you point at a private adapter repo |

## Run: `hts-classifier:wrapper`

Lightweight image (~340 MB). Does no inference itself; needs an OpenAI-compatible upstream.

The typical pattern: run `mlx_lm.server` natively on a Mac mini (see the [install page](https://canonical.agency/install)), then run the wrapper in Docker pointing at it.

```bash
docker run --rm \
    -p 8000:8000 \
    -e HTS_MLX_URL=http://host.docker.internal:8080/v1 \
    -e HTS_MLX_MODEL=/path/to/models/nemotron-hts-fused \
    hts-classifier:wrapper
```

`host.docker.internal` resolves to the host on Docker Desktop. On Linux use the host's IP or `--network=host`.

### Environment variables

| Var | Default | What it controls |
|---|---|---|
| `HTS_MLX_URL` | `http://host.docker.internal:8080/v1` | Upstream OpenAI-compat base URL |
| `HTS_MLX_MODEL` | unset (required) | Upstream model identifier (absolute path or HF repo) |
| `HTS_MODEL_NAME` | `hts-nemotron-8b-lora-v1` | Friendly name in the response payload |

## Push to Docker Hub

Owner of the namespace pushes; tags follow `<namespace>/hts-classifier:<variant>-v<n>`.

```bash
# Log in once (skips if already authenticated)
docker login

# Tag with the owner's namespace + version
docker tag hts-classifier:cuda    mfbaig35r/hts-classifier:cuda-v1
docker tag hts-classifier:wrapper mfbaig35r/hts-classifier:wrapper-v1

# Also push moving 'latest' tags per variant
docker tag hts-classifier:cuda    mfbaig35r/hts-classifier:cuda-latest
docker tag hts-classifier:wrapper mfbaig35r/hts-classifier:wrapper-latest

docker push mfbaig35r/hts-classifier:cuda-v1
docker push mfbaig35r/hts-classifier:wrapper-v1
docker push mfbaig35r/hts-classifier:cuda-latest
docker push mfbaig35r/hts-classifier:wrapper-latest
```

After publishing, users can pull without building:

```bash
docker pull mfbaig35r/hts-classifier:cuda-latest
docker pull mfbaig35r/hts-classifier:wrapper-latest
```

## Smoke-test against either image

Once the container is running and the `/health` endpoint reports OK:

```bash
curl -X POST localhost:8000/classify \
    -H 'Content-Type: application/json' \
    -d '{
      "description": "wool sweater, knitted, mens, size large",
      "country_of_origin": "Peru"
    }'
```

Expected response shape (`hts_code` will vary):

```json
{
  "hts_code": "6110.11.0030",
  "chapter": {"code": "61", "description": "ARTICLES OF APPAREL..."},
  "heading": {"code": "61.10", "description": "Sweaters, pullovers..."},
  "subheading": {"code": "6110.11", "description": "Of wool"},
  "reasoning": "...",
  "is_abstention": false,
  "parse_ok": true,
  "model": "hts-nemotron-8b-lora-v1",
  "latency_ms": 4500
}
```

## What's intentionally not shipped here

- **MLX inference image.** MLX is Apple Silicon only; no Linux container can use it.
- **CPU-only inference image.** Would work anywhere but slow enough that it is not a useful deployment target for the v1 model.
- **GPU-accelerated wrapper.** The wrapper has no model in-process; there is nothing to accelerate.

If a v2 model ships, the same two image variants apply, retagged accordingly.
