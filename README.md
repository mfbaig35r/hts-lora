# HTS LoRA

LoRA fine-tuning pipeline for HTS (Harmonized Tariff Schedule) classification using Llama-3.1-Nemotron-Nano-8B-v1.

## Setup

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install
cd hts-lora
uv sync

# Copy and fill in environment variables
cp .env.example .env
```

Requires a CUDA GPU with at least 8GB VRAM (4-bit quantization).

## Architecture

The pipeline has three task modes (priority order):

1. **Rerank** (60%): Given product description + candidate codes, select best + rank all
2. **RAG Classify** (25%): Given product description + regulatory context, classify
3. **Direct Classify** (15%): Given product description only, classify

All modes output structured JSON with predicted code, confidence, and rationale.

### Key Design Decisions

- **Reasoning OFF**: System prompt starts with `"detailed thinking off"`, assistant prefixed with `<think>\n</think>\n` to suppress chain-of-thought and get direct JSON
- **Completion-only loss**: Training loss masked on system+user tokens; only assistant JSON is learned
- **4-bit NF4 quantization**: Base model quantized via bitsandbytes; LoRA adapters in bf16
- **LoRA on all linear layers**: q/k/v/o/gate/up/down projections (r=32, alpha=64)

## Project Structure

```
hts-lora/
├── configs/
│   ├── data.yaml          # Data pipeline config
│   ├── train.yaml         # Training config (model, LoRA, hyperparams)
│   └── eval.yaml          # Evaluation config
├── src/hts_lora/
│   ├── utils/             # Config, I/O, logging, HTS code utilities
│   ├── data/              # Ingest, normalize, build examples, split, format, audit
│   ├── training/          # Model factory, collator, callbacks, train loop
│   ├── inference/         # Single + batch prediction
│   └── evaluation/        # Metrics, error analysis, reports
├── scripts/
│   ├── run_data_prep.py   # Data preparation CLI
│   ├── run_train.py       # Training CLI
│   └── run_eval.py        # Evaluation CLI
├── tests/                 # Unit tests
└── data/samples/          # Example formatted data
```

## Quickstart

### 1. Prepare Data

Place raw data files in `data/raw/`, then:

```bash
uv run python scripts/run_data_prep.py --config configs/data.yaml
```

Steps: ingest → normalize → build → split → format → audit. Run individual steps with `--steps ingest,normalize`.

### 2. Train

```bash
uv run python scripts/run_train.py --config configs/train.yaml
```

Creates a timestamped run in `outputs/` with adapter weights, metrics, and sample predictions.

### 3. Evaluate

```bash
uv run python scripts/run_eval.py --config configs/eval.yaml
```

Generates `report.json`, `report.md`, `failures.jsonl`, and `per_chapter.json`.

### 4. Predict

```bash
uv run python -m hts_lora predict "Fresh cut roses from Colombia" \
  --adapter outputs/latest/adapter \
  --mode direct_classify
```

## CLI Reference

### Data Prep
```
run_data_prep.py [OPTIONS]
  --config       Path to data config YAML (default: configs/data.yaml)
  --steps        Comma-separated steps: ingest,normalize,build,split,format,audit
  --output-dir   Override output directory
```

### Training
```
run_train.py [OPTIONS]
  --config       Path to train config YAML (default: configs/train.yaml)
  --data-dir     Override formatted data directory
  --output-dir   Override output directory
```

### Evaluation
```
run_eval.py [OPTIONS]
  --config        Path to eval config YAML (default: configs/eval.yaml)
  --train-config  Path to train config for model loading
  --output-dir    Override output directory
```

### Prediction
```
python -m hts_lora predict DESCRIPTION [OPTIONS]
  --adapter       Path to LoRA adapter directory (required)
  --train-config  Path to train config (default: configs/train.yaml)
  --mode          Task mode: direct_classify, rag_classify, rerank
  --candidates    Comma-separated candidate codes (for rerank)
  --context       Regulatory context text (for rag_classify)
  --max-tokens    Max tokens to generate (default: 512)
```

## Running Tests

```bash
uv run pytest tests/ -v
```

## TODO

- [ ] Real CROSS rulings data loader
- [ ] Context retrieval pipeline for RAG mode
- [ ] Token log-probability confidence scoring
- [ ] WandB integration
- [ ] Multi-GPU training support
- [ ] GGUF export for llama.cpp inference
