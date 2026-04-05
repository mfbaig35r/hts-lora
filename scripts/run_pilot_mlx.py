"""MLX pilot: convert model, subset data, train LoRA, generate samples.

Usage:
    uv run python scripts/run_pilot_mlx.py convert
    uv run python scripts/run_pilot_mlx.py subset
    uv run python scripts/run_pilot_mlx.py train
    uv run python scripts/run_pilot_mlx.py generate
    uv run python scripts/run_pilot_mlx.py all        # run all steps
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="MLX LoRA pilot for HTS classification")
console = Console()

# Defaults
DEFAULT_MODEL = "bourn23/nvidia-llama-3.1-nemotron-nano-8b-v1-mlx-4bit"
DEFAULT_CONFIG = "configs/train_mlx.yaml"
DEFAULT_DATA_DIR = Path("data/formatted")
DEFAULT_PILOT_DIR = Path("data/pilot")
DEFAULT_ADAPTER_DIR = Path("adapters")

TRAIN_SUBSET_SIZE = 5000
VALID_SUBSET_SIZE = 500


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


@app.command()
def convert(
    model_id: str = typer.Option(DEFAULT_MODEL, help="HuggingFace model ID"),
    mlx_path: str = typer.Option("models/nemotron-nano-8b-mlx", help="Local MLX model path"),
    quantize: bool = typer.Option(True, "--quantize/--no-quantize", help="Apply 4-bit quantization"),
) -> None:
    """Convert HuggingFace model to MLX format (or use pre-quantized community model)."""
    mlx_dir = Path(mlx_path)

    # If using a community pre-quantized model, mlx_lm handles it automatically
    if "/" in model_id and not model_id.startswith("nvidia/"):
        console.print(f"[green]Using pre-quantized community model: {model_id}[/green]")
        console.print("mlx-lm will download and cache automatically during training.")
        return

    if mlx_dir.exists() and (mlx_dir / "config.json").exists():
        console.print(f"[yellow]MLX model already exists at {mlx_dir}, skipping conversion[/yellow]")
        return

    console.print(f"[bold]Converting {model_id} to MLX format...[/bold]")
    cmd = [
        sys.executable, "-m", "mlx_lm", "convert",
        "--hf-path", model_id,
        "--mlx-path", str(mlx_dir),
    ]
    if quantize:
        cmd.append("-q")

    subprocess.run(cmd, check=True)
    console.print(f"[green]Model converted to {mlx_dir}[/green]")


@app.command()
def subset(
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, help="Source formatted data directory"),
    pilot_dir: Path = typer.Option(DEFAULT_PILOT_DIR, help="Output pilot data directory"),
    train_size: int = typer.Option(TRAIN_SUBSET_SIZE, help="Number of training examples"),
    valid_size: int = typer.Option(VALID_SUBSET_SIZE, help="Number of validation examples"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    """Create a small training subset preserving task type distribution."""
    rng = random.Random(seed)

    train_path = data_dir / "train.jsonl"
    valid_path = data_dir / "valid.jsonl"

    if not train_path.exists():
        console.print(f"[red]Train data not found: {train_path}[/red]")
        raise typer.Exit(1)

    console.print("[bold]Creating pilot subset...[/bold]")

    # Read full data
    train_data = _read_jsonl(train_path)
    valid_data = _read_jsonl(valid_path) if valid_path.exists() else []

    console.print(f"  Full train: {len(train_data)}, full valid: {len(valid_data)}")

    # Stratified subset by task_type (preserve ~90% classify / ~10% abstain ratio)
    train_subset = _stratified_sample(train_data, train_size, rng)
    valid_subset = _stratified_sample(valid_data, valid_size, rng) if valid_data else []

    # Write pilot data
    _write_jsonl(train_subset, pilot_dir / "train.jsonl")
    _write_jsonl(valid_subset, pilot_dir / "valid.jsonl")

    # Also copy test.jsonl if it exists (for later evaluation)
    test_path = data_dir / "test.jsonl"
    if test_path.exists():
        test_data = _read_jsonl(test_path)
        _write_jsonl(test_data[:500], pilot_dir / "test.jsonl")

    # Stats
    train_types = Counter(r.get("task_type", "unknown") for r in train_subset)
    console.print(f"  Pilot train: {len(train_subset)} (types: {dict(train_types)})")
    console.print(f"  Pilot valid: {len(valid_subset)}")
    console.print(f"[green]Pilot data written to {pilot_dir}[/green]")


def _stratified_sample(records: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Sample n records preserving task_type distribution."""
    if len(records) <= n:
        return records

    by_type: dict[str, list[dict]] = {}
    for r in records:
        task_type = r.get("task_type", "unknown")
        by_type.setdefault(task_type, []).append(r)

    # Calculate proportional sizes
    result = []
    remaining = n
    type_keys = sorted(by_type.keys())

    for i, key in enumerate(type_keys):
        group = by_type[key]
        if i == len(type_keys) - 1:
            count = remaining
        else:
            count = max(1, int(n * len(group) / len(records)))
            count = min(count, remaining, len(group))

        rng.shuffle(group)
        result.extend(group[:count])
        remaining -= count

    rng.shuffle(result)
    return result


@app.command()
def train(
    config: str = typer.Option(DEFAULT_CONFIG, help="MLX training config YAML"),
    data_dir: str = typer.Option(str(DEFAULT_PILOT_DIR), help="Data directory (use pilot for fast iteration)"),
) -> None:
    """Run mlx-lm LoRA training with the pilot config."""
    config_path = Path(config)
    if not config_path.exists():
        console.print(f"[red]Config not found: {config}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Starting MLX LoRA training...[/bold]")
    console.print(f"  Config: {config}")
    console.print(f"  Data: {data_dir}")

    # Ensure adapter directory exists
    DEFAULT_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--train",
        "--config", config,
        "--data", data_dir,
    ]

    subprocess.run(cmd, check=True)
    console.print("[green]Training complete![/green]")


@app.command()
def generate(
    model_id: str = typer.Option(DEFAULT_MODEL, help="Base model ID"),
    adapter_path: str = typer.Option(
        str(DEFAULT_ADAPTER_DIR), help="Path to adapter directory"
    ),
    data_dir: Path = typer.Option(DEFAULT_PILOT_DIR, help="Data directory for test examples"),
    num_examples: int = typer.Option(50, help="Number of examples to generate"),
    max_tokens: int = typer.Option(512, help="Max tokens to generate"),
) -> None:
    """Generate sample predictions and compare with expected output."""
    try:
        from mlx_lm import generate as mlx_generate
        from mlx_lm import load as mlx_load
    except ImportError:
        console.print("[red]mlx-lm not installed. Run: uv pip install 'hts-lora[mlx]'[/red]")
        raise typer.Exit(1)

    adapter_dir = Path(adapter_path)
    if not (adapter_dir / "adapter_config.json").exists():
        console.print(f"[red]No adapter_config.json in: {adapter_path}[/red]")
        raise typer.Exit(1)

    # Load validation examples
    valid_path = data_dir / "valid.jsonl"
    if not valid_path.exists():
        valid_path = data_dir / "test.jsonl"
    if not valid_path.exists():
        console.print(f"[red]No valid.jsonl or test.jsonl found in {data_dir}[/red]")
        raise typer.Exit(1)

    examples = _read_jsonl(valid_path)[:num_examples]
    console.print(f"[bold]Generating predictions for {len(examples)} examples...[/bold]")

    # Load model with adapter
    model, tokenizer = mlx_load(model_id, adapter_path=adapter_path)

    from hts_lora.inference.parse_output import parse_prediction

    parse_ok_count = 0
    results = []

    for i, ex in enumerate(examples):
        messages = ex["messages"][:2]  # system + user only

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt += "<think>\n</think>\n\n"

        generated = mlx_generate(
            model, tokenizer, prompt=prompt, max_tokens=max_tokens
        )

        parsed = parse_prediction(generated)
        if parsed.parse_ok:
            parse_ok_count += 1

        # Extract expected from assistant message
        expected = ex["messages"][2]["content"] if len(ex["messages"]) > 2 else ""
        expected_body = expected.split("</think>\n\n", 1)[-1] if "</think>" in expected else expected

        results.append({
            "index": i,
            "parse_ok": parsed.parse_ok,
            "predicted_code": parsed.hts_code,
            "expected_code": ex.get("hts_code", ""),
            "generated": generated[:300],
            "expected": expected_body[:300],
        })

        if (i + 1) % 10 == 0:
            console.print(f"  [{i + 1}/{len(examples)}] parse_ok: {parse_ok_count}/{i + 1}")

    # Summary
    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"  Parse rate: {parse_ok_count}/{len(examples)} ({parse_ok_count / len(examples):.1%})")

    # Show a few side-by-side comparisons
    table = Table(title="Sample Predictions (first 5)")
    table.add_column("#", width=3)
    table.add_column("Expected Code", width=15)
    table.add_column("Predicted Code", width=15)
    table.add_column("Parse OK", width=8)

    for r in results[:5]:
        table.add_row(
            str(r["index"]),
            r["expected_code"],
            r["predicted_code"] or "N/A",
            "Y" if r["parse_ok"] else "N",
        )

    console.print(table)

    # Write full results
    output_path = Path("outputs/mlx_pilot/pilot_predictions.jsonl")
    _write_jsonl(results, output_path)
    console.print(f"\n[green]Full results written to {output_path}[/green]")


@app.command()
def all(
    config: str = typer.Option(DEFAULT_CONFIG, help="MLX training config"),
    model_id: str = typer.Option(DEFAULT_MODEL, help="Model ID"),
    train_size: int = typer.Option(TRAIN_SUBSET_SIZE, help="Pilot training subset size"),
    valid_size: int = typer.Option(VALID_SUBSET_SIZE, help="Pilot validation subset size"),
) -> None:
    """Run all pilot steps: convert → subset → train → generate."""
    console.print("[bold]Running full MLX pilot pipeline[/bold]\n")

    console.print("[bold]Step 1/4: Convert[/bold]")
    convert(model_id=model_id)

    console.print("\n[bold]Step 2/4: Subset[/bold]")
    subset(train_size=train_size, valid_size=valid_size)

    console.print("\n[bold]Step 3/4: Train[/bold]")
    train(config=config, data_dir=str(DEFAULT_PILOT_DIR))

    console.print("\n[bold]Step 4/4: Generate[/bold]")
    generate(model_id=model_id)

    console.print("\n[bold green]Pilot complete![/bold green]")


if __name__ == "__main__":
    app()
