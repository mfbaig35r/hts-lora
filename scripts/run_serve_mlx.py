"""Serve the HTS LoRA adapter as a local OpenAI-compatible MLX server.

The trained adapter at outputs/train_h100_20260406/adapter/ is in HuggingFace
PEFT format, not MLX format, so the pipeline is:

  1. merge:    Apply PEFT adapter to the HF base model with merge_and_unload(),
               save the merged model in HF format. This is the only step that
               touches the (large) full-precision weights.
  2. convert:  Run mlx_lm.convert on the merged HF model to produce a 4-bit
               quantized MLX directory.
  3. serve:    Run mlx_lm.server pointed at the fused MLX model.

After convert succeeds you can delete the merged HF directory; only the MLX
output is needed to serve.

Usage:
    uv run python scripts/run_serve_mlx.py merge
    uv run python scripts/run_serve_mlx.py convert
    uv run python scripts/run_serve_mlx.py serve
    uv run python scripts/run_serve_mlx.py all      # merge -> convert -> serve
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(help="Serve the HTS LoRA adapter as a local MLX server")
console = Console()

# Defaults
DEFAULT_BASE_MODEL = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
DEFAULT_ADAPTER = "outputs/train_h100_20260406/adapter"
DEFAULT_HF_FALLBACK = "mfbaig35r/hts-nemotron-8b-lora-v1"
DEFAULT_MERGED_DIR = Path("models/nemotron-hts-merged-hf")
DEFAULT_FUSED_DIR = Path("models/nemotron-hts-fused")
DEFAULT_PORT = 8080
DEFAULT_HOST = "127.0.0.1"


@app.command()
def merge(
    base_model: str = typer.Option(DEFAULT_BASE_MODEL, help="HuggingFace base model ID"),
    adapter: str = typer.Option(
        DEFAULT_ADAPTER,
        help="Local PEFT adapter dir or HF repo (mfbaig35r/hts-nemotron-8b-lora-v1)",
    ),
    output_dir: Path = typer.Option(
        DEFAULT_MERGED_DIR, help="Where to write the merged HF model"
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite existing output_dir"),
) -> None:
    """Merge the PEFT LoRA adapter into the base model and save in HF format."""
    if output_dir.exists() and not force:
        console.print(
            f"[yellow]Merged model already exists at {output_dir}. "
            "Use --force to overwrite.[/yellow]"
        )
        return

    # Resolve adapter: prefer local dir, fall back to HF
    adapter_path = Path(adapter)
    if not adapter_path.exists():
        console.print(
            f"[yellow]Local adapter not found at {adapter_path}, "
            f"falling back to HF repo: {DEFAULT_HF_FALLBACK}[/yellow]"
        )
        adapter = DEFAULT_HF_FALLBACK

    console.print(f"[bold]Loading base model:[/bold] {base_model}")
    console.print(f"[bold]Loading adapter:[/bold] {adapter}")
    console.print("[dim]This will download the base model (~16 GB) if not cached.[/dim]")

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/red]")
        raise typer.Exit(1)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # CPU merge is slow but avoids 16GB MPS spike
        low_cpu_mem_usage=True,
    )

    console.print("[bold]Applying LoRA adapter...[/bold]")
    model = PeftModel.from_pretrained(base, adapter)

    console.print("[bold]Merging adapter into base weights...[/bold]")
    merged = model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold]Saving merged model to {output_dir}...[/bold]")
    merged.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    size_gb = sum(p.stat().st_size for p in output_dir.rglob("*") if p.is_file()) / 1e9
    console.print(f"[green]Merged model saved ({size_gb:.1f} GB)[/green]")


@app.command()
def convert(
    merged_dir: Path = typer.Option(DEFAULT_MERGED_DIR, help="Merged HF model dir"),
    output_dir: Path = typer.Option(DEFAULT_FUSED_DIR, help="Output MLX dir"),
    quantize: bool = typer.Option(True, "--quantize/--no-quantize", help="Apply q4 quantization"),
    cleanup: bool = typer.Option(
        False, "--cleanup", help="Delete the merged HF dir after successful conversion"
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite existing output_dir"),
) -> None:
    """Convert the merged HF model to MLX format (q4 by default)."""
    if not merged_dir.exists():
        console.print(
            f"[red]Merged HF model not found at {merged_dir}. "
            "Run `merge` first.[/red]"
        )
        raise typer.Exit(1)

    if output_dir.exists() and not force:
        console.print(
            f"[yellow]MLX model already exists at {output_dir}. "
            "Use --force to overwrite.[/yellow]"
        )
        return

    if output_dir.exists() and force:
        shutil.rmtree(output_dir)

    console.print(f"[bold]Converting {merged_dir} -> {output_dir} (q4={quantize})[/bold]")

    cmd = [
        sys.executable, "-m", "mlx_lm", "convert",
        "--hf-path", str(merged_dir),
        "--mlx-path", str(output_dir),
    ]
    if quantize:
        cmd.extend(["-q", "--q-bits", "4"])

    subprocess.run(cmd, check=True)

    size_gb = sum(p.stat().st_size for p in output_dir.rglob("*") if p.is_file()) / 1e9
    console.print(f"[green]MLX model saved to {output_dir} ({size_gb:.1f} GB)[/green]")

    if cleanup:
        console.print(f"[yellow]Removing intermediate merged HF model: {merged_dir}[/yellow]")
        shutil.rmtree(merged_dir)


@app.command()
def serve(
    model_dir: Path = typer.Option(DEFAULT_FUSED_DIR, help="Path to fused MLX model"),
    host: str = typer.Option(DEFAULT_HOST, help="Bind host"),
    port: int = typer.Option(DEFAULT_PORT, help="Bind port"),
) -> None:
    """Run mlx_lm.server pointed at the fused MLX model."""
    if not model_dir.exists():
        console.print(
            f"[red]Fused MLX model not found at {model_dir}. "
            "Run `merge` and `convert` first.[/red]"
        )
        raise typer.Exit(1)

    console.print(
        f"[bold green]Starting mlx_lm.server[/bold green] "
        f"model={model_dir} on {host}:{port}"
    )
    console.print(
        f"[dim]Test with: curl http://{host}:{port}/v1/models[/dim]\n"
    )

    cmd = [
        sys.executable, "-m", "mlx_lm", "server",
        "--model", str(model_dir),
        "--host", host,
        "--port", str(port),
    ]
    # Foreground process — Ctrl-C exits cleanly
    subprocess.run(cmd, check=True)


@app.command()
def all(
    base_model: str = typer.Option(DEFAULT_BASE_MODEL, help="HF base model"),
    adapter: str = typer.Option(DEFAULT_ADAPTER, help="PEFT adapter path"),
    cleanup: bool = typer.Option(
        True, "--cleanup/--keep", help="Delete merged HF dir after MLX conversion"
    ),
) -> None:
    """Run merge -> convert -> serve in one command."""
    console.print("[bold]Step 1/3: merge[/bold]")
    merge(base_model=base_model, adapter=adapter, output_dir=DEFAULT_MERGED_DIR, force=False)

    console.print("\n[bold]Step 2/3: convert[/bold]")
    convert(
        merged_dir=DEFAULT_MERGED_DIR,
        output_dir=DEFAULT_FUSED_DIR,
        quantize=True,
        cleanup=cleanup,
        force=False,
    )

    console.print("\n[bold]Step 3/3: serve[/bold]")
    serve(model_dir=DEFAULT_FUSED_DIR, host=DEFAULT_HOST, port=DEFAULT_PORT)


if __name__ == "__main__":
    app()
