"""Training CLI: configure and launch LoRA fine-tuning."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

from hts_lora.training.train_lora import train
from hts_lora.utils.config import load_train_config
from hts_lora.utils.logging import setup_logging

app = typer.Typer(help="HTS LoRA training pipeline")
console = Console()


@app.command()
def main(
    config: str = typer.Option("configs/train.yaml", help="Path to train config YAML"),
    data_dir: Optional[str] = typer.Option(None, help="Override data directory"),
    output_dir: Optional[str] = typer.Option(None, help="Override output directory"),
    resume: Optional[str] = typer.Option(None, help="Resume from checkpoint path"),
) -> None:
    """Launch LoRA fine-tuning."""
    setup_logging()

    cfg = load_train_config(config)
    if data_dir:
        cfg.data_dir = data_dir
    if output_dir:
        cfg.output_dir = output_dir

    console.print("[bold]HTS LoRA Training[/bold]")
    console.print(f"  Model: {cfg.model.model_id}")
    console.print(f"  LoRA rank: {cfg.lora.r}, alpha: {cfg.lora.lora_alpha}")
    console.print(f"  Epochs: {cfg.training.num_train_epochs}")
    console.print(f"  Batch size: {cfg.training.per_device_train_batch_size} x {cfg.training.gradient_accumulation_steps}")
    console.print(f"  Data: {cfg.data_dir}")
    if resume:
        console.print(f"  Resume from: {resume}")
    console.print()

    adapter_path = train(cfg, resume_from=resume)
    console.print(f"\n[bold green]Training complete![/bold green] Adapter saved to: {adapter_path}")


if __name__ == "__main__":
    app()
