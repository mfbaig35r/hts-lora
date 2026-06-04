"""Evaluation CLI: run inference on test set and generate reports."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from hts_lora.utils.config import load_eval_config, load_train_config
from hts_lora.utils.logging import setup_logging

app = typer.Typer(help="HTS LoRA evaluation pipeline")
console = Console()


@app.command()
def main(
    config: str = typer.Option("configs/eval.yaml", help="Path to eval config YAML"),
    train_config: str = typer.Option("configs/train.yaml", help="Path to train config (for model loading)"),
    output_dir: Optional[str] = typer.Option(None, help="Override output directory"),
) -> None:
    """Run evaluation: inference on test set + generate reports."""
    setup_logging()

    eval_cfg = load_eval_config(config)
    train_cfg = load_train_config(train_config)
    if output_dir:
        eval_cfg.output_dir = output_dir

    out = Path(eval_cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    mode = "base-only (no adapter)" if not eval_cfg.adapter_path else f"adapter: {eval_cfg.adapter_path}"
    console.print("[bold]HTS LoRA Evaluation[/bold]")
    console.print(f"  Mode:      {mode}")
    console.print(f"  Test data: {eval_cfg.test_data}")
    console.print(f"  Output:    {eval_cfg.output_dir}")
    console.print()

    # Load model
    console.print("Loading model + adapter...")
    from hts_lora.training.model_factory import load_for_inference
    model, tokenizer = load_for_inference(train_cfg, eval_cfg.adapter_path)

    # Run inference
    console.print("Running inference on test set...")
    from hts_lora.inference.batch_predict import batch_predict
    predictions_path = out / "predictions.jsonl"
    batch_predict(
        model=model,
        tokenizer=tokenizer,
        input_path=eval_cfg.test_data,
        output_path=predictions_path,
        max_new_tokens=eval_cfg.generation.max_new_tokens,
    )

    # Generate report
    console.print("Generating report...")
    from hts_lora.utils.io import read_jsonl
    predictions = read_jsonl(predictions_path)

    from hts_lora.evaluation.reports import generate_report
    report = generate_report(predictions, out)

    # Print summary
    metrics = report["metrics"]
    console.print("\n[bold]Results:[/bold]")
    console.print(f"  Exact match:    {metrics.get('exact_match', 0):.3f}")
    console.print(f"  Chapter match:  {metrics.get('chapter_match', 0):.3f}")
    console.print(f"  Heading match:  {metrics.get('heading_match', 0):.3f}")
    console.print(f"  Parse rate:     {metrics.get('parse_rate', 0):.1%}")
    console.print(f"\n[bold green]Report saved to {out}[/bold green]")


if __name__ == "__main__":
    app()
