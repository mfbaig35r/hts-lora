"""CLI entry point: python -m hts_lora predict "description" --adapter path."""

from __future__ import annotations

import json
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(help="HTS LoRA CLI")
console = Console()


@app.command()
def predict(
    description: str = typer.Argument(..., help="Product description to classify"),
    adapter: str = typer.Option(..., help="Path to LoRA adapter directory"),
    train_config: str = typer.Option("configs/train.yaml", help="Path to train config"),
    mode: str = typer.Option("direct_classify", help="Task mode: direct_classify, rag_classify, rerank"),
    candidates: Optional[str] = typer.Option(None, help="Comma-separated candidate codes (for rerank mode)"),
    context: Optional[str] = typer.Option(None, help="Regulatory context text (for rag_classify mode)"),
    max_tokens: int = typer.Option(512, help="Max new tokens to generate"),
) -> None:
    """Classify a product description using the fine-tuned model."""
    from hts_lora.utils.config import load_train_config
    from hts_lora.utils.logging import setup_logging

    setup_logging()

    console.print(f"[bold]Loading model from {adapter}...[/bold]")
    config = load_train_config(train_config)

    from hts_lora.training.model_factory import load_for_inference
    model, tokenizer = load_for_inference(config, adapter)

    cand_list = [c.strip() for c in candidates.split(",")] if candidates else None

    from hts_lora.inference.predict import predict as run_predict
    result = run_predict(
        model=model,
        tokenizer=tokenizer,
        description=description,
        mode=mode,  # type: ignore[arg-type]
        candidates=cand_list,
        context=context,
        max_new_tokens=max_tokens,
    )

    if result["parse_ok"]:
        console.print("\n[bold green]Prediction:[/bold green]")
        console.print(json.dumps(result["parsed"], indent=2))
    else:
        console.print("\n[bold red]JSON parse failed. Raw output:[/bold red]")
        console.print(result["raw"])


if __name__ == "__main__":
    app()
