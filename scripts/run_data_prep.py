"""Data preparation CLI: ingest → normalize → build → split → format → audit."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from hts_lora.data.audit import audit_examples, print_audit
from hts_lora.data.build_examples import build_examples
from hts_lora.data.formatters import format_dataset
from hts_lora.data.ingest import ingest
from hts_lora.data.normalize import normalize_examples
from hts_lora.data.split import stratified_split
from hts_lora.utils.config import load_data_config
from hts_lora.utils.io import write_json, write_jsonl
from hts_lora.utils.logging import setup_logging

app = typer.Typer(help="HTS LoRA data preparation pipeline")
console = Console()

ALL_STEPS = ["ingest", "normalize", "build", "split", "format", "audit"]


@app.command()
def main(
    config: str = typer.Option("configs/data.yaml", help="Path to data config YAML"),
    steps: str = typer.Option(
        ",".join(ALL_STEPS),
        help="Comma-separated list of steps to run",
    ),
    output_dir: Optional[str] = typer.Option(None, help="Override output directory"),
) -> None:
    """Run the data preparation pipeline."""
    setup_logging()
    cfg = load_data_config(config)
    if output_dir:
        cfg.output_dir = output_dir

    step_list = [s.strip() for s in steps.split(",")]
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fmt_out = Path(cfg.formatted_dir)
    fmt_out.mkdir(parents=True, exist_ok=True)

    examples = None
    training_examples = None
    split = None

    if "ingest" in step_list:
        console.print("[bold]Step 1/6: Ingest[/bold]")
        examples = ingest(cfg)
        write_jsonl(
            [e.model_dump() for e in examples],
            out / "raw_examples.jsonl",
        )

    if "normalize" in step_list:
        if examples is None:
            from hts_lora.data.ingest import RawExample
            from hts_lora.utils.io import read_jsonl
            examples = [RawExample(**r) for r in read_jsonl(out / "raw_examples.jsonl")]
        console.print("[bold]Step 2/6: Normalize[/bold]")
        examples = normalize_examples(examples, cfg.normalization)
        write_jsonl(
            [e.model_dump() for e in examples],
            out / "normalized_examples.jsonl",
        )

    if "build" in step_list:
        if examples is None:
            from hts_lora.data.ingest import RawExample
            from hts_lora.utils.io import read_jsonl
            examples = [RawExample(**r) for r in read_jsonl(out / "normalized_examples.jsonl")]
        console.print("[bold]Step 3/6: Build training examples[/bold]")
        training_examples = build_examples(examples, cfg)
        write_jsonl(
            [e.model_dump() for e in training_examples],
            out / "training_examples.jsonl",
        )

    if "split" in step_list:
        if training_examples is None:
            from hts_lora.data.build_examples import TrainingExample
            from hts_lora.utils.io import read_jsonl
            training_examples = [TrainingExample(**r) for r in read_jsonl(out / "training_examples.jsonl")]
        console.print("[bold]Step 4/6: Split[/bold]")
        split = stratified_split(training_examples, cfg.split)
        console.print(f"  Split: {split.summary()}")

    if "format" in step_list:
        if split is None:
            console.print("[red]Cannot format without split step[/red]")
            raise typer.Exit(1)
        console.print("[bold]Step 5/6: Format[/bold]")
        for name, data in [("train", split.train), ("valid", split.valid), ("test", split.test)]:
            formatted = format_dataset(data)
            write_jsonl(formatted, fmt_out / f"{name}.jsonl")
            console.print(f"  {name}: {len(formatted)} examples -> {fmt_out / f'{name}.jsonl'}")

    if "audit" in step_list:
        if training_examples is None and split is not None:
            training_examples = split.train + split.valid + split.test
        if training_examples is None:
            from hts_lora.data.build_examples import TrainingExample
            from hts_lora.utils.io import read_jsonl
            training_examples = [TrainingExample(**r) for r in read_jsonl(out / "training_examples.jsonl")]
        console.print("[bold]Step 6/6: Audit[/bold]")
        stats = audit_examples(training_examples)
        write_json(stats, out / "audit.json")
        print_audit(stats)

    console.print("\n[bold green]Data preparation complete![/bold green]")


if __name__ == "__main__":
    app()
