"""Walk through each pipeline stage with sample data to show transformations."""

import json
import random
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from hts_lora.data.ingest import RawExample
from hts_lora.data.normalize import normalize_examples
from hts_lora.data.build_examples import build_examples
from hts_lora.data.formatters import format_example
from hts_lora.utils.config import DataConfig, NormalizationConfig, SourceConfig
from hts_lora.utils.io import read_jsonl

console = Console()
HERE = Path(__file__).parent


def show_json(label: str, obj: dict, max_lines: int = 30) -> None:
    text = json.dumps(obj, indent=2, ensure_ascii=False)
    lines = text.split("\n")
    if len(lines) > max_lines:
        text = "\n".join(lines[:max_lines]) + "\n  ... (truncated)"
    console.print(Panel(Syntax(text, "json", theme="monokai"), title=label, border_style="cyan"))


def main() -> None:
    raw_path = HERE / "walkthrough_raw.jsonl"

    # ── Stage 1: Raw Input ───────────────────────────────────────────────
    console.print("\n[bold yellow]═══ STAGE 1: RAW INPUT ═══[/bold yellow]")
    console.print("This is what YOU provide. Each line is a product description + its correct HTS code.\n")

    raw_records = read_jsonl(raw_path)
    console.print(f"Loaded {len(raw_records)} raw records\n")
    show_json("Raw example (what you put in data/raw/)", raw_records[0])

    raw_examples = [
        RawExample(
            description=r["description"],
            hts_code=r["hts_code"],
            source=r.get("source", ""),
            metadata={k: v for k, v in r.items() if k not in ("description", "hts_code", "source")},
        )
        for r in raw_records
    ]

    # ── Stage 2: After Normalization ─────────────────────────────────────
    console.print("\n[bold yellow]═══ STAGE 2: AFTER NORMALIZATION ═══[/bold yellow]")
    console.print("Codes become digit-only. Duplicates removed. Length-filtered.\n")

    config = DataConfig(
        normalization=NormalizationConfig(
            min_description_length=10,
            max_description_length=2048,
        ),
    )
    normalized = normalize_examples(raw_examples, config.normalization)

    show_json("Normalized example (code is now digits-only)", normalized[0].model_dump())

    # ── Stage 3: After Build Examples ────────────────────────────────────
    console.print("\n[bold yellow]═══ STAGE 3: AFTER BUILD EXAMPLES ═══[/bold yellow]")
    console.print("Each example gets assigned a task type. Rerank gets candidates, RAG gets context.\n")

    training_examples = build_examples(normalized, config, rng=random.Random(42))

    # Show one of each type
    by_type = {}
    for ex in training_examples:
        if ex.task_type not in by_type and not ex.abstain:
            by_type[ex.task_type] = ex
    for task_type in ["rerank", "rag_classify", "direct_classify"]:
        if task_type in by_type:
            show_json(f"Training example — {task_type}", by_type[task_type].model_dump())

    # Show an abstain example if one exists
    abstain_ex = next((ex for ex in training_examples if ex.abstain), None)
    if abstain_ex:
        show_json("Training example — ABSTAIN (corrupted description)", abstain_ex.model_dump())

    # ── Stage 4: After Formatting (what the model sees) ──────────────────
    console.print("\n[bold yellow]═══ STAGE 4: FORMATTED FOR TRAINING (what goes into the model) ═══[/bold yellow]")
    console.print("Each example becomes a chat conversation: system + user + assistant.\n")
    console.print("[dim]The model learns ONLY the assistant response (system+user tokens are masked).[/dim]\n")

    for task_type in ["rerank", "rag_classify", "direct_classify"]:
        if task_type in by_type:
            formatted = format_example(by_type[task_type])
            msgs = formatted["messages"]

            console.print(f"\n[bold magenta]── {task_type.upper()} ──[/bold magenta]")

            # System (truncated)
            sys_preview = msgs[0]["content"][:200] + "..."
            console.print(Panel(sys_preview, title="SYSTEM prompt", border_style="green"))

            # User
            console.print(Panel(msgs[1]["content"], title="USER prompt", border_style="blue"))

            # Assistant
            console.print(Panel(msgs[2]["content"], title="ASSISTANT response (model learns THIS)", border_style="red"))

    # ── Summary ──────────────────────────────────────────────────────────
    console.print("\n[bold yellow]═══ SUMMARY ═══[/bold yellow]")
    console.print(f"""
[bold]Pipeline flow:[/bold]
  1. [cyan]Raw[/cyan]:        {len(raw_records)} records (description + HTS code)
  2. [cyan]Normalized[/cyan]: {len(normalized)} records (digit-only codes, deduped)
  3. [cyan]Built[/cyan]:      {len(training_examples)} examples (task types assigned)
  4. [cyan]Formatted[/cyan]:  {len(training_examples)} chat conversations (system/user/assistant)

[bold]Task distribution:[/bold]
  Rerank:         {sum(1 for e in training_examples if e.task_type == 'rerank')}
  RAG Classify:   {sum(1 for e in training_examples if e.task_type == 'rag_classify')}
  Direct Classify:{sum(1 for e in training_examples if e.task_type == 'direct_classify')}
  Abstain:        {sum(1 for e in training_examples if e.abstain)}

[bold]What you need to provide:[/bold]
  A JSONL or CSV file with two columns:
    - Product description (natural language)
    - HTS code (any format: 0101.21.0010 or 0101210010)

  Best sources:
    - CBP CROSS rulings (description + classification)
    - Your own labeled data
    - HTS database enriched descriptions
""")


if __name__ == "__main__":
    main()
