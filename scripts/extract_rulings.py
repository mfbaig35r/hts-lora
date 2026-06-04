"""CLI for CROSS ruling LLM extraction pipeline.

Usage:
    uv run python scripts/extract_rulings.py extract --dry-run
    uv run python scripts/extract_rulings.py extract --max-rulings 100
    uv run python scripts/extract_rulings.py extract
    uv run python scripts/extract_rulings.py stats
    uv run python scripts/extract_rulings.py export --output data/raw/cross_rulings.jsonl
    uv run python scripts/extract_rulings.py validate --sample 100
"""

from __future__ import annotations

import json
import logging
import re
import statistics
from pathlib import Path

import typer
from rich.console import Console
from rich.live import Live
from rich.table import Table

console = Console()
app = typer.Typer(help="CROSS ruling LLM extraction pipeline")


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _make_progress_table(
    processed: int, total: int, tokens: int, extracted: int, failed: int, elapsed: float
) -> Table:
    """Build a Rich table for the live progress display."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold cyan", width=20)
    table.add_column(width=30)

    pct = (processed / total * 100) if total > 0 else 0
    bar_width = 30
    filled = int(pct / 100 * bar_width)
    bar = "[green]" + "\u2588" * filled + "[/green]" + "\u2591" * (bar_width - filled)

    # Cost estimate
    input_tok = int(tokens * 0.8)
    output_tok = int(tokens * 0.2)
    cost = (input_tok * 0.20 / 1_000_000) + (output_tok * 1.25 / 1_000_000)

    # ETA
    rate = processed / elapsed if elapsed > 0 else 0
    remaining = (total - processed) / rate if rate > 0 else 0

    table.add_row("Progress", f"{bar} {processed:,}/{total:,} ({pct:.1f}%)")
    table.add_row("Extracted", f"[green]{extracted:,}[/green]  Failed: [red]{failed:,}[/red]")
    table.add_row("Tokens", f"{tokens:,.0f}  Cost: [yellow]${cost:.2f}[/yellow]")
    table.add_row("Speed", f"{rate:.1f}/s  ETA: {remaining:.0f}s")

    return table


@app.command()
def extract(
    concurrency: int = typer.Option(100, help="Max concurrent API calls"),
    chunk_size: int = typer.Option(500, help="Rulings per async chunk"),
    chunk_delay: float = typer.Option(0.5, help="Seconds between chunks"),
    flush_size: int = typer.Option(200, help="Rows before DB flush"),
    max_rulings: int | None = typer.Option(None, help="Limit rulings (for testing)"),
    model: str = typer.Option("gpt-5.4-nano", help="OpenAI model to use"),
    dry_run: bool = typer.Option(False, help="Process 5 rulings, print results, no DB writes"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
) -> None:
    """Extract structured data from CROSS ruling texts via LLM."""
    _setup_logging(verbose)

    from hts_lora.extraction.db import (
        ensure_table,
        get_total_counts,
        get_unextracted_rulings,
    )

    if dry_run:
        _run_dry_run(model)
        return

    counts = get_total_counts()
    console.print("[bold]CROSS Ruling Extraction[/bold]")
    console.print(f"  Total eligible: {counts['total']:,}")
    console.print(f"  Already extracted: {counts['extracted']:,}")
    console.print(f"  Failed: {counts['failed']:,}")
    console.print(f"  Remaining: {counts['remaining']:,}")
    console.print(f"  Model: {model}")
    console.print(f"  Concurrency: {concurrency}")
    if max_rulings:
        console.print(f"  Limit: {max_rulings}")
    console.print()

    if counts["remaining"] == 0:
        console.print("[green]All rulings already extracted![/green]")
        return

    import time

    from hts_lora.extraction.pipeline import extract_all

    t0 = time.time()

    # Live progress display
    state = {"processed": 0, "total": 0, "tokens": 0, "extracted": 0, "failed": 0}

    def progress_callback(**kwargs: int) -> None:
        state.update(kwargs)

    with Live(console=console, refresh_per_second=2) as live:

        def updating_callback(**kwargs: int) -> None:
            state.update(kwargs)
            elapsed = time.time() - t0
            live.update(
                _make_progress_table(
                    state["processed"], state["total"], state["tokens"],
                    state["extracted"], state["failed"], elapsed,
                )
            )

        result = extract_all(
            concurrency=concurrency,
            chunk_size=chunk_size,
            chunk_delay=chunk_delay,
            flush_size=flush_size,
            max_rulings=max_rulings,
            model=model,
            progress_callback=updating_callback,
        )

    console.print()
    console.print("[bold green]Extraction complete![/bold green]")
    console.print(f"  Rulings processed: {result['extracted'] + result['failed']:,}")
    console.print(f"  Successfully extracted: [green]{result['extracted']:,}[/green]")
    console.print(f"  Products found: {result['products']:,}")
    console.print(f"  Failed: [red]{result['failed']:,}[/red]")
    console.print(f"  Total tokens: {result['tokens_total']:,}")
    console.print(f"  Estimated cost: [yellow]${result['cost_estimate']:.2f}[/yellow]")
    console.print(f"  Time: {result['time_seconds']:.1f}s")


def _run_dry_run(model: str) -> None:
    """Process 5 rulings without DB writes — just print results."""
    import asyncio

    from openai import AsyncOpenAI

    from hts_lora.extraction.db import get_unextracted_rulings
    from hts_lora.extraction.parser import parse_extraction
    from hts_lora.extraction.prompts import SYSTEM_PROMPT, build_user_prompt

    console.print("[bold]Dry run — extracting 5 rulings (no DB writes)[/bold]\n")

    rulings = get_unextracted_rulings(limit=5)
    if not rulings:
        console.print("[yellow]No unextracted rulings found.[/yellow]")
        return

    client = AsyncOpenAI()

    async def run() -> None:
        for ruling in rulings:
            console.print(f"[cyan]--- {ruling['ruling_number']} ---[/cyan]")
            console.print(f"Subject: {ruling.get('subject', '')[:100]}")
            console.print(f"Tariffs: {ruling.get('tariffs', [])}")

            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(ruling)},
                ],
                temperature=0.0,
                max_completion_tokens=1000,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            tokens = response.usage.total_tokens if response.usage else 0

            console.print(f"Tokens: {tokens}")
            rows = parse_extraction(content, ruling, tokens)

            if rows:
                for row in rows:
                    console.print(f"\n  [green]Product {row['product_idx']}:[/green]")
                    console.print(f"    Description: {row['description'][:120]}")
                    console.print(f"    HTS code: {row.get('hts_code', 'null')}")
                    console.print(f"    Materials: {row.get('materials', 'null')}")
                    console.print(f"    Use: {row.get('product_use', 'null')}")
                    if row.get("reasoning"):
                        console.print(f"    Reasoning: {row['reasoning'][:120]}")
            else:
                console.print("  [yellow]No products extracted[/yellow]")
            console.print()

    asyncio.run(run())


@app.command()
def stats() -> None:
    """Show extraction progress statistics."""
    _setup_logging()

    from hts_lora.extraction.db import ensure_table, get_total_counts

    ensure_table()
    counts = get_total_counts()

    console.print("[bold]Extraction Progress[/bold]")
    console.print(f"  Total eligible rulings: {counts['total']:,}")
    console.print(f"  Extracted: [green]{counts['extracted']:,}[/green]")
    console.print(f"  Failed: [red]{counts['failed']:,}[/red]")
    console.print(f"  Remaining: [yellow]{counts['remaining']:,}[/yellow]")

    if counts["total"] > 0:
        pct = counts["extracted"] / counts["total"] * 100
        console.print(f"  Progress: {pct:.1f}%")

    # Estimate cost for remaining
    remaining = counts["remaining"]
    est_tokens = remaining * 1200  # ~1000 input + 200 output per ruling
    input_tok = int(est_tokens * 0.8)
    output_tok = int(est_tokens * 0.2)
    cost = (input_tok * 0.20 / 1_000_000) + (output_tok * 1.25 / 1_000_000)
    console.print(f"\n  Est. cost to complete: [yellow]${cost:.2f}[/yellow]")


@app.command()
def export(
    output: str = typer.Option("data/raw/cross_rulings.jsonl", help="Output JSONL path"),
    min_length: int = typer.Option(20, help="Minimum description length"),
) -> None:
    """Export extractions to JSONL file."""
    _setup_logging()

    from hts_lora.extraction.db import export_to_jsonl

    console.print(f"[bold]Exporting to {output}[/bold]")
    count = export_to_jsonl(output, min_desc_length=min_length)
    console.print(f"[green]Exported {count:,} records[/green]")

    # Show sample
    if count > 0:
        console.print("\n[bold]Sample records:[/bold]")
        with open(output) as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                record = json.loads(line)
                console.print(f"\n[cyan]#{i + 1}[/cyan] [{record['ruling_number']}]")
                console.print(f"  desc: {record['description'][:120]}")
                console.print(f"  code: {record.get('hts_code', 'null')}")
                if record.get("reasoning"):
                    console.print(f"  reasoning: {record['reasoning'][:100]}")


@app.command()
def validate(
    sample: int = typer.Option(100, help="Sample size for validation"),
) -> None:
    """Spot-check extraction quality on a random sample."""
    _setup_logging()

    from hts_lora.extraction.db import get_validation_sample

    console.print(f"[bold]Validating random sample of {sample} extractions[/bold]\n")

    rows = get_validation_sample(sample)
    if not rows:
        console.print("[yellow]No extractions found.[/yellow]")
        return

    # Description length distribution
    desc_lengths = [len(r["description"]) for r in rows]
    console.print("[bold]Description Length Distribution[/bold]")
    console.print(f"  Min: {min(desc_lengths)}")
    console.print(f"  Max: {max(desc_lengths)}")
    console.print(f"  Mean: {statistics.mean(desc_lengths):.0f}")
    console.print(f"  Median: {statistics.median(desc_lengths):.0f}")

    # HTS code match rate
    total_with_code = sum(1 for r in rows if r.get("hts_code"))
    matches = 0
    for r in rows:
        if r.get("hts_code") and r.get("tariffs"):
            code_digits = re.sub(r"[\s.\-]", "", r["hts_code"])
            tariff_digits = {re.sub(r"[\s.\-]", "", t) for t in r["tariffs"]}
            if any(
                td.startswith(code_digits[:8]) or code_digits.startswith(td[:8])
                for td in tariff_digits
            ):
                matches += 1

    console.print(f"\n[bold]HTS Code Quality[/bold]")
    console.print(f"  Has hts_code: {total_with_code}/{len(rows)} ({total_with_code / len(rows):.0%})")
    if total_with_code > 0:
        console.print(
            f"  Matches ruling tariffs: {matches}/{total_with_code} "
            f"({matches / total_with_code:.0%})"
        )

    # Null field rates
    console.print(f"\n[bold]Field Coverage[/bold]")
    for field in ("hts_code", "product_use", "materials", "reasoning", "hts_text", "country"):
        non_null = sum(1 for r in rows if r.get(field))
        console.print(f"  {field}: {non_null}/{len(rows)} ({non_null / len(rows):.0%})")

    # Multi-product rulings
    ruling_numbers = [r["ruling_number"] for r in rows]
    unique_rulings = set(ruling_numbers)
    multi = len(ruling_numbers) - len(unique_rulings)
    console.print(f"\n[bold]Multi-product[/bold]")
    console.print(f"  Unique rulings in sample: {len(unique_rulings)}")
    console.print(f"  Extra products (multi-product rulings): {multi}")

    # Show 5 random examples
    console.print(f"\n[bold]Sample Examples[/bold]")
    import random
    examples = random.sample(rows, min(5, len(rows)))
    for r in examples:
        console.print(f"\n[cyan]{r['ruling_number']}[/cyan] (idx {r.get('product_idx', 0)})")
        console.print(f"  [dim]Subject:[/dim] {(r.get('subject') or '')[:100]}")
        console.print(f"  [green]Description:[/green] {r['description'][:150]}")
        console.print(f"  Code: {r.get('hts_code', 'null')}  Country: {r.get('country', 'null')}")


if __name__ == "__main__":
    app()
