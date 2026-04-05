"""Export pipeline CLI: extractions → glossary → enrichments → filter → audit.

Usage:
    uv run python scripts/run_export.py --config configs/export.yaml
    uv run python scripts/run_export.py --steps extractions,filter_current_valid
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from hts_lora.export.current_valid import filter_current_valid
from hts_lora.export.enrichments import export_enrichments
from hts_lora.export.extractions import export_extractions
from hts_lora.export.glossary import export_glossary
from hts_lora.export.stats import ExportStats
from hts_lora.utils.config import load_export_config
from hts_lora.utils.io import create_run_dir, snapshot_config, write_json
from hts_lora.utils.logging import setup_logging

app = typer.Typer(help="HTS LoRA export pipeline: DB → raw JSONL")
console = Console()

ALL_STEPS = ["extractions", "glossary", "enrichments", "filter_current_valid", "audit"]


@app.command()
def main(
    config: str = typer.Option("configs/export.yaml", help="Path to export config YAML"),
    steps: str = typer.Option(
        ",".join(ALL_STEPS),
        help="Comma-separated list of steps to run",
    ),
    output_dir: Optional[str] = typer.Option(None, help="Override output directory"),
) -> None:
    """Run the export pipeline."""
    setup_logging()
    cfg = load_export_config(config)

    step_list = [s.strip() for s in steps.split(",")]
    stats = ExportStats()
    stats.start()

    # Create versioned output directory
    base_dir = output_dir or cfg.output.base_dir
    if cfg.output.versioned:
        run_dir = create_run_dir(base_dir, prefix="export")
    else:
        run_dir = Path(base_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Export output: {run_dir}[/bold]\n")
    snapshot_config(cfg, run_dir)

    # Paths
    raw_path = run_dir / "_raw_extractions.jsonl"
    glossary_path = run_dir / "glossary.jsonl"
    enrichments_path = run_dir / "hts6_enrichments.jsonl"
    valid_path = run_dir / "cross_rulings_enriched.jsonl"
    excluded_path = run_dir / "excluded_codes.jsonl"
    audit_path = run_dir / "export_audit.json"

    if "extractions" in step_list:
        if raw_path.exists():
            console.print("[yellow]Step 1: Extractions — output exists, skipping[/yellow]")
            stats.total_extractions = sum(1 for _ in open(raw_path))
        else:
            console.print("[bold]Step 1/5: Export extractions[/bold]")
            t0 = time.time()
            count = export_extractions(cfg, raw_path)
            stats.total_extractions = count
            stats.step_times["extractions"] = round(time.time() - t0, 1)
            console.print(f"  {count:,} extractions → {raw_path.name}")

    if "glossary" in step_list and cfg.glossary.enabled:
        if glossary_path.exists():
            console.print("[yellow]Step 2: Glossary — output exists, skipping[/yellow]")
            stats.glossary_terms = sum(1 for _ in open(glossary_path))
        else:
            console.print("[bold]Step 2/5: Export glossary[/bold]")
            t0 = time.time()
            count = export_glossary(cfg, glossary_path)
            stats.glossary_terms = count
            stats.step_times["glossary"] = round(time.time() - t0, 1)
            console.print(f"  {count:,} terms → {glossary_path.name}")

    if "enrichments" in step_list and cfg.enrichments.enabled:
        if enrichments_path.exists():
            console.print("[yellow]Step 3: Enrichments — output exists, skipping[/yellow]")
            stats.enrichments_exported = sum(1 for _ in open(enrichments_path))
        else:
            console.print("[bold]Step 3/5: Export enrichments[/bold]")
            t0 = time.time()
            count = export_enrichments(cfg, enrichments_path)
            stats.enrichments_exported = count
            stats.step_times["enrichments"] = round(time.time() - t0, 1)
            console.print(f"  {count:,} enrichments → {enrichments_path.name}")

    if "filter_current_valid" in step_list and cfg.current_valid.enabled:
        if valid_path.exists():
            console.print("[yellow]Step 4: Current-valid filter — output exists, skipping[/yellow]")
            stats.valid_codes = sum(1 for _ in open(valid_path))
            if excluded_path.exists():
                stats.excluded_codes = sum(1 for _ in open(excluded_path))
        else:
            if not raw_path.exists():
                console.print("[red]Cannot filter without extractions step[/red]")
                raise typer.Exit(1)
            console.print("[bold]Step 4/5: Filter to current-valid codes[/bold]")
            t0 = time.time()
            kept, excluded = filter_current_valid(
                cfg, raw_path, valid_path,
                excluded_path=excluded_path if cfg.current_valid.log_excluded else None,
            )
            stats.valid_codes = kept
            stats.excluded_codes = excluded
            stats.step_times["filter_current_valid"] = round(time.time() - t0, 1)
            console.print(f"  {kept:,} kept, {excluded:,} excluded → {valid_path.name}")

    if "audit" in step_list:
        console.print("[bold]Step 5/5: Write audit[/bold]")
        stats.complete()
        write_json(stats.to_dict(), audit_path)
        console.print(f"  Audit → {audit_path.name}")

        # Print summary
        console.print("\n[bold]Export Summary[/bold]")
        console.print(f"  Raw extractions:    {stats.total_extractions:,}")
        console.print(f"  After valid filter: {stats.valid_codes:,}")
        console.print(f"  Excluded codes:     {stats.excluded_codes:,}")
        console.print(f"  Glossary terms:     {stats.glossary_terms:,}")
        console.print(f"  HTS6 enrichments:   {stats.enrichments_exported:,}")
        if stats.step_times:
            total_time = sum(stats.step_times.values())
            console.print(f"  Total time:         {total_time:.1f}s")

    console.print("\n[bold green]Export complete![/bold green]")


if __name__ == "__main__":
    app()
