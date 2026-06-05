"""Apply Layer A (stat-suffix validator) to a saved predictions.jsonl.

Reads <eval_dir>/predictions.jsonl, runs each prediction through
Layer A using the training-derived index, writes:

    <eval_dir>_postprocessed/
        predictions.jsonl   - same shape as input, with hts_code possibly rewritten
        report.json         - re-scored metrics
        report.md           - markdown summary with v0 vs v1 deltas
        failures.jsonl
        per_chapter.json
        stat_suffix_audit.jsonl  - one line per prediction, what Layer A did

The model is not loaded. No GPU. No re-inference.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from hts_lora.evaluation.reports import generate_report
from hts_lora.inference.parse_output import ParsedPrediction
from hts_lora.postprocess.stat_suffix import (
    StatSuffixIndex,
    validate_and_complete,
)
from hts_lora.utils.io import read_jsonl, write_jsonl

app = typer.Typer(help="Apply Layer A to existing predictions and re-score")
console = Console()


@app.command()
def main(
    eval_dir: str = typer.Argument(..., help="Path to an eval output dir containing predictions.jsonl"),
    index_path: str = typer.Option(
        "data/external/stat_suffix_index.json",
        help="Path to the stat-suffix index",
    ),
    output_suffix: str = typer.Option(
        "_postprocessed",
        help="Suffix appended to eval_dir for the output directory",
    ),
) -> None:
    src = Path(eval_dir)
    src_predictions = src / "predictions.jsonl"
    if not src_predictions.exists():
        raise typer.Exit(f"No predictions.jsonl found at {src_predictions}")

    dst = src.with_name(src.name + output_suffix)
    dst.mkdir(parents=True, exist_ok=True)

    console.print("[bold]Applying Layer A[/bold]")
    console.print(f"  Source predictions: {src_predictions}")
    console.print(f"  Index:              {index_path}")
    console.print(f"  Output dir:         {dst}")

    index = StatSuffixIndex.load(index_path)
    console.print(
        f"  Index loaded: {len(index.valid_10digit):,} 10-digit codes, "
        f"{len(index.completions_by_8digit):,} 8-digit subheadings"
    )

    records = read_jsonl(src_predictions)
    console.print(f"  Loaded {len(records):,} predictions")

    new_records: list[dict] = []
    audit: list[dict] = []
    reason_counts: dict[str, int] = {}
    changed_count = 0

    for r in records:
        pred_dict = r.get("prediction") or {}
        pred = ParsedPrediction(**pred_dict) if pred_dict else ParsedPrediction()
        out = validate_and_complete(pred, index)

        reason_counts[out.reason] = reason_counts.get(out.reason, 0) + 1
        if out.changed:
            changed_count += 1

        new_record = dict(r)
        new_record["prediction"] = asdict(out.prediction)
        new_records.append(new_record)

        audit.append({
            "hts_code_gold": r.get("hts_code"),
            "hts_code_before": out.original_hts_code,
            "hts_code_after": out.chosen_hts_code,
            "changed": out.changed,
            "reason": out.reason,
        })

    write_jsonl(new_records, dst / "predictions.jsonl")
    write_jsonl(audit, dst / "stat_suffix_audit.jsonl")

    # Re-score with the existing eval pipeline.
    report = generate_report(new_records, dst)

    # Pull original metrics for comparison
    original_report_path = src / "report.json"
    orig_metrics = {}
    if original_report_path.exists():
        orig_metrics = json.loads(original_report_path.read_text()).get("metrics", {})

    new_metrics = report["metrics"]

    table = Table(title="Layer A: before vs after")
    table.add_column("Metric")
    table.add_column("Before", justify="right")
    table.add_column("After", justify="right")
    table.add_column("Delta", justify="right")
    for k in ["exact_match", "chapter_match", "heading_match", "subheading_match",
              "parse_rate", "hierarchy_consistency", "abstain_rate"]:
        before = orig_metrics.get(k, float("nan"))
        after = new_metrics.get(k, float("nan"))
        delta = after - before if isinstance(before, (int, float)) and isinstance(after, (int, float)) else float("nan")
        delta_str = (
            f"{delta:+.4f}" if not (isinstance(delta, float) and delta != delta) else "n/a"
        )
        table.add_row(k, f"{before:.4f}", f"{after:.4f}", delta_str)
    console.print(table)

    console.print("\n[bold]Layer A actions[/bold]")
    console.print(f"  Records changed: {changed_count:,} / {len(records):,} ({changed_count/len(records):.1%})")
    for reason, cnt in sorted(reason_counts.items(), key=lambda x: -x[1]):
        console.print(f"  {reason:30s} {cnt:,}")


if __name__ == "__main__":
    app()
