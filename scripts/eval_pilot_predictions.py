"""Run the v2 eval pipeline against the existing 50 MLX pilot predictions.

Smoke test only. Reuses parse_output + metrics + reports against
outputs/mlx_pilot/pilot_predictions.jsonl, which has shape
{index, parse_ok, predicted_code, expected_code, generated, expected}.
We re-parse `generated` through parse_prediction() to produce the
ParsedPrediction the eval pipeline expects.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from dataclasses import asdict

from hts_lora.evaluation.reports import generate_report
from hts_lora.inference.parse_output import parse_prediction
from hts_lora.utils.io import read_jsonl

app = typer.Typer(help="Smoke-test v2 eval pipeline against pilot predictions")
console = Console()


def _adapt(record: dict) -> dict:
    """Convert a pilot prediction record to the eval-pipeline shape.

    Pilot data uses the sentinel "__ABSTAIN__" for expected_code on
    examples whose ground-truth answer is "Cannot classify". We flip
    that into the abstain=True / hts_code="" shape the pipeline wants.
    """
    parsed = parse_prediction(record.get("generated", ""))
    is_abstain_gt = record.get("expected_code") == "__ABSTAIN__"
    return {
        "hts_code": "" if is_abstain_gt else record["expected_code"],
        "prediction": asdict(parsed),
        "parse_ok": parsed.parse_ok,
        "abstain": is_abstain_gt,
        "description": "",
        "raw": record.get("generated", ""),
    }


@app.command()
def main(
    pilot_path: str = typer.Option(
        "outputs/mlx_pilot/pilot_predictions.jsonl",
        help="Pilot predictions JSONL",
    ),
    output_dir: str = typer.Option(
        "outputs/mlx_pilot/eval_smoketest",
        help="Where to write report.json / report.md / failures.jsonl / per_chapter.json",
    ),
) -> None:
    pilot_records = read_jsonl(pilot_path)
    console.print(f"Loaded {len(pilot_records)} pilot predictions from {pilot_path}")

    adapted = [_adapt(r) for r in pilot_records]
    console.print(f"Adapted to eval shape. Sample parse_ok rate: "
                  f"{sum(1 for a in adapted if a['parse_ok']) / len(adapted):.1%}")

    out = Path(output_dir)
    report = generate_report(adapted, out)

    m = report["metrics"]
    console.print("\n[bold]Smoke-test results (50 examples):[/bold]")
    console.print(f"  Total:                  {m.get('total', 0)}")
    console.print(f"  Parse rate:             {m.get('parse_rate', 0):.1%}")
    console.print(f"  Exact match:            {m.get('exact_match', 0):.3f}")
    console.print(f"  Chapter match:          {m.get('chapter_match', 0):.3f}")
    console.print(f"  Heading match:          {m.get('heading_match', 0):.3f}")
    console.print(f"  Subheading match:       {m.get('subheading_match', 0):.3f}")
    console.print(f"  Hierarchy consistency:  {m.get('hierarchy_consistency', 0):.3f}")
    console.print(f"\nReport: {out}")


if __name__ == "__main__":
    app()
