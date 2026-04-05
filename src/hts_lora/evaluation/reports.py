"""Generate evaluation reports: JSON, Markdown, per-chapter breakdown."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from hts_lora.evaluation.error_analysis import analyze_errors
from hts_lora.evaluation.metrics import compute_metrics
from hts_lora.utils.hts_codes import chapter, match_at_level, validate_code
from hts_lora.utils.io import write_json, write_jsonl
from hts_lora.utils.logging import get_logger

logger = get_logger("evaluation.reports")


def generate_report(
    predictions: list[dict[str, Any]],
    output_dir: str | Path,
    top_k_values: list[int] | None = None,
) -> dict[str, Any]:
    """Generate a full evaluation report.

    Creates:
        - report.json: overall metrics + error summary
        - report.md: human-readable markdown report
        - failures.jsonl: all failed predictions
        - per_chapter.json: metrics broken down by chapter
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute metrics
    metrics = compute_metrics(predictions, top_k_values)

    # Error analysis
    errors = analyze_errors(predictions)

    # Per-chapter breakdown
    per_chapter = _per_chapter_metrics(predictions)

    # Failures
    failures = [
        p for p in predictions
        if p.get("parse_ok") and p.get("prediction")
        and not p.get("abstain", False)
        and p["prediction"].get("predicted_code")
        and not match_at_level(
            str(p["prediction"]["predicted_code"]),
            p["hts_code"],
            "exact",
        )
    ]

    # Assemble report
    report = {
        "metrics": metrics,
        "error_analysis": errors,
        "per_chapter_summary": {
            k: v["metrics"] for k, v in per_chapter.items()
        },
    }

    # Write outputs
    write_json(report, output_dir / "report.json")
    write_json(per_chapter, output_dir / "per_chapter.json")
    write_jsonl(failures, output_dir / "failures.jsonl")
    _write_markdown_report(report, output_dir / "report.md")

    logger.info(f"Report generated at {output_dir}")
    return report


def _per_chapter_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute metrics per chapter."""
    by_chapter: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for p in predictions:
        if p.get("abstain"):
            continue
        chap = chapter(p["hts_code"])
        by_chapter[chap].append(p)

    result = {}
    for chap, preds in sorted(by_chapter.items()):
        metrics = compute_metrics(preds)
        result[chap] = {
            "count": len(preds),
            "metrics": metrics,
        }

    return result


def _write_markdown_report(report: dict[str, Any], path: Path) -> None:
    """Write a human-readable Markdown evaluation report."""
    metrics = report["metrics"]
    errors = report["error_analysis"]
    per_chapter = report.get("per_chapter_summary", {})

    lines = [
        "# HTS LoRA Evaluation Report\n",
        "## Overall Metrics\n",
        f"| Metric | Value |",
        f"|--------|-------|",
    ]

    metric_display = [
        ("Total examples", metrics.get("total", 0), "d"),
        ("JSON parse rate", metrics.get("json_parse_rate", 0), ".1%"),
        ("Exact match", metrics.get("exact_match", 0), ".3f"),
        ("Chapter match", metrics.get("chapter_match", 0), ".3f"),
        ("Heading match", metrics.get("heading_match", 0), ".3f"),
        ("Subheading match", metrics.get("subheading_match", 0), ".3f"),
        ("Abstain rate (on abstain examples)", metrics.get("abstain_rate", 0), ".3f"),
        ("Confidence ECE", metrics.get("confidence_ece", 0), ".4f"),
    ]
    for name, val, fmt in metric_display:
        lines.append(f"| {name} | {val:{fmt}} |")

    # Top-k
    for k in [1, 3, 5]:
        key = f"top_{k}_accuracy"
        if key in metrics:
            lines.append(f"| Top-{k} accuracy (rerank) | {metrics[key]:.3f} |")

    lines.append("")

    # Error breakdown
    lines.append("## Error Analysis\n")
    bucket_counts = errors.get("bucket_counts", {})
    if bucket_counts:
        lines.append(f"Total errors: {errors.get('total_errors', 0)}\n")
        lines.append("| Error Type | Count |")
        lines.append("|------------|-------|")
        for bucket, count in sorted(bucket_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| {bucket} | {count} |")
    else:
        lines.append("No errors detected.\n")

    lines.append("")

    # Per-chapter summary (top 10 by count)
    if per_chapter:
        lines.append("## Per-Chapter Accuracy (top 10 by count)\n")
        lines.append("| Chapter | Count | Exact | Chapter Match | Heading Match |")
        lines.append("|---------|-------|-------|---------------|---------------|")
        sorted_chapters = sorted(per_chapter.items(), key=lambda x: -x[1].get("total", 0))
        for chap, m in sorted_chapters[:10]:
            lines.append(
                f"| {chap} | {m.get('total', 0)} | "
                f"{m.get('exact_match', 0):.3f} | "
                f"{m.get('chapter_match', 0):.3f} | "
                f"{m.get('heading_match', 0):.3f} |"
            )

    path.write_text("\n".join(lines))


def compare_runs(
    run_dirs: list[str | Path],
) -> dict[str, Any]:
    """Compare metrics across multiple evaluation runs."""
    from hts_lora.utils.io import read_json

    comparison: dict[str, Any] = {"runs": []}

    for run_dir in run_dirs:
        report_path = Path(run_dir) / "report.json"
        if not report_path.exists():
            logger.warning(f"No report.json found in {run_dir}")
            continue
        report = read_json(report_path)
        comparison["runs"].append({
            "run_dir": str(run_dir),
            "metrics": report.get("metrics", {}),
        })

    if len(comparison["runs"]) >= 2:
        latest = comparison["runs"][-1]["metrics"]
        previous = comparison["runs"][-2]["metrics"]
        comparison["delta"] = {
            key: round(latest.get(key, 0) - previous.get(key, 0), 4)
            for key in ["exact_match", "chapter_match", "heading_match", "json_parse_rate"]
        }

    return comparison
