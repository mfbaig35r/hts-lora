"""Build a stat-suffix validation index from training data.

Walks data/formatted/train.jsonl, extracts every non-abstain gold
hts_code, and produces a JSON index for Layer A post-processing:

    {
      "valid_10digit": {"8544421000": 142, ...},     # code -> training freq
      "completions_by_8digit": {                       # 8-digit -> list of children
        "85444210": ["8544421000", "8544422000"],
        ...
      },
      "_meta": {"source": "training", "n_examples": ..., "n_codes": ...}
    }

This is the path-B index per docs/layer-a-followup-usitc.md. It is
correct for evaluations drawn from the CROSS-ruling distribution but
incomplete for production deployment (codes never seen in training are
flagged as invalid).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import typer
from rich.console import Console

from hts_lora.utils.io import read_jsonl, write_json

app = typer.Typer(help="Build training-derived stat-suffix validation index")
console = Console()


@app.command()
def main(
    input_path: str = typer.Option("data/formatted/train.jsonl", help="Training JSONL"),
    output_path: str = typer.Option(
        "data/external/stat_suffix_index.json",
        help="Where to write the index",
    ),
) -> None:
    records = read_jsonl(input_path)
    console.print(f"Loaded {len(records):,} records from {input_path}")

    valid_freq: Counter[str] = Counter()
    skipped_abstain = 0
    skipped_short = 0

    for r in records:
        if r.get("abstain"):
            skipped_abstain += 1
            continue
        code = r.get("hts_code", "")
        # Normalize: strip dots, pad/trim to 10. Pad shorter codes with 0
        # at the end (matches the convention in hts_codes.normalize_code).
        cleaned = code.replace(".", "").strip()
        if not cleaned.isdigit() or len(cleaned) < 8:
            skipped_short += 1
            continue
        if len(cleaned) < 10:
            cleaned = cleaned.ljust(10, "0")
        elif len(cleaned) > 10:
            cleaned = cleaned[:10]
        valid_freq[cleaned] += 1

    # Group 10-digit codes by their 8-digit prefix
    by_8digit: dict[str, list[str]] = defaultdict(list)
    for code in valid_freq:
        by_8digit[code[:8]].append(code)
    # Sort children by descending freq for deterministic picker behavior
    for prefix in by_8digit:
        by_8digit[prefix].sort(key=lambda c: -valid_freq[c])

    index = {
        "valid_10digit": dict(valid_freq),
        "completions_by_8digit": dict(by_8digit),
        "_meta": {
            "source": "training",
            "n_examples_total": len(records),
            "n_examples_skipped_abstain": skipped_abstain,
            "n_examples_skipped_short": skipped_short,
            "n_distinct_10digit_codes": len(valid_freq),
            "n_distinct_8digit_subheadings": len(by_8digit),
            "max_completions_per_subheading": max(
                (len(v) for v in by_8digit.values()), default=0
            ),
            "median_completions_per_subheading": (
                sorted(len(v) for v in by_8digit.values())[len(by_8digit) // 2]
                if by_8digit
                else 0
            ),
            "source_path": input_path,
        },
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(index, out)

    m = index["_meta"]
    console.print(f"\nWrote index to {out}")
    console.print(f"  Distinct 10-digit codes:       {m['n_distinct_10digit_codes']:,}")
    console.print(f"  Distinct 8-digit subheadings:  {m['n_distinct_8digit_subheadings']:,}")
    console.print(f"  Max children per subheading:   {m['max_completions_per_subheading']}")
    console.print(f"  Median children per subheading: {m['median_completions_per_subheading']}")
    console.print(f"  Skipped (abstain):              {skipped_abstain:,}")
    console.print(f"  Skipped (short/invalid code):   {skipped_short:,}")


if __name__ == "__main__":
    app()
