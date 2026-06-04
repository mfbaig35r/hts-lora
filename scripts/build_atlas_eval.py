"""Convert ATLAS test set into our v2 eval-pipeline shape.

ATLAS source rows (`data/external/atlas_test.jsonl`) look like:

    {"messages": [
        {"role": "user", "content": "What is the HTS US Code for ...?"},
        {"role": "assistant", "content": "HTS US Code -> 8477.10.9015; 8480.71.8045\\nReasoning -> ..."}
    ]}

We emit (`data/external/atlas_test_v2.jsonl`) records the v2 eval pipeline
understands, with our training system prompt and ground-truth as a list
of acceptable codes (any-of-set scoring per the ATLAS paper).

Multi-code rows in ATLAS (16/200 in v1, ~8%) are preserved as list-gold.
"""

from __future__ import annotations

import re
from pathlib import Path

import typer
from rich.console import Console

from hts_lora.data.formatters import _SYSTEM_PROMPT
from hts_lora.utils.io import read_jsonl, write_jsonl

app = typer.Typer(help="Convert ATLAS test set to v2 eval shape")
console = Console()

# Find the gold-codes region: everything after "HTS US Code ->" up to the
# Reasoning marker or end. ATLAS uses both clean separators (";", ",") and
# natural language ("or", "and", "for items a-f"), so we grab the region
# then regex-extract code-shaped substrings.
_GOLD_REGION_RE = re.compile(
    r"HTS US Code\s*->\s*(.+?)(?:\n\s*Reasoning|$)",
    re.IGNORECASE | re.DOTALL,
)
# Dotted HTS code: 4 digits . 2 digits . 2-4 digits.
_CODE_RE = re.compile(r"\b(\d{4}\.\d{2}\.\d{2,4})\b")
# Strip leading "What is the HTS US Code for " (any casing, optional trailing ?).
_Q_PREFIX_RE = re.compile(
    r"^\s*what\s+is\s+the\s+hts\s*us\s+code\s+for\s+",
    re.IGNORECASE,
)


def _extract_gold(asst_content: str) -> list[str]:
    m = _GOLD_REGION_RE.search(asst_content)
    region = m.group(1) if m else ""
    codes = _CODE_RE.findall(region)
    # Dedupe preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _question_to_product(q: str) -> str:
    body = _Q_PREFIX_RE.sub("", q).strip()
    # Drop trailing question mark, period, whitespace.
    body = body.rstrip("?. \t")
    return body


@app.command()
def main(
    input_path: str = typer.Option("data/external/atlas_test.jsonl"),
    output_path: str = typer.Option("data/external/atlas_test_v2.jsonl"),
) -> None:
    src = read_jsonl(input_path)
    console.print(f"Loaded {len(src)} ATLAS source records from {input_path}")

    out: list[dict] = []
    skipped_no_gold = 0
    multi_code = 0
    for rec in src:
        messages = rec.get("messages", [])
        if len(messages) < 2:
            skipped_no_gold += 1
            continue

        question = messages[0]["content"]
        gold = _extract_gold(messages[-1]["content"])
        if not gold:
            skipped_no_gold += 1
            continue
        if len(gold) > 1:
            multi_code += 1

        product = _question_to_product(question)
        user_content = f"Product: {product}"

        out.append({
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "hts_code": gold,
            "abstain": False,
            "task_type": "atlas_classify",
            "input_variant": "minimal",
            "description": product,
            "source": "atlas_test",
        })

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, output)

    console.print(f"Wrote {len(out)} records to {output_path}")
    console.print(f"  Multi-code rows: {multi_code} ({multi_code / len(out):.1%})")
    if skipped_no_gold:
        console.print(f"  [yellow]Skipped {skipped_no_gold} rows with no parseable gold code[/yellow]")


if __name__ == "__main__":
    app()
