"""Export CROSS rulings from PostgreSQL into training-ready JSONL.

Two export modes:
  1. 'subjects' — Uses cross_rulings.subject only (fast, thin descriptions)
  2. 'full'     — Joins with cross_ruling_texts to extract detailed product
                  descriptions + classification reasoning from ruling text

Connects directly to the hts-api database.

Usage:
    uv run python scripts/export_cross_rulings.py --mode full --output data/raw/cross_rulings.jsonl
    uv run python scripts/export_cross_rulings.py --mode subjects --output data/raw/cross_subjects.jsonl
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

console = Console()
app = typer.Typer(help="Export CROSS rulings for HTS LoRA training")

# ── Subject Cleaning ─────────────────────────────────────────────────────────

# Common boilerplate prefixes in ruling subjects
_SUBJECT_PREFIXES = [
    r"^The tariff classification (?:and [^of]+ )?of ",
    r"^Classification of ",
    r"^Tariff classification of ",
    r"^RE:\s*The tariff classification of ",
    r"^RE:\s*",
]
_PREFIX_PATTERN = re.compile("|".join(f"(?:{p})" for p in _SUBJECT_PREFIXES), re.IGNORECASE)


def clean_subject(subject: str) -> str:
    """Strip boilerplate prefix from ruling subject to get product description."""
    cleaned = _PREFIX_PATTERN.sub("", subject).strip()
    # Remove trailing period
    if cleaned.endswith("."):
        cleaned = cleaned[:-1].strip()
    # If cleaning removed everything, fall back to original
    return cleaned if len(cleaned) > 10 else subject.strip()


# ── Ruling Text Parsing ──────────────────────────────────────────────────────

# Pattern to find the product description section in ruling text
# Usually between "RE:" line and "Dear Mr/Ms" or after "The subject product(s)"
_PRODUCT_DESC_PATTERNS = [
    # "The subject product is/are..." or "The subject merchandise is..."
    re.compile(
        r"(?:The subject (?:product|merchandise|item|article)s?\s*(?:,\s*\w+,?\s*)*"
        r"(?:is|are|consists?|comprises?)\s+)(.*?)(?=\n\s*\n|\nThe applicable)",
        re.DOTALL | re.IGNORECASE,
    ),
    # "The product at issue is..."
    re.compile(
        r"(?:The (?:product|merchandise|item|article)s?\s+(?:at issue|under consideration|in question)"
        r"\s+(?:is|are)\s+)(.*?)(?=\n\s*\n|\nThe applicable)",
        re.DOTALL | re.IGNORECASE,
    ),
    # "You have submitted..." or "You describe the product as..."
    re.compile(
        r"(?:You (?:have submitted|describe|state that)\s+)(.*?)(?=\n\s*\n|\nThe applicable)",
        re.DOTALL | re.IGNORECASE,
    ),
]

# Pattern to find the classification reasoning
_REASONING_PATTERNS = [
    # "The applicable subheading for ... will be XXXX.XX.XXXX ... which provides for ..."
    re.compile(
        r"(The applicable (?:sub)?heading.*?which provides for\s*[\"']?.*?[\"']?\.)",
        re.DOTALL | re.IGNORECASE,
    ),
    # Broader: "The applicable subheading ... will be ..."
    re.compile(
        r"(The applicable (?:sub)?heading.*?\.)\s",
        re.DOTALL | re.IGNORECASE,
    ),
]


def extract_product_description(ruling_text: str) -> str | None:
    """Extract the product description paragraph from ruling text."""
    for pattern in _PRODUCT_DESC_PATTERNS:
        match = pattern.search(ruling_text)
        if match:
            desc = match.group(1).strip()
            # Clean up carriage returns and excessive whitespace
            desc = re.sub(r"\r", "", desc)
            desc = re.sub(r"\s+", " ", desc).strip()
            if len(desc) > 20:
                return desc
    return None


def extract_reasoning(ruling_text: str) -> str | None:
    """Extract the classification reasoning paragraph from ruling text."""
    for pattern in _REASONING_PATTERNS:
        match = pattern.search(ruling_text)
        if match:
            reasoning = match.group(1).strip()
            reasoning = re.sub(r"\r", "", reasoning)
            reasoning = re.sub(r"\s+", " ", reasoning).strip()
            if len(reasoning) > 30:
                return reasoning
    return None


# ── Database Export ───────────────────────────────────────────────────────────

def get_db_url() -> str:
    """Get database URL from environment or hts-api config."""
    # Check common env vars
    for var in ["DATABASE_URL", "HTS_DATABASE_URL", "SUPABASE_DB_URL"]:
        url = os.environ.get(var)
        if url:
            return url

    # Try to read from hts-api .env
    env_path = Path.home() / "Projects" / "hts-api" / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("DATABASE_URL="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
            if line.startswith("SUPABASE_DB_URL="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")

    raise RuntimeError(
        "No database URL found. Set DATABASE_URL env var or ensure "
        "~/Projects/hts-api/.env has DATABASE_URL defined."
    )


def export_subjects(output_path: Path, limit: int | None = None) -> int:
    """Export using subject field only (no ruling text needed)."""
    import psycopg2

    db_url = get_db_url()
    console.print(f"Connecting to database...")

    conn = psycopg2.connect(db_url)
    cur = conn.cursor()

    query = """
        SELECT ruling_number, subject, tariffs, ruling_date, ruling_url
        FROM cross_rulings
        WHERE categories ILIKE '%%classification%%'
          AND NOT is_revoked
          AND array_length(tariffs, 1) > 0
          AND subject IS NOT NULL
          AND length(subject) > 20
        ORDER BY ruling_date DESC NULLS LAST
    """
    if limit:
        query += f" LIMIT {limit}"

    cur.execute(query)
    rows = cur.fetchall()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(output_path, "w") as f:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Exporting subjects...", total=len(rows))

            for ruling_number, subject, tariffs, ruling_date, ruling_url in rows:
                description = clean_subject(subject)
                if len(description) < 10:
                    progress.advance(task)
                    continue

                # Use first tariff code as primary, rest as alternates
                primary_code = tariffs[0] if tariffs else None
                if not primary_code:
                    progress.advance(task)
                    continue

                record = {
                    "description": description,
                    "hts_code": primary_code,
                    "source": "cross_ruling",
                    "ruling_number": ruling_number,
                    "ruling_date": str(ruling_date) if ruling_date else None,
                    "ruling_url": ruling_url,
                    "all_tariffs": tariffs,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                progress.advance(task)

    cur.close()
    conn.close()
    return count


def export_full(output_path: Path, limit: int | None = None) -> int:
    """Export with full ruling text parsing for rich descriptions."""
    import psycopg2

    db_url = get_db_url()
    console.print(f"Connecting to database...")

    conn = psycopg2.connect(db_url)
    cur = conn.cursor()

    query = """
        SELECT cr.ruling_number, cr.subject, cr.tariffs, cr.ruling_date,
               cr.ruling_url, crt.ruling_text
        FROM cross_rulings cr
        JOIN cross_ruling_texts crt USING (ruling_number)
        WHERE cr.categories ILIKE '%%classification%%'
          AND NOT cr.is_revoked
          AND array_length(cr.tariffs, 1) > 0
          AND cr.subject IS NOT NULL
          AND crt.ruling_text IS NOT NULL
          AND length(crt.ruling_text) > 100
        ORDER BY cr.ruling_date DESC NULLS LAST
    """
    if limit:
        query += f" LIMIT {limit}"

    cur.execute(query)
    rows = cur.fetchall()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    extracted_desc_count = 0
    extracted_reasoning_count = 0

    with open(output_path, "w") as f:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Exporting with text parsing...", total=len(rows))

            for ruling_number, subject, tariffs, ruling_date, ruling_url, ruling_text in rows:
                primary_code = tariffs[0] if tariffs else None
                if not primary_code:
                    progress.advance(task)
                    continue

                # Try to extract rich description from ruling text
                product_desc = extract_product_description(ruling_text)
                reasoning = extract_reasoning(ruling_text)

                if product_desc:
                    extracted_desc_count += 1

                if reasoning:
                    extracted_reasoning_count += 1

                # Use extracted description if available, fall back to cleaned subject
                description = product_desc or clean_subject(subject)
                if len(description) < 10:
                    progress.advance(task)
                    continue

                record = {
                    "description": description,
                    "hts_code": primary_code,
                    "source": "cross_ruling",
                    "ruling_number": ruling_number,
                    "ruling_date": str(ruling_date) if ruling_date else None,
                    "ruling_url": ruling_url,
                    "all_tariffs": tariffs,
                    "subject": clean_subject(subject),
                    "reasoning": reasoning,
                    "description_source": "ruling_text" if product_desc else "subject",
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                progress.advance(task)

    cur.close()
    conn.close()

    console.print(f"\n[dim]Text extraction stats:[/dim]")
    console.print(f"  Product descriptions extracted: {extracted_desc_count}/{len(rows)} ({extracted_desc_count/max(len(rows),1):.0%})")
    console.print(f"  Reasoning extracted: {extracted_reasoning_count}/{len(rows)} ({extracted_reasoning_count/max(len(rows),1):.0%})")

    return count


# ── CLI ───────────────────────────────────────────────────────────────────────


@app.command()
def main(
    mode: str = typer.Option("full", help="Export mode: 'subjects' (fast) or 'full' (with text parsing)"),
    output: str = typer.Option("data/raw/cross_rulings.jsonl", help="Output JSONL path"),
    limit: int | None = typer.Option(None, help="Limit number of rulings (for testing)"),
) -> None:
    """Export CROSS rulings to JSONL for HTS LoRA training."""
    output_path = Path(output)

    console.print(f"[bold]CROSS Rulings Export[/bold]")
    console.print(f"  Mode: {mode}")
    console.print(f"  Output: {output_path}")
    if limit:
        console.print(f"  Limit: {limit}")
    console.print()

    if mode == "subjects":
        count = export_subjects(output_path, limit)
    elif mode == "full":
        count = export_full(output_path, limit)
    else:
        console.print(f"[red]Unknown mode: {mode}. Use 'subjects' or 'full'.[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold green]Exported {count:,} rulings to {output_path}[/bold green]")

    # Show sample
    console.print("\n[bold]Sample records:[/bold]")
    with open(output_path) as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            record = json.loads(line)
            console.print(f"\n[cyan]#{i+1}[/cyan] [{record.get('ruling_number')}]")
            console.print(f"  desc: {record['description'][:120]}...")
            console.print(f"  code: {record['hts_code']}")
            if record.get("reasoning"):
                console.print(f"  reasoning: {record['reasoning'][:120]}...")


if __name__ == "__main__":
    app()
