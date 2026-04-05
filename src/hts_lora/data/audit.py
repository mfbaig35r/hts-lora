"""Data audit: statistics on sources, chapters, task types, variants, frequency caps."""

from __future__ import annotations

from collections import Counter
from typing import Any

from hts_lora.data.build_examples import TrainingExample
from hts_lora.utils.hts_codes import chapter
from hts_lora.utils.logging import get_logger

logger = get_logger("data.audit")


def audit_examples(examples: list[TrainingExample]) -> dict[str, Any]:
    """Compute audit statistics for a set of training examples."""
    if not examples:
        return {"total": 0}

    # Counts by source
    source_counts = Counter(ex.source for ex in examples)

    # Counts by chapter
    chapter_counts = Counter(
        chapter(ex.hts_code) if not ex.abstain else "__abstain__"
        for ex in examples
    )

    # Counts by task type
    task_counts = Counter(ex.task_type for ex in examples)

    # Input variant distribution
    variant_counts = Counter(
        ex.input_variant for ex in examples if not ex.abstain
    )

    # Abstention stats
    n_abstain = sum(1 for ex in examples if ex.abstain)
    abstain_categories = Counter(
        ex.abstain_category for ex in examples if ex.abstain and ex.abstain_category
    )

    # Description length stats
    desc_lengths = [len(ex.description) for ex in examples]
    desc_lengths.sort()

    # Label frequency (top 20)
    label_counts = Counter(ex.hts_code for ex in examples if not ex.abstain)
    top_labels = label_counts.most_common(20)

    # Code histogram (count of codes with N examples)
    freq_dist = Counter(label_counts.values())
    code_histogram = {
        "1_example": sum(1 for c in label_counts.values() if c == 1),
        "2_5_examples": sum(1 for c in label_counts.values() if 2 <= c <= 5),
        "6_20_examples": sum(1 for c in label_counts.values() if 6 <= c <= 20),
        "21_50_examples": sum(1 for c in label_counts.values() if 21 <= c <= 50),
        "51_100_examples": sum(1 for c in label_counts.values() if 51 <= c <= 100),
        "over_100_examples": sum(1 for c in label_counts.values() if c > 100),
    }

    # Hierarchy coverage
    unique_chapters = len({c for c in chapter_counts if c != "__abstain__"})
    unique_headings = len({
        ex.hts_code[:4] for ex in examples
        if not ex.abstain and len(ex.hts_code) >= 4
    })

    # Imbalance: ratio of most common to least common chapter
    chapter_values = [v for k, v in chapter_counts.items() if k != "__abstain__"]
    imbalance_ratio = max(chapter_values) / max(min(chapter_values), 1) if chapter_values else 0

    # Enrichment coverage
    has_reasoning = sum(1 for ex in examples if ex.reasoning and not ex.abstain)
    has_materials = sum(1 for ex in examples if ex.materials and not ex.abstain)
    has_hierarchy = sum(1 for ex in examples if ex.chapter_description and not ex.abstain)
    has_glossary = sum(1 for ex in examples if ex.glossary_terms and not ex.abstain)
    non_abstain = sum(1 for ex in examples if not ex.abstain)

    stats: dict[str, Any] = {
        "total": len(examples),
        "by_source": dict(source_counts.most_common()),
        "by_chapter": dict(sorted(chapter_counts.items())),
        "by_task_type": dict(task_counts),
        "by_input_variant": dict(variant_counts.most_common()),
        "abstention_count": n_abstain,
        "abstention_rate": round(n_abstain / len(examples), 4),
        "abstention_categories": dict(abstain_categories),
        "unique_codes": len(label_counts),
        "unique_chapters": unique_chapters,
        "unique_headings": unique_headings,
        "top_20_codes": [{"code": c, "count": n} for c, n in top_labels],
        "code_histogram": code_histogram,
        "chapter_imbalance_ratio": round(imbalance_ratio, 2),
        "description_length": {
            "min": desc_lengths[0],
            "max": desc_lengths[-1],
            "median": desc_lengths[len(desc_lengths) // 2],
            "mean": round(sum(desc_lengths) / len(desc_lengths), 1),
        },
        "enrichment_coverage": {
            "has_reasoning": has_reasoning,
            "has_materials": has_materials,
            "has_hierarchy": has_hierarchy,
            "has_glossary": has_glossary,
            "total_non_abstain": non_abstain,
        },
    }

    logger.info(
        f"Audit: {stats['total']} examples, {stats['unique_codes']} unique codes, "
        f"{stats['unique_chapters']} chapters, {stats['abstention_count']} abstentions, "
        f"imbalance ratio {stats['chapter_imbalance_ratio']}"
    )
    return stats


def print_audit(stats: dict[str, Any]) -> None:
    """Print a human-readable audit summary."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    console.print(f"\n[bold]Dataset Audit[/bold] — {stats.get('total', 0)} examples")

    # Task type table
    table = Table(title="Task Type Distribution")
    table.add_column("Task Type")
    table.add_column("Count", justify="right")
    for task, count in stats.get("by_task_type", {}).items():
        table.add_row(task, str(count))
    console.print(table)

    # Input variant table
    variants = stats.get("by_input_variant", {})
    if variants:
        vtable = Table(title="Input Variant Distribution")
        vtable.add_column("Variant")
        vtable.add_column("Count", justify="right")
        for variant, count in variants.items():
            vtable.add_row(variant, str(count))
        console.print(vtable)

    # Hierarchy coverage
    console.print(f"\nChapters covered: {stats.get('unique_chapters', 0)}")
    console.print(f"Unique headings: {stats.get('unique_headings', 0)}")
    console.print(f"Chapter imbalance ratio: {stats.get('chapter_imbalance_ratio', 'N/A')}")

    # Code histogram
    hist = stats.get("code_histogram", {})
    if hist:
        console.print("\nCode frequency distribution:")
        for bucket, count in hist.items():
            console.print(f"  {bucket}: {count}")

    # Length stats
    lengths = stats.get("description_length", {})
    console.print(
        f"\nDescription length: min={lengths.get('min')}, "
        f"max={lengths.get('max')}, median={lengths.get('median')}, "
        f"mean={lengths.get('mean')}"
    )

    # Abstention details
    console.print(f"\nAbstention rate: {stats.get('abstention_rate', 0):.1%}")
    categories = stats.get("abstention_categories", {})
    if categories:
        for cat, count in categories.items():
            console.print(f"  {cat}: {count}")

    # Enrichment coverage
    enrich = stats.get("enrichment_coverage", {})
    if enrich:
        total = enrich.get("total_non_abstain", 1)
        console.print(f"\nEnrichment coverage (of {total:,} non-abstain):")
        console.print(f"  Has reasoning:  {enrich.get('has_reasoning', 0):,}")
        console.print(f"  Has materials:  {enrich.get('has_materials', 0):,}")
        console.print(f"  Has hierarchy:  {enrich.get('has_hierarchy', 0):,}")
        console.print(f"  Has glossary:   {enrich.get('has_glossary', 0):,}")

    console.print(f"\nUnique codes: {stats.get('unique_codes', 0)}")
