"""Convert TrainingExamples to chat-formatted JSONL for HF training.

v2 format: hierarchical classification with structured text output.

Each example becomes a messages list:
  - system: "detailed thinking off\n\n{classification expert prompt}"
  - user: product info (4 variants based on input_variant field)
  - assistant: "<think>\n</think>\n\n{hierarchy + reasoning}" or abstention
"""

from __future__ import annotations

from typing import Any

from hts_lora.data.build_examples import TrainingExample
from hts_lora.utils.hts_codes import chapter, format_code, heading, subheading
from hts_lora.utils.logging import get_logger

logger = get_logger("data.formatters")

# ── System Prompt ─────────────��─────────────────────────────────────────────

_SYSTEM_PROMPT = """\
detailed thinking off

You are an expert in Harmonized Tariff Schedule (HTS) classification. Given a product description, classify it to the correct HTS code by identifying the full tariff hierarchy.

For each product, provide:
1. The chapter (2-digit), heading (4-digit), subheading (6-digit), and full HTS code (8-digit)
2. A brief reasoning explaining why this classification applies
3. The "provides for" text from the tariff schedule

If the description is too vague, lacks critical details, or is ambiguous, respond with "Cannot classify" and explain what information is missing."""


# ── User Prompt Variants ────────────���───────────────────────────────────────


def _format_user_rich(ex: TrainingExample) -> str:
    """Rich variant: description + materials + product_use + country."""
    parts = [f"Product: {ex.description}"]
    if ex.materials:
        parts.append(f"Materials: {ex.materials}")
    if ex.product_use:
        parts.append(f"Use: {ex.product_use}")
    if ex.country:
        parts.append(f"Country of origin: {ex.country}")
    return "\n".join(parts)


def _format_user_minimal(ex: TrainingExample) -> str:
    """Minimal variant: description only."""
    return f"Product: {ex.description}"


def _format_user_glossary_enriched(ex: TrainingExample) -> str:
    """Glossary-enriched variant: description + relevant definitions."""
    parts = [f"Product: {ex.description}"]
    if ex.glossary_terms:
        defs = "\n".join(
            f"- {t['term']}: {t['definition']}" for t in ex.glossary_terms
        )
        parts.append(f"\nRelevant trade terms:\n{defs}")
    return "\n".join(parts)


def _format_user_materials_only(ex: TrainingExample) -> str:
    """Materials-only variant: materials + use, no description."""
    parts = []
    if ex.materials:
        parts.append(f"Materials: {ex.materials}")
    if ex.product_use:
        parts.append(f"Use: {ex.product_use}")
    if ex.country:
        parts.append(f"Country of origin: {ex.country}")
    return "\n".join(parts) if parts else f"Product: {ex.description}"


_USER_FORMATTERS = {
    "rich": _format_user_rich,
    "minimal": _format_user_minimal,
    "glossary_enriched": _format_user_glossary_enriched,
    "materials_only": _format_user_materials_only,
}


# ── Assistant Response ──────────��───────────────────────────────────────────


def _format_assistant_classify(ex: TrainingExample) -> str:
    """Format the assistant response as structured hierarchy text."""
    code = ex.hts_code
    chap = chapter(code)
    head = heading(code)
    sub = subheading(code)
    formatted = format_code(code)

    # Build hierarchy lines
    chap_desc = ex.chapter_description or ""
    head_desc = ex.heading_description or ""
    tariff_desc = ex.tariff_description or ""

    lines = []
    lines.append(f"Chapter {chap}: {chap_desc}".strip())
    lines.append(f"Heading {head[:2]}.{head[2:]}: {head_desc}".strip())
    lines.append(f"Subheading {sub[:4]}.{sub[4:]}: {tariff_desc}".strip())
    lines.append(f"HTS Code: {formatted}")

    # Reasoning
    reasoning = ex.reasoning or f"Classified under {formatted} based on product description."
    lines.append(f"\nReasoning: {reasoning}")

    # Provides for
    hts_text = ex.hts_text or tariff_desc
    if hts_text:
        lines.append(f"\nProvides for: {hts_text}")

    body = "\n".join(lines)
    return f"<think>\n</think>\n\n{body}"


def _format_assistant_abstain(ex: TrainingExample) -> str:
    """Format the assistant response for abstention."""
    category = ex.abstain_category or "vague_description"

    explanations = {
        "vague_description": (
            "Cannot classify: The product description is too vague or generic "
            "to determine a specific tariff classification. A more detailed "
            "description of the product's composition, function, and intended "
            "use is needed."
        ),
        "missing_materials": (
            "Cannot classify: The product's material composition is not specified. "
            "HTS classification often depends on the primary material (e.g., steel, "
            "plastic, textile fiber). Please provide material details."
        ),
        "ambiguous_use": (
            "Cannot classify: The product's intended use or function is unclear. "
            "Many tariff headings distinguish products by their application. "
            "Please clarify the product's primary use."
        ),
    }

    explanation = explanations.get(category, explanations["vague_description"])
    return f"<think>\n</think>\n\n{explanation}"


# ── Public API ────────��───────────────────────────────��─────────────────────


def format_example(ex: TrainingExample) -> dict[str, Any]:
    """Convert a TrainingExample to a chat messages dict for JSONL output.

    Returns:
        {"messages": [...], "task_type": ..., "hts_code": ..., "abstain": ..., ...}
    """
    system_prompt = _SYSTEM_PROMPT

    if ex.abstain:
        user_content = _format_user_minimal(ex)
        assistant_content = _format_assistant_abstain(ex)
    else:
        formatter = _USER_FORMATTERS.get(ex.input_variant, _format_user_rich)
        user_content = formatter(ex)
        assistant_content = _format_assistant_classify(ex)

    result: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "task_type": ex.task_type,
        "hts_code": ex.hts_code,
        "abstain": ex.abstain,
        "input_variant": ex.input_variant,
    }

    if ex.ruling_number:
        result["ruling_number"] = ex.ruling_number

    return result


def format_dataset(examples: list[TrainingExample]) -> list[dict[str, Any]]:
    """Format a list of TrainingExamples into chat JSONL records."""
    formatted = [format_example(ex) for ex in examples]
    logger.info(f"Formatted {len(formatted)} examples for training")
    return formatted
