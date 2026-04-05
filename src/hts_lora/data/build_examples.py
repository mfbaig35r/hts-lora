"""Transform RawExamples into TrainingExamples for hierarchical classification.

Stage 1 training has two task types:
- hierarchical_classify: Rich product description → full HTS hierarchy path
- abstention: Corrupted/vague description → model should refuse to classify

Input variants (for hierarchical_classify):
- rich (45%): description + materials + product_use + country
- minimal (25%): description only
- glossary_enriched (20%): description + injected glossary definitions
- materials_only (10%): materials + product_use (no description)
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from hts_lora.data.ingest import RawExample
from hts_lora.utils.config import DataConfig
from hts_lora.utils.hts_codes import chapter, format_code, heading, subheading
from hts_lora.utils.logging import get_logger

logger = get_logger("data.build")

TaskType = Literal["hierarchical_classify", "abstention"]
InputVariant = Literal["rich", "minimal", "glossary_enriched", "materials_only"]
AbstainCategory = Literal["vague_description", "missing_materials", "ambiguous_use"]


class TrainingExample(BaseModel):
    """A fully prepared training example ready for formatting."""

    description: str
    hts_code: str
    task_type: TaskType
    source: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Enrichment fields carried from RawExample
    ruling_number: str | None = None
    reasoning: str | None = None
    hts_text: str | None = None
    materials: str | None = None
    product_use: str | None = None
    country: str | None = None

    # Hierarchy context
    chapter_code: str | None = None
    chapter_description: str | None = None
    heading_code: str | None = None
    heading_description: str | None = None
    tariff_description: str | None = None

    # V2 fields
    input_variant: InputVariant = "rich"
    glossary_terms: list[dict[str, str]] = Field(default_factory=list)
    abstain: bool = False
    abstain_category: AbstainCategory | None = None

    # Legacy compat (not used in v2, kept for test fixtures)
    candidates: list[str] = Field(default_factory=list)
    context: str = ""


def build_examples(
    raw_examples: list[RawExample],
    config: DataConfig,
    rng: random.Random | None = None,
) -> list[TrainingExample]:
    """Build training examples with hierarchical classification and abstention."""
    if rng is None:
        rng = random.Random(config.split.seed)

    # Load glossary if configured
    glossary_dict = _load_glossary(config.glossary_path) if config.glossary_path else {}
    glossary = _GlossaryMatcher(glossary_dict) if glossary_dict else None

    # Input variant weights
    variant_weights = [0.45, 0.25, 0.20, 0.10]
    variant_names: list[InputVariant] = ["rich", "minimal", "glossary_enriched", "materials_only"]

    # Build classify examples
    examples: list[TrainingExample] = []
    for raw in raw_examples:
        variant: InputVariant = rng.choices(variant_names, weights=variant_weights, k=1)[0]

        # Fall back to "rich" if materials_only but no materials data
        if variant == "materials_only" and not (raw.materials or raw.product_use):
            variant = "rich"
        # Fall back to "rich" if glossary_enriched but no glossary loaded
        if variant == "glossary_enriched" and glossary is None:
            variant = "rich"

        glossary_terms: list[dict[str, str]] = []
        if variant == "glossary_enriched" and glossary is not None:
            glossary_terms = glossary.find_terms(raw.description, rng)

        ex = TrainingExample(
            description=raw.description,
            hts_code=raw.hts_code,
            task_type="hierarchical_classify",
            source=raw.source,
            metadata=raw.metadata,
            ruling_number=raw.ruling_number,
            reasoning=raw.reasoning,
            hts_text=raw.hts_text,
            materials=raw.materials,
            product_use=raw.product_use,
            country=raw.country,
            chapter_code=raw.chapter_code,
            chapter_description=raw.chapter_description,
            heading_code=raw.heading_code,
            heading_description=raw.heading_description,
            tariff_description=raw.tariff_description,
            input_variant=variant,
            glossary_terms=glossary_terms,
        )
        examples.append(ex)

    # Inject abstention examples
    examples = _inject_abstentions(examples, config.abstention.rate, config.abstention.categories, rng)

    # Apply frequency cap
    if config.frequency_cap.enabled:
        examples = _apply_frequency_cap(
            examples, config.frequency_cap.max_per_code, rng
        )

    # Log stats
    variant_counts = Counter(e.input_variant for e in examples if not e.abstain)
    logger.info(
        f"Built {len(examples)} training examples "
        f"(classify={sum(1 for e in examples if e.task_type == 'hierarchical_classify' and not e.abstain)}, "
        f"abstain={sum(1 for e in examples if e.abstain)}, "
        f"variants={dict(variant_counts)})"
    )
    return examples


def _load_glossary(glossary_path: str) -> dict[str, str]:
    """Load glossary JSONL into a term → definition dict."""
    path = Path(glossary_path)
    if not path.exists():
        logger.warning(f"Glossary not found: {path}")
        return {}

    glossary: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            term = entry.get("term", "").strip().lower()
            definition = entry.get("definition", "").strip()
            if term and definition:
                glossary[term] = definition

    logger.info(f"Loaded {len(glossary)} glossary terms")
    return glossary


class _GlossaryMatcher:
    """Pre-compiled regex matcher for efficient glossary term lookup."""

    def __init__(self, glossary: dict[str, str]):
        # Filter terms and build a single alternation regex
        self._glossary = glossary
        terms = [t for t in glossary if len(t) >= 3]
        if terms:
            # Sort longest first so longer terms match preferentially
            terms.sort(key=len, reverse=True)
            escaped = [re.escape(t) for t in terms]
            self._pattern = re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)
        else:
            self._pattern = None

    def find_terms(
        self, description: str, rng: random.Random, max_terms: int = 3
    ) -> list[dict[str, str]]:
        if self._pattern is None:
            return []
        found = set(self._pattern.findall(description.lower()))
        matches = [
            {"term": t, "definition": self._glossary[t]}
            for t in found if t in self._glossary
        ]
        if len(matches) > max_terms:
            matches = rng.sample(matches, max_terms)
        return matches


def _find_glossary_terms(
    description: str,
    glossary: dict[str, str] | _GlossaryMatcher,
    rng: random.Random,
    max_terms: int = 3,
) -> list[dict[str, str]]:
    """Find glossary terms that appear in a description."""
    if isinstance(glossary, _GlossaryMatcher):
        return glossary.find_terms(description, rng, max_terms)

    # Fallback for plain dict (shouldn't happen in normal flow)
    matcher = _GlossaryMatcher(glossary)
    return matcher.find_terms(description, rng, max_terms)


def _inject_abstentions(
    examples: list[TrainingExample],
    rate: float,
    categories: list[str],
    rng: random.Random,
) -> list[TrainingExample]:
    """Create abstention examples by corrupting real descriptions.

    Three categories:
    - vague_description: Replace with generic/vague text
    - missing_materials: Remove material-specific information
    - ambiguous_use: Remove product use context
    """
    n_abstain = int(len(examples) * rate)
    if n_abstain == 0:
        return examples

    indices = rng.sample(range(len(examples)), min(n_abstain, len(examples)))
    for i in indices:
        ex = examples[i]
        category: AbstainCategory = rng.choice(categories)  # type: ignore[assignment]
        ex.description = _corrupt_for_category(ex.description, category, rng)
        ex.abstain = True
        ex.abstain_category = category
        ex.task_type = "abstention"
        ex.hts_code = "__ABSTAIN__"

    return examples


def _corrupt_for_category(
    text: str, category: AbstainCategory, rng: random.Random
) -> str:
    """Corrupt a description based on the abstention category."""
    if category == "vague_description":
        templates = [
            "miscellaneous goods for general use",
            "product item for commercial purposes",
            "assorted merchandise for import",
            "general commodity for trade",
        ]
        return rng.choice(templates)

    elif category == "missing_materials":
        # Strip material-related words
        material_words = {
            "steel", "iron", "copper", "aluminum", "plastic", "rubber",
            "cotton", "silk", "wool", "leather", "wood", "ceramic",
            "glass", "paper", "nylon", "polyester", "titanium", "brass",
        }
        words = text.split()
        filtered = [w for w in words if w.lower().strip(",.;:") not in material_words]
        result = " ".join(filtered).strip()
        return result if len(result) > 10 else "product of unspecified composition"

    elif category == "ambiguous_use":
        # Take just the first few words, removing context
        words = text.split()
        n = min(3, len(words))
        return " ".join(words[:n]) + "..."

    return text


def _apply_frequency_cap(
    examples: list[TrainingExample],
    max_per_code: int,
    rng: random.Random,
) -> list[TrainingExample]:
    """Cap the number of training examples per HTS code.

    Abstention examples are exempt from capping.
    """
    # Separate abstention from classify
    abstain_examples = [e for e in examples if e.abstain]
    classify_examples = [e for e in examples if not e.abstain]

    # Count per code and cap
    code_counts: Counter[str] = Counter()
    kept: list[TrainingExample] = []
    capped = 0

    # Shuffle first so capping is random, not biased by input order
    rng.shuffle(classify_examples)

    for ex in classify_examples:
        if code_counts[ex.hts_code] < max_per_code:
            kept.append(ex)
            code_counts[ex.hts_code] += 1
        else:
            capped += 1

    if capped:
        logger.info(f"Frequency cap removed {capped} examples (max {max_per_code} per code)")

    return kept + abstain_examples
