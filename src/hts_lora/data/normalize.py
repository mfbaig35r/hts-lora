"""Normalize HTS codes, deduplicate, and filter examples."""

from __future__ import annotations

from datasketch import MinHash, MinHashLSH

from hts_lora.data.ingest import RawExample
from hts_lora.utils.config import NormalizationConfig
from hts_lora.utils.hts_codes import normalize_code, validate_code
from hts_lora.utils.logging import get_logger

logger = get_logger("data.normalize")


def normalize_examples(examples: list[RawExample], config: NormalizationConfig) -> list[RawExample]:
    """Run the full normalization pipeline: validate, filter, dedup."""
    logger.info(f"Starting normalization of {len(examples)} examples")

    # Step 1: Validate and normalize HTS codes
    valid = []
    invalid_count = 0
    for ex in examples:
        if not validate_code(ex.hts_code):
            invalid_count += 1
            continue
        ex.hts_code = normalize_code(ex.hts_code)
        valid.append(ex)
    if invalid_count:
        logger.warning(f"Dropped {invalid_count} examples with invalid HTS codes")

    # Step 2: Length filters
    filtered = []
    too_short = 0
    too_long = 0
    for ex in valid:
        desc_len = len(ex.description)
        if desc_len < config.min_description_length:
            too_short += 1
        elif desc_len > config.max_description_length:
            too_long += 1
        else:
            filtered.append(ex)
    if too_short:
        logger.info(f"Dropped {too_short} examples below min length ({config.min_description_length})")
    if too_long:
        logger.info(f"Dropped {too_long} examples above max length ({config.max_description_length})")

    # Step 3: Exact dedup on (description, hts_code)
    seen: set[tuple[str, str]] = set()
    exact_deduped = []
    for ex in filtered:
        key = (ex.description.lower().strip(), ex.hts_code)
        if key not in seen:
            seen.add(key)
            exact_deduped.append(ex)
    exact_dupes = len(filtered) - len(exact_deduped)
    if exact_dupes:
        logger.info(f"Removed {exact_dupes} exact duplicates")

    # Step 4: Fuzzy dedup via MinHash LSH
    deduped = _fuzzy_dedup(
        exact_deduped,
        threshold=config.dedup_minhash_threshold,
        num_perm=config.dedup_minhash_num_perm,
    )

    logger.info(f"Normalization complete: {len(examples)} -> {len(deduped)} examples")
    return deduped


def _make_minhash(text: str, num_perm: int) -> MinHash:
    """Create a MinHash from word-level shingles."""
    m = MinHash(num_perm=num_perm)
    words = text.lower().split()
    for i in range(len(words) - 2):
        shingle = " ".join(words[i : i + 3])
        m.update(shingle.encode("utf-8"))
    return m


def _fuzzy_dedup(
    examples: list[RawExample],
    threshold: float,
    num_perm: int,
) -> list[RawExample]:
    """Remove near-duplicate descriptions using MinHash LSH."""
    if not examples:
        return examples

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept: list[RawExample] = []
    removed = 0

    for i, ex in enumerate(examples):
        mh = _make_minhash(ex.description, num_perm)
        key = f"doc_{i}"
        if lsh.query(mh):
            removed += 1
            continue
        try:
            lsh.insert(key, mh)
        except ValueError:
            # Duplicate key — skip
            removed += 1
            continue
        kept.append(ex)

    if removed:
        logger.info(f"Fuzzy dedup removed {removed} near-duplicates")
    return kept
