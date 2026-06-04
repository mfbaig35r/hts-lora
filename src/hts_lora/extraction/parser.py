"""Parse and validate LLM JSON responses from CROSS ruling extraction."""

from __future__ import annotations

import json
import logging
import re

from hts_lora.utils.hts_codes import validate_code

logger = logging.getLogger(__name__)

# Match a JSON object (greedy) — fallback when json.loads fails on wrapped text
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _strip_nul(obj: object) -> object:
    """Recursively strip NUL bytes from strings (PostgreSQL rejects \\x00)."""
    if isinstance(obj, str):
        return obj.replace("\x00", "")
    if isinstance(obj, dict):
        return {k: _strip_nul(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_nul(v) for v in obj]
    return obj


def parse_extraction(
    content: str,
    ruling: dict,
    tokens_used: int = 0,
) -> list[dict] | None:
    """Parse the LLM response into validated extraction rows.

    Returns a list of row dicts ready for DB upsert, or None on total failure.
    """
    # Parse JSON
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(content)
        if not match:
            logger.warning("No JSON found in response for %s", ruling["ruling_number"])
            return None
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            logger.warning("JSON regex fallback failed for %s", ruling["ruling_number"])
            return None

    data = _strip_nul(data)

    products = data.get("products")
    if not isinstance(products, list):
        logger.warning("No products list in response for %s", ruling["ruling_number"])
        return None

    country = data.get("country")
    if isinstance(country, str):
        country = country.strip() or None

    ruling_number = ruling["ruling_number"]
    ruling_tariffs = set(ruling.get("tariffs") or [])

    rows: list[dict] = []
    for idx, product in enumerate(products):
        if not isinstance(product, dict):
            continue

        description = (product.get("description") or "").strip()
        if len(description) < 15:
            logger.debug(
                "Skipping product %d in %s: description too short (%d chars)",
                idx, ruling_number, len(description),
            )
            continue

        hts_code = product.get("hts_code")
        if hts_code is not None:
            hts_code = str(hts_code).strip()
            if not validate_code(hts_code):
                logger.debug(
                    "Invalid hts_code %r in %s product %d, setting null",
                    hts_code, ruling_number, idx,
                )
                hts_code = None
            elif ruling_tariffs and hts_code:
                # Normalize for comparison: strip dots/spaces
                code_digits = re.sub(r"[\s.\-]", "", hts_code)
                tariff_digits = {re.sub(r"[\s.\-]", "", t) for t in ruling_tariffs}
                # Check if any tariff starts with or matches this code
                matched = any(
                    td.startswith(code_digits[:8]) or code_digits.startswith(td[:8])
                    for td in tariff_digits
                )
                if not matched:
                    logger.info(
                        "HTS code %s from LLM not in ruling tariffs %s for %s",
                        hts_code, ruling_tariffs, ruling_number,
                    )

        row = {
            "ruling_number": ruling_number,
            "product_idx": idx,
            "description": description,
            "hts_code": hts_code,
            "product_use": _clean_optional(product.get("product_use")),
            "materials": _clean_optional(product.get("materials")),
            "reasoning": _clean_optional(product.get("reasoning")),
            "hts_text": _clean_optional(product.get("hts_text")),
            "country": country,
            "tokens_used": tokens_used,
            "raw_response": data,
        }
        rows.append(row)

    if not rows:
        # LLM returned empty products — might be legitimate (admin ruling)
        logger.info("No valid products extracted from %s", ruling_number)
        return None

    return rows


def _clean_optional(value: object) -> str | None:
    """Clean an optional string field — return None if empty/missing."""
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None
