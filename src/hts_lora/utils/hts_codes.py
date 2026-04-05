"""HTS code normalization, formatting, matching, and validation utilities.

Internal representation: digit-only strings (e.g., "0101210010").
Display format: XXXX.XX.XXXX (e.g., "0101.21.0010").
"""

from __future__ import annotations

import re

_DIGITS_ONLY = re.compile(r"^\d+$")


def normalize_code(code: str) -> str:
    """Strip dots, dashes, and whitespace from an HTS code.

    Returns digit-only string. Pads to 10 digits if between 4-9 digits.
    """
    cleaned = re.sub(r"[\s.\-]", "", code.strip())
    if not _DIGITS_ONLY.match(cleaned):
        raise ValueError(f"Invalid HTS code after cleaning: {code!r} -> {cleaned!r}")
    if len(cleaned) < 4:
        raise ValueError(f"HTS code too short (min 4 digits): {code!r}")
    if len(cleaned) <= 10:
        cleaned = cleaned.ljust(10, "0")
    return cleaned


def format_code(code: str) -> str:
    """Format a digit-only HTS code as XXXX.XX.XXXX for display."""
    digits = normalize_code(code)
    return f"{digits[:4]}.{digits[4:6]}.{digits[6:10]}"


def validate_code(code: str) -> bool:
    """Check if a string is a valid HTS code (4-10 digits after cleaning)."""
    try:
        cleaned = re.sub(r"[\s.\-]", "", code.strip())
        return bool(_DIGITS_ONLY.match(cleaned)) and 4 <= len(cleaned) <= 10
    except Exception:
        return False


def chapter(code: str) -> str:
    """Extract 2-digit chapter from an HTS code."""
    return normalize_code(code)[:2]


def heading(code: str) -> str:
    """Extract 4-digit heading from an HTS code."""
    return normalize_code(code)[:4]


def subheading(code: str) -> str:
    """Extract 6-digit subheading from an HTS code."""
    return normalize_code(code)[:6]


def hierarchy_path(code: str) -> dict[str, str]:
    """Return the full hierarchy for an HTS code.

    Returns dict with chapter, heading, subheading, and full code.
    """
    digits = normalize_code(code)
    return {
        "chapter": digits[:2],
        "heading": digits[:4],
        "subheading": digits[:6],
        "full": digits,
    }


def match_at_level(predicted: str, actual: str, level: str) -> bool:
    """Check if two codes match at a given hierarchy level.

    Level must be one of: chapter, heading, subheading, exact.
    """
    p = normalize_code(predicted)
    a = normalize_code(actual)
    extractors = {
        "chapter": lambda c: c[:2],
        "heading": lambda c: c[:4],
        "subheading": lambda c: c[:6],
        "exact": lambda c: c,
    }
    if level not in extractors:
        raise ValueError(f"Unknown level: {level!r}. Must be one of {list(extractors)}")
    return extractors[level](p) == extractors[level](a)
