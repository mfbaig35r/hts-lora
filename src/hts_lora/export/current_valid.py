"""Filter enriched extractions to only current-valid HTS codes.

This is a separate step for auditability — we can see exactly which codes
were excluded and why.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from hts_lora.extraction.db import get_connection
from hts_lora.utils.config import ExportConfig
from hts_lora.utils.logging import get_logger

logger = get_logger("export.current_valid")

_VALID_CODES_SQL = {
    "hts8": "SELECT DISTINCT hts8 FROM tariffs",
    "hts6": "SELECT DISTINCT LEFT(hts8, 6) FROM tariffs",
    "hts4": "SELECT DISTINCT LEFT(hts8, 4) FROM tariffs",
}

_MATCH_LENGTHS = {"hts8": 8, "hts6": 6, "hts4": 4}


def _load_valid_codes(match_level: str) -> set[str]:
    """Load the set of valid codes from the tariffs table."""
    sql = _VALID_CODES_SQL.get(match_level)
    if sql is None:
        raise ValueError(f"Unknown match_level: {match_level!r}")

    conn = get_connection()
    cur = conn.execute(sql)
    codes = {row[0] for row in cur.fetchall()}
    logger.info(f"Loaded {len(codes):,} valid {match_level} codes from tariffs table")
    return codes


def filter_current_valid(
    config: ExportConfig,
    input_path: Path,
    output_path: Path,
    excluded_path: Path | None = None,
) -> tuple[int, int]:
    """Filter enriched extractions to current-valid HTS codes.

    Args:
        config: Export configuration.
        input_path: Path to raw enriched extractions JSONL.
        output_path: Path to write valid extractions JSONL.
        excluded_path: Optional path to write excluded records for audit.

    Returns:
        Tuple of (kept_count, excluded_count).
    """
    cv_cfg = config.current_valid
    match_level = cv_cfg.match_level
    match_len = _MATCH_LENGTHS[match_level]

    valid_codes = _load_valid_codes(match_level)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if excluded_path:
        excluded_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    excluded = 0

    out_f = open(output_path, "w")
    exc_f = open(excluded_path, "w") if excluded_path and cv_cfg.log_excluded else None

    try:
        with open(input_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                hts_code = record.get("hts_code") or ""
                digits = re.sub(r"[^0-9]", "", hts_code)
                code_prefix = digits[:match_len]

                if code_prefix in valid_codes:
                    out_f.write(line + "\n")
                    kept += 1
                else:
                    excluded += 1
                    if exc_f is not None:
                        exc_record = {
                            "ruling_number": record.get("ruling_number"),
                            "product_idx": record.get("product_idx"),
                            "hts_code": hts_code,
                            "digits": digits,
                            "match_level": match_level,
                        }
                        exc_f.write(json.dumps(exc_record, ensure_ascii=False) + "\n")
    finally:
        out_f.close()
        if exc_f is not None:
            exc_f.close()

    logger.info(
        f"Current-valid filter ({match_level}): {kept:,} kept, {excluded:,} excluded "
        f"({excluded / max(kept + excluded, 1) * 100:.1f}% excluded)"
    )
    return kept, excluded
