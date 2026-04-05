"""Export enriched CROSS ruling extractions from the database.

Joins extractions with HTS hierarchy (chapters, headings, tariffs) and
streams results to JSONL in batches to avoid memory issues with ~325k rows.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from hts_lora.extraction.db import get_connection
from hts_lora.export.models import EnrichedExtraction
from hts_lora.utils.config import ExportConfig
from hts_lora.utils.logging import get_logger

logger = get_logger("export.extractions")

# Uses LIMIT/OFFSET batching (Supabase doesn't handle long-lived server-side
# cursors well). ORDER BY is deterministic so pagination is safe.
_ENRICHED_SQL = """\
SELECT
    e.ruling_number, e.product_idx, e.description, e.hts_code,
    e.reasoning, e.hts_text, e.materials, e.product_use, e.country, e.extracted_at,
    cr.subject, cr.ruling_date, cr.ruling_url, cr.is_revoked,
    c.chapter AS chapter_code, c.description AS chapter_description,
    h.hts4 AS heading_code, h.description AS heading_description,
    t.brief_description AS tariff_description
FROM cross_ruling_extractions e
JOIN cross_rulings cr ON e.ruling_number = cr.ruling_number
LEFT JOIN hts_chapters c
    ON c.chapter = lpad(substring(regexp_replace(e.hts_code, '[^0-9]', '', 'g') FROM 1 FOR 2), 2, '0')
LEFT JOIN hts_headings h
    ON h.hts4 = substring(regexp_replace(e.hts_code, '[^0-9]', '', 'g') FROM 1 FOR 4)
LEFT JOIN tariffs t
    ON t.hts8 = substring(regexp_replace(e.hts_code, '[^0-9]', '', 'g') FROM 1 FOR 8)
WHERE e.description != '__EXTRACTION_FAILED__'
  AND e.hts_code IS NOT NULL
  AND length(e.description) >= %(min_desc_length)s
ORDER BY e.ruling_number, e.product_idx
LIMIT %(limit)s OFFSET %(offset)s
"""


def export_extractions(config: ExportConfig, output_path: Path) -> int:
    """Export enriched extractions to JSONL, using LIMIT/OFFSET batching.

    Returns the number of records written.
    """
    conn = get_connection()
    batch_size = config.database.batch_size
    ext_cfg = config.extractions

    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    offset = 0

    with open(output_path, "w") as f:
        while True:
            cur = conn.execute(
                _ENRICHED_SQL,
                {
                    "min_desc_length": ext_cfg.min_description_length,
                    "limit": batch_size,
                    "offset": offset,
                },
            )
            columns = [desc.name for desc in cur.description]
            rows = cur.fetchall()

            if not rows:
                break

            for row in rows:
                record = dict(zip(columns, row))
                if not _passes_filters(record, ext_cfg):
                    continue
                for key in ("ruling_date", "extracted_at"):
                    if record.get(key) is not None:
                        record[key] = str(record[key])
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                count += 1

            offset += len(rows)

            if offset % 50_000 == 0:
                logger.info(f"Exported {count:,} extractions ({offset:,} rows scanned)...")

            # If we got fewer rows than the batch size, we're done
            if len(rows) < batch_size:
                break

    logger.info(f"Exported {count:,} enriched extractions to {output_path}")
    return count


def _passes_filters(record: dict, cfg) -> bool:
    """Apply extraction-level quality filters."""
    if cfg.require_reasoning:
        reasoning = record.get("reasoning") or ""
        if len(reasoning) < cfg.min_reasoning_length:
            return False
        if len(reasoning) > cfg.max_reasoning_length:
            return False

    if cfg.require_hts_code:
        hts_code = record.get("hts_code") or ""
        digits = re.sub(r"[^0-9]", "", hts_code)
        if len(digits) < 4:
            return False

    return True
