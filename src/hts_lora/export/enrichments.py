"""Export HTS6 AI enrichments from the database."""

from __future__ import annotations

import json
from pathlib import Path

from hts_lora.extraction.db import get_connection
from hts_lora.utils.config import ExportConfig
from hts_lora.utils.logging import get_logger

logger = get_logger("export.enrichments")

_ENRICHMENTS_SQL = """\
SELECT hts6, enriched_description, keywords, exclusionary_terms, common_attributes
FROM hts6_enrichments
WHERE length(enriched_description) >= %(min_desc_length)s
ORDER BY hts6
"""


def export_enrichments(config: ExportConfig, output_path: Path) -> int:
    """Export HTS6 enrichments to JSONL.

    Returns the number of enrichments written.
    """
    conn = get_connection()
    enr_cfg = config.enrichments

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cur = conn.execute(
        _ENRICHMENTS_SQL, {"min_desc_length": enr_cfg.min_description_length}
    )
    columns = [desc.name for desc in cur.description]

    count = 0
    with open(output_path, "w") as f:
        for row in cur.fetchall():
            record = dict(zip(columns, row))
            # Ensure list fields are properly serialized
            for key in ("keywords", "exclusionary_terms", "common_attributes"):
                val = record.get(key)
                if val is None:
                    record[key] = []
                elif isinstance(val, str):
                    try:
                        record[key] = json.loads(val)
                    except json.JSONDecodeError:
                        record[key] = []
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            count += 1

    logger.info(f"Exported {count:,} HTS6 enrichments to {output_path}")
    return count
