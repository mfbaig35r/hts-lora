"""Export glossary terms from the database.

The hts_glossary table stores terms with a JSONB `senses` array,
each sense having a definition, category, and chapter list.
We flatten each sense into a separate JSONL record for the training pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

from hts_lora.extraction.db import get_connection
from hts_lora.utils.config import ExportConfig
from hts_lora.utils.logging import get_logger

logger = get_logger("export.glossary")

_GLOSSARY_SQL = """\
SELECT term, display_term, senses, source_count
FROM hts_glossary
WHERE jsonb_array_length(senses) >= %(min_senses)s
ORDER BY term
"""


def export_glossary(config: ExportConfig, output_path: Path) -> int:
    """Export glossary terms to JSONL, flattening senses.

    Each sense becomes a separate record with: term, definition, category, source.
    Returns the number of records written.
    """
    conn = get_connection()
    glos_cfg = config.glossary

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cur = conn.execute(_GLOSSARY_SQL, {"min_senses": glos_cfg.min_senses})
    columns = [desc.name for desc in cur.description]

    count = 0
    with open(output_path, "w") as f:
        for row in cur.fetchall():
            record = dict(zip(columns, row))
            term = record["term"]
            senses = record["senses"]

            # senses is a JSONB array of dicts
            if isinstance(senses, str):
                senses = json.loads(senses)

            for sense in senses:
                entry = {
                    "term": term,
                    "definition": sense.get("definition", ""),
                    "category": sense.get("category", ""),
                    "source": f"hts_glossary (source_count={record.get('source_count', 0)})",
                    "senses": len(senses),
                }
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
                count += 1

    logger.info(f"Exported {count:,} glossary entries to {output_path}")
    return count
