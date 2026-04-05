"""Load raw data sources into a unified RawExample format.

Supports enriched CROSS ruling exports with hierarchy context.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from hts_lora.utils.config import DataConfig, SourceConfig
from hts_lora.utils.logging import get_logger

logger = get_logger("data.ingest")


class RawExample(BaseModel):
    """A single raw training example before normalization.

    Includes optional enrichment fields from the export pipeline.
    """

    description: str
    hts_code: str
    source: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Enrichment fields (from export pipeline)
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

    # Ruling metadata
    subject: str | None = None
    ruling_date: str | None = None


# Fields to extract from JSONL rows into enrichment fields on RawExample
_ENRICHMENT_FIELDS = {
    "ruling_number", "reasoning", "hts_text", "materials", "product_use",
    "country", "chapter_code", "chapter_description", "heading_code",
    "heading_description", "tariff_description", "subject", "ruling_date",
}


def _load_csv(path: Path, text_field: str, code_field: str) -> list[dict[str, str]]:
    records = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get(text_field, "").strip()
            code = row.get(code_field, "").strip()
            if text and code:
                records.append({"description": text, "hts_code": code, "row": row})
    return records


def _load_jsonl(path: Path, text_field: str, code_field: str) -> list[dict[str, str]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = str(obj.get(text_field, "")).strip()
            code = str(obj.get(code_field, "")).strip()
            if text and code:
                records.append({"description": text, "hts_code": code, "row": obj})
    return records


def _load_json(path: Path, text_field: str, code_field: str) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("data", data.get("records", data.get("items", [])))
    records = []
    for obj in data:
        text = str(obj.get(text_field, "")).strip()
        code = str(obj.get(code_field, "")).strip()
        if text and code:
            records.append({"description": text, "hts_code": code, "row": obj})
    return records


_LOADERS = {
    "csv": _load_csv,
    "jsonl": _load_jsonl,
    "json": _load_json,
}


def load_source(source: SourceConfig) -> list[RawExample]:
    """Load a single data source into RawExample objects."""
    path = Path(source.path)
    if not path.exists():
        logger.warning(f"Source file not found, skipping: {path}")
        return []

    loader = _LOADERS.get(source.format)
    if loader is None:
        raise ValueError(f"Unsupported format: {source.format}")

    raw_records = loader(path, source.text_field, source.code_field)
    examples = []
    for rec in raw_records:
        row_data = rec.pop("row", {})

        # Extract enrichment fields from the row data
        enrichment_kwargs: dict[str, Any] = {}
        remaining_meta: dict[str, Any] = {}
        if isinstance(row_data, dict):
            for k, v in row_data.items():
                if k in _ENRICHMENT_FIELDS:
                    enrichment_kwargs[k] = v
                elif k not in ("description", "hts_code"):
                    remaining_meta[k] = v

        examples.append(
            RawExample(
                description=rec["description"],
                hts_code=rec["hts_code"],
                source=str(path.name),
                metadata=remaining_meta,
                **enrichment_kwargs,
            )
        )
    logger.info(f"Loaded {len(examples)} examples from {path.name}")
    return examples


def ingest(config: DataConfig) -> list[RawExample]:
    """Load all configured sources and merge into a single list."""
    all_examples: list[RawExample] = []
    for source in config.sources:
        all_examples.extend(load_source(source))
    logger.info(f"Total ingested: {len(all_examples)} examples from {len(config.sources)} sources")
    return all_examples
