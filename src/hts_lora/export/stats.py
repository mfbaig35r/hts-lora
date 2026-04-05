"""Export audit statistics and utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ExportStats:
    """Tracks counts and metadata across the export pipeline."""

    # Extraction stats
    total_extractions: int = 0
    filtered_extractions: int = 0

    # Current-valid filter
    valid_codes: int = 0
    excluded_codes: int = 0

    # Glossary
    glossary_terms: int = 0

    # Enrichments
    enrichments_exported: int = 0

    # Timing
    started_at: str = ""
    completed_at: str = ""
    step_times: dict[str, float] = field(default_factory=dict)

    def start(self) -> None:
        self.started_at = datetime.now(timezone.utc).isoformat()

    def complete(self) -> None:
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_extractions": self.total_extractions,
            "filtered_extractions": self.filtered_extractions,
            "valid_codes": self.valid_codes,
            "excluded_codes": self.excluded_codes,
            "glossary_terms": self.glossary_terms,
            "enrichments_exported": self.enrichments_exported,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "step_times": self.step_times,
        }
