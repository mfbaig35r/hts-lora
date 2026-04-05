"""Pydantic models for export pipeline data."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field


class EnrichedExtraction(BaseModel):
    """A CROSS ruling extraction enriched with HTS hierarchy context."""

    # Extraction fields
    ruling_number: str
    product_idx: int
    description: str
    hts_code: str | None = None
    reasoning: str | None = None
    hts_text: str | None = None
    materials: str | None = None
    product_use: str | None = None
    country: str | None = None
    extracted_at: str | None = None

    # Ruling metadata
    subject: str | None = None
    ruling_date: str | None = None
    ruling_url: str | None = None
    is_revoked: bool = False

    # Hierarchy context
    chapter_code: str | None = None
    chapter_description: str | None = None
    heading_code: str | None = None
    heading_description: str | None = None
    tariff_description: str | None = None


class GlossaryEntry(BaseModel):
    """A glossary term with its definition and metadata."""

    term: str
    definition: str
    source: str | None = None
    senses: int = 1


class HTS6Enrichment(BaseModel):
    """AI-generated enrichment for a 6-digit HTS code group."""

    hts6: str
    enriched_description: str
    keywords: list[str] = Field(default_factory=list)
    exclusionary_terms: list[str] = Field(default_factory=list)
    common_attributes: list[str] = Field(default_factory=list)
