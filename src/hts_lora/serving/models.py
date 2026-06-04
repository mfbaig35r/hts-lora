"""Pydantic request/response models for the HTS classify API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    """Request body for POST /classify."""

    description: str = Field(..., description="Product description (required)")
    materials: str | None = Field(None, description="Materials of construction")
    use: str | None = Field(None, description="Intended product use")
    country_of_origin: str | None = Field(None, description="Country of origin")
    max_tokens: int = Field(512, ge=16, le=2048, description="Max output tokens")


class HierarchyLevel(BaseModel):
    """A single level of the HTS hierarchy."""

    code: str
    description: str | None = None


class ClassificationResponse(BaseModel):
    """Response body for POST /classify."""

    hts_code: str | None = None
    chapter: HierarchyLevel | None = None
    heading: HierarchyLevel | None = None
    subheading: HierarchyLevel | None = None
    reasoning: str | None = None
    provides_for: str | None = None
    is_abstention: bool = False
    abstention_reason: str | None = None
    parse_ok: bool = False
    raw: str | None = None
    model: str
    latency_ms: int


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str
    upstream: str
    upstream_reachable: bool
    model: str
