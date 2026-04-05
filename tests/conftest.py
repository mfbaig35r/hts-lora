"""Shared test fixtures."""

from __future__ import annotations

import random

import pytest

from hts_lora.data.build_examples import TrainingExample
from hts_lora.data.ingest import RawExample
from hts_lora.utils.config import DataConfig


@pytest.fixture
def sample_raw_examples() -> list[RawExample]:
    """A small set of raw examples for testing, with enrichment fields."""
    return [
        RawExample(
            description="Live horses for breeding",
            hts_code="0101.21.0010",
            source="test",
            ruling_number="N100001",
            reasoning="Classified under Chapter 1 for live animals.",
            chapter_code="01",
            chapter_description="Live animals",
            heading_code="0101",
            heading_description="Live horses, asses, mules and hinnies",
        ),
        RawExample(
            description="Frozen beef cuts boneless",
            hts_code="0202.30.0000",
            source="test",
            ruling_number="N100002",
            reasoning="Classified under Chapter 2 for meat.",
            materials="beef",
            chapter_code="02",
            chapter_description="Meat and edible meat offal",
            heading_code="0202",
            heading_description="Meat of bovine animals, frozen",
        ),
        RawExample(
            description="Fresh cut roses",
            hts_code="0603.11.0030",
            source="test",
            ruling_number="N100003",
            chapter_code="06",
            chapter_description="Live trees and other plants",
        ),
        RawExample(
            description="Raw cane sugar",
            hts_code="1701.13.0010",
            source="test",
            ruling_number="N100004",
            materials="sugar cane",
            chapter_code="17",
        ),
        RawExample(
            description="Copper wire insulated",
            hts_code="8544.30.0000",
            source="test",
            ruling_number="N100005",
            reasoning="Classified under Chapter 85 for insulated wire.",
            materials="copper",
            product_use="electrical wiring",
            chapter_code="85",
            chapter_description="Electrical machinery and equipment",
            heading_code="8544",
            heading_description="Insulated wire, cable",
            tariff_description="Insulated winding wire",
        ),
        RawExample(description="Cotton T-shirts men", hts_code="6109.10.0012", source="test"),
        RawExample(description="Wooden furniture chairs", hts_code="9401.61.4011", source="test"),
        RawExample(description="Plastic bottles PET", hts_code="3923.30.0090", source="test"),
        RawExample(description="Steel bolts hex head", hts_code="7318.15.2065", source="test"),
        RawExample(description="Ceramic tiles glazed", hts_code="6908.90.0010", source="test"),
    ]


@pytest.fixture
def sample_training_examples() -> list[TrainingExample]:
    """Pre-built training examples for testing."""
    return [
        TrainingExample(
            description="Live horses for breeding",
            hts_code="0101210010",
            task_type="hierarchical_classify",
            input_variant="rich",
            ruling_number="N100001",
            reasoning="Classified under Chapter 1 for live animals.",
            chapter_code="01",
            chapter_description="Live animals",
            heading_code="0101",
            heading_description="Live horses, asses, mules and hinnies",
        ),
        TrainingExample(
            description="Copper wire insulated",
            hts_code="8544300000",
            task_type="hierarchical_classify",
            input_variant="minimal",
            ruling_number="N100005",
            chapter_code="85",
            chapter_description="Electrical machinery and equipment",
            heading_code="8544",
            heading_description="Insulated wire, cable",
            tariff_description="Insulated winding wire",
        ),
        TrainingExample(
            description="Fresh cut roses",
            hts_code="0603110030",
            task_type="hierarchical_classify",
            input_variant="glossary_enriched",
            glossary_terms=[{"term": "roses", "definition": "Cut flowers of genus Rosa"}],
        ),
        TrainingExample(
            description="Ambiguous product",
            hts_code="__ABSTAIN__",
            task_type="abstention",
            abstain=True,
            abstain_category="vague_description",
        ),
    ]


@pytest.fixture
def data_config() -> DataConfig:
    return DataConfig()


@pytest.fixture
def rng() -> random.Random:
    return random.Random(42)
