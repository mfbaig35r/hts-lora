"""Tests for the export pipeline modules."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hts_lora.export.current_valid import filter_current_valid
from hts_lora.export.extractions import _passes_filters
from hts_lora.export.models import EnrichedExtraction, GlossaryEntry, HTS6Enrichment
from hts_lora.export.stats import ExportStats
from hts_lora.utils.config import ExportConfig, ExportExtractionsConfig


# ── Model Tests ─────────────────────────────────────────────────────────────


class TestEnrichedExtraction:
    def test_minimal_fields(self):
        ex = EnrichedExtraction(
            ruling_number="N123456",
            product_idx=0,
            description="Copper wire insulated",
        )
        assert ex.ruling_number == "N123456"
        assert ex.hts_code is None
        assert ex.chapter_code is None

    def test_full_fields(self):
        ex = EnrichedExtraction(
            ruling_number="N123456",
            product_idx=1,
            description="Copper wire insulated",
            hts_code="8544.30.00",
            reasoning="Classified under Chapter 85",
            chapter_code="85",
            chapter_description="Electrical machinery",
            heading_code="8544",
            heading_description="Insulated wire",
            tariff_description="Wire of copper",
        )
        assert ex.hts_code == "8544.30.00"
        assert ex.chapter_code == "85"

    def test_serialization_roundtrip(self):
        ex = EnrichedExtraction(
            ruling_number="N999",
            product_idx=0,
            description="Test product",
            hts_code="0101.21.00",
        )
        data = ex.model_dump()
        restored = EnrichedExtraction(**data)
        assert restored == ex


class TestGlossaryEntry:
    def test_basic(self):
        entry = GlossaryEntry(term="alloy steel", definition="Steel containing...")
        assert entry.term == "alloy steel"
        assert entry.senses == 1


class TestHTS6Enrichment:
    def test_defaults(self):
        e = HTS6Enrichment(hts6="854430", enriched_description="Insulated copper wire")
        assert e.keywords == []
        assert e.exclusionary_terms == []


# ── Extraction Filter Tests ─────────────────────────────────────────────────


class TestPassesFilters:
    def test_valid_record(self):
        cfg = ExportExtractionsConfig()
        record = {
            "hts_code": "8544.30.0000",
            "reasoning": "Classified under Chapter 85 for insulated wire products.",
        }
        assert _passes_filters(record, cfg) is True

    def test_short_reasoning_rejected(self):
        cfg = ExportExtractionsConfig(min_reasoning_length=20)
        record = {"hts_code": "8544.30.0000", "reasoning": "Short."}
        assert _passes_filters(record, cfg) is False

    def test_long_reasoning_rejected(self):
        cfg = ExportExtractionsConfig(max_reasoning_length=50)
        record = {"hts_code": "8544.30.0000", "reasoning": "x" * 100}
        assert _passes_filters(record, cfg) is False

    def test_missing_hts_code_rejected(self):
        cfg = ExportExtractionsConfig()
        record = {"hts_code": "", "reasoning": "A valid reasoning string here"}
        assert _passes_filters(record, cfg) is False

    def test_no_reasoning_required(self):
        cfg = ExportExtractionsConfig(require_reasoning=False)
        record = {"hts_code": "8544.30.0000", "reasoning": ""}
        assert _passes_filters(record, cfg) is True

    def test_no_hts_required(self):
        cfg = ExportExtractionsConfig(require_hts_code=False)
        record = {"hts_code": "", "reasoning": "Some valid reasoning text here."}
        assert _passes_filters(record, cfg) is True


# ── Current-Valid Filter Tests ──────────────────────────────────────────────


class TestFilterCurrentValid:
    @pytest.fixture
    def sample_extractions(self, tmp_path: Path) -> Path:
        """Create a sample extractions JSONL with known codes."""
        records = [
            {"ruling_number": "N001", "product_idx": 0, "hts_code": "8544.30.0000",
             "description": "Valid copper wire"},
            {"ruling_number": "N002", "product_idx": 0, "hts_code": "9999.99.9999",
             "description": "Invalid code product"},
            {"ruling_number": "N003", "product_idx": 0, "hts_code": "0101.21.0010",
             "description": "Valid horse"},
        ]
        path = tmp_path / "input.jsonl"
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return path

    @patch("hts_lora.export.current_valid.get_connection")
    def test_filters_invalid_codes(self, mock_conn, sample_extractions, tmp_path):
        # Mock the valid codes query
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [("85443000",), ("01012100",)]
        mock_conn.return_value.execute.return_value = mock_cursor

        config = ExportConfig()
        output = tmp_path / "valid.jsonl"
        excluded = tmp_path / "excluded.jsonl"

        kept, excl_count = filter_current_valid(config, sample_extractions, output, excluded)

        assert kept == 2
        assert excl_count == 1

        # Verify output contents
        with open(output) as f:
            valid_records = [json.loads(line) for line in f]
        assert len(valid_records) == 2
        codes = {r["hts_code"] for r in valid_records}
        assert "9999.99.9999" not in codes

        # Verify excluded log
        with open(excluded) as f:
            exc_records = [json.loads(line) for line in f]
        assert len(exc_records) == 1
        assert exc_records[0]["hts_code"] == "9999.99.9999"

    @patch("hts_lora.export.current_valid.get_connection")
    def test_all_valid(self, mock_conn, tmp_path):
        records = [
            {"ruling_number": "N001", "product_idx": 0, "hts_code": "8544.30.0000"},
        ]
        input_path = tmp_path / "input.jsonl"
        with open(input_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [("85443000",)]
        mock_conn.return_value.execute.return_value = mock_cursor

        config = ExportConfig()
        output = tmp_path / "valid.jsonl"

        kept, excl = filter_current_valid(config, input_path, output)
        assert kept == 1
        assert excl == 0


# ── Stats Tests ─────────────────────────────────────────────────────────────


class TestExportStats:
    def test_to_dict(self):
        stats = ExportStats(total_extractions=100, valid_codes=80, excluded_codes=20)
        d = stats.to_dict()
        assert d["total_extractions"] == 100
        assert d["valid_codes"] == 80
        assert d["excluded_codes"] == 20

    def test_start_complete(self):
        stats = ExportStats()
        stats.start()
        assert stats.started_at != ""
        stats.complete()
        assert stats.completed_at != ""
