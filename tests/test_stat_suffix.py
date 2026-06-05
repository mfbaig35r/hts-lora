"""Tests for Layer A: stat-suffix validation and completion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hts_lora.inference.parse_output import ParsedPrediction
from hts_lora.postprocess.stat_suffix import (
    StatSuffixIndex,
    validate_and_complete,
)


@pytest.fixture
def tiny_index(tmp_path: Path) -> StatSuffixIndex:
    """A 3-subheading toy index just big enough to exercise all branches."""
    payload = {
        "valid_10digit": {
            # 8544.30.00 subheading has 2 children
            "8544300010": 142,
            "8544300090": 89,
            # 6110.30.00 subheading has 1 child (median case)
            "6110300010": 50,
            # 0101.21.00 subheading has 3 children
            "0101210010": 200,
            "0101210020": 30,
            "0101210090": 5,
        },
        "completions_by_8digit": {
            "85443000": ["8544300010", "8544300090"],   # sorted by freq desc
            "61103000": ["6110300010"],
            "01012100": ["0101210010", "0101210020", "0101210090"],
        },
        "_meta": {"source": "test"},
    }
    p = tmp_path / "index.json"
    p.write_text(json.dumps(payload))
    return StatSuffixIndex.load(p)


def _pred(hts_code: str | None, **kw) -> ParsedPrediction:
    return ParsedPrediction(
        hts_code=hts_code,
        chapter_code=kw.get("chapter_code"),
        heading_code=kw.get("heading_code"),
        subheading_code=kw.get("subheading_code"),
        is_abstention=kw.get("is_abstention", False),
        parse_ok=kw.get("parse_ok", True),
    )


class TestValid:
    def test_valid_code_passes_through_unchanged(self, tiny_index):
        pred = _pred("8544.30.0010")
        out = validate_and_complete(pred, tiny_index)
        assert out.changed is False
        assert out.reason == "valid"
        assert out.prediction.hts_code == "8544.30.0010"


class TestCompletion:
    def test_unrecognized_code_completes_to_most_common(self, tiny_index):
        # 8544.30.0099 has 8-digit prefix 85443000 (in the index). The
        # 10-digit code is not valid, so Layer A picks the most-common
        # completion: 8544300010 (freq 142 vs 89).
        pred = _pred("8544.30.0099")
        out = validate_and_complete(pred, tiny_index)
        assert out.changed is True
        assert out.reason == "completed"
        assert out.prediction.hts_code == "8544.30.0010"
        assert out.original_hts_code == "8544.30.0099"

    def test_singleton_subheading_always_completes_to_only_child(self, tiny_index):
        # 6110.30.00 has one child. Any invalid 10-digit completion
        # under that 8-digit prefix collapses to that single child.
        pred = _pred("6110.30.0099")
        out = validate_and_complete(pred, tiny_index)
        assert out.changed is True
        assert out.prediction.hts_code == "6110.30.0010"

    def test_completion_uses_frequency_ordering(self, tiny_index):
        # 0101.21.00 has three children. 0010 has freq 200 (highest),
        # 0020 has 30, 0090 has 5. Most-common picker picks 0010.
        pred = _pred("0101.21.0077")
        out = validate_and_complete(pred, tiny_index)
        assert out.prediction.hts_code == "0101.21.0010"

    def test_8digit_level_error_leaves_code_alone(self, tiny_index):
        # 8544.30.9999 has 8-digit prefix 85443099, which is NOT in the
        # index. This is an 8-digit-level error (positions 6-7 differ),
        # not a stat-suffix error. Layer A cannot anchor a correction.
        pred = _pred("8544.30.9999")
        out = validate_and_complete(pred, tiny_index)
        assert out.changed is False
        assert out.reason == "no_8digit_match"


class TestBypass:
    def test_abstention_is_not_touched(self, tiny_index):
        pred = ParsedPrediction(
            hts_code=None,
            is_abstention=True,
            abstention_reason="too vague",
            parse_ok=True,
        )
        out = validate_and_complete(pred, tiny_index)
        assert out.changed is False
        assert out.reason == "skipped_abstain"

    def test_missing_code_is_not_touched(self, tiny_index):
        pred = _pred(None)
        out = validate_and_complete(pred, tiny_index)
        assert out.changed is False
        assert out.reason == "skipped_no_code"

    def test_unknown_8digit_subheading_leaves_code_alone(self, tiny_index):
        # 9999.99.0001 has no entry in completions_by_8digit, so Layer A
        # can't anchor a correction.
        pred = _pred("9999.99.0001")
        out = validate_and_complete(pred, tiny_index)
        assert out.changed is False
        assert out.reason == "no_8digit_match"
        assert out.prediction.hts_code == "9999.99.0001"


class TestNormalization:
    def test_undotted_code_is_accepted(self, tiny_index):
        # Raw 10-char code without dots should also validate.
        pred = _pred("8544300010")
        out = validate_and_complete(pred, tiny_index)
        assert out.reason == "valid"

    def test_short_code_is_padded(self, tiny_index):
        # An 8-char code becomes 8544300000 after padding. That's not in
        # the index, but 8544.30.00 IS, so it should complete.
        pred = _pred("85443000")
        out = validate_and_complete(pred, tiny_index)
        assert out.changed is True
        assert out.reason == "completed"

    def test_dotted_8digit_completes(self, tiny_index):
        pred = _pred("8544.30.00")
        out = validate_and_complete(pred, tiny_index)
        assert out.changed is True
        assert out.prediction.hts_code == "8544.30.0010"


class TestPreservation:
    def test_other_fields_are_preserved(self, tiny_index):
        pred = _pred(
            "8544.30.9999",
            chapter_code="85",
            heading_code="85.44",
            subheading_code="8544.30",
        )
        out = validate_and_complete(pred, tiny_index)
        assert out.prediction.chapter_code == "85"
        assert out.prediction.heading_code == "85.44"
        assert out.prediction.subheading_code == "8544.30"
