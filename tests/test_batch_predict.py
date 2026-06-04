"""Tests for batch_predict prompt building.

These exist primarily as a regression guard against the bug where
_predict_batch hard-required record["description"] but the v2 data
pipeline writes pre-built `messages` instead, causing KeyError on
every eval against data/formatted/test.jsonl.
"""

from __future__ import annotations

import pytest

from hts_lora.inference.batch_predict import build_messages_for_record


SYSTEM_PROMPT = "detailed thinking off\n\nYou are an expert..."


class TestPreBuiltMessages:
    """Records that ship with their own `messages` should be used as-is."""

    def test_uses_pre_built_messages_without_description(self):
        # Mirrors the shape of records in data/formatted/test.jsonl:
        # `messages` is pre-built, no `description` key at all.
        record = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Materials: silk\nUse: scarf"},
                {"role": "assistant", "content": "Chapter 62..."},
            ],
            "hts_code": "6214900090",
            "abstain": False,
            "input_variant": "materials_only",
        }
        messages = build_messages_for_record(record)
        assert len(messages) == 2  # system + user only, assistant stripped
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "silk" in messages[1]["content"]

    def test_returns_first_two_when_only_two_present(self):
        record = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Product: copper wire"},
            ],
            "hts_code": "8544.30.0000",
        }
        messages = build_messages_for_record(record)
        assert len(messages) == 2

    def test_atlas_record_shape(self):
        # The shape produced by scripts/build_atlas_eval.py.
        record = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Product: brass pipe seamless"},
            ],
            "hts_code": ["7411.21.1000", "7411.21.5000"],
            "abstain": False,
            "task_type": "atlas_classify",
            "input_variant": "minimal",
            "description": "brass pipe seamless",
            "source": "atlas_test",
        }
        messages = build_messages_for_record(record)
        assert len(messages) == 2
        # Pre-built path wins even when description is also present.
        assert messages[1]["content"] == "Product: brass pipe seamless"


class TestRawFieldsFallback:
    """Records without `messages` should be built from raw fields."""

    def test_builds_from_description_when_no_messages(self):
        record = {
            "description": "insulated copper electrical wire, 12 AWG",
            "materials": "copper",
            "country": "Mexico",
        }
        messages = build_messages_for_record(record, default_variant="rich")
        assert len(messages) == 2
        assert "insulated copper" in messages[1]["content"]
        assert "Mexico" in messages[1]["content"]

    def test_empty_messages_list_falls_back_to_raw_fields(self):
        # `messages: []` should be treated as absent (the v1 data pipeline
        # could in principle emit an empty list, and we want to fall back
        # rather than produce zero-message prompts).
        record = {
            "messages": [],
            "description": "fresh cut roses",
        }
        messages = build_messages_for_record(record)
        assert len(messages) == 2
        assert "fresh cut roses" in messages[1]["content"]

    def test_raises_keyerror_only_when_both_missing(self):
        # If neither `messages` nor `description` is present, the raw-fields
        # path tries to read record["description"] and KeyErrors. This is
        # the original (correct) failure mode for genuinely malformed input;
        # we just want to make sure pre-built messages bypass it.
        with pytest.raises(KeyError):
            build_messages_for_record({})
