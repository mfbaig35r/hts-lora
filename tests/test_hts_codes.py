"""Tests for HTS code normalization, formatting, and matching."""

import pytest

from hts_lora.utils.hts_codes import (
    chapter,
    format_code,
    heading,
    hierarchy_path,
    match_at_level,
    normalize_code,
    subheading,
    validate_code,
)


class TestNormalizeCode:
    def test_strips_dots(self):
        assert normalize_code("0101.21.0010") == "0101210010"

    def test_strips_dashes(self):
        assert normalize_code("0101-21-0010") == "0101210010"

    def test_pads_short_codes(self):
        assert normalize_code("0101") == "0101000000"
        assert normalize_code("010121") == "0101210000"

    def test_preserves_10_digit(self):
        assert normalize_code("0101210010") == "0101210010"

    def test_strips_whitespace(self):
        assert normalize_code(" 0101.21.0010 ") == "0101210010"

    def test_rejects_non_digits(self):
        with pytest.raises(ValueError, match="Invalid HTS code"):
            normalize_code("01AB.21.0010")

    def test_rejects_too_short(self):
        with pytest.raises(ValueError, match="too short"):
            normalize_code("01")


class TestFormatCode:
    def test_formats_10_digit(self):
        assert format_code("0101210010") == "0101.21.0010"

    def test_formats_with_dots(self):
        assert format_code("0101.21.0010") == "0101.21.0010"

    def test_formats_short_code(self):
        assert format_code("0101") == "0101.00.0000"


class TestValidateCode:
    def test_valid_codes(self):
        assert validate_code("0101.21.0010") is True
        assert validate_code("0101210010") is True
        assert validate_code("0101") is True

    def test_invalid_codes(self):
        assert validate_code("01") is False
        assert validate_code("ABCD") is False
        assert validate_code("") is False


class TestHierarchy:
    def test_chapter(self):
        assert chapter("0101210010") == "01"
        assert chapter("8544.30.0000") == "85"

    def test_heading(self):
        assert heading("0101210010") == "0101"
        assert heading("8544.30.0000") == "8544"

    def test_subheading(self):
        assert subheading("0101210010") == "010121"
        assert subheading("8544.30.0000") == "854430"

    def test_hierarchy_path(self):
        h = hierarchy_path("8544.30.0000")
        assert h == {
            "chapter": "85",
            "heading": "8544",
            "subheading": "854430",
            "full": "8544300000",
        }


class TestMatchAtLevel:
    def test_exact_match(self):
        assert match_at_level("0101210010", "0101.21.0010", "exact") is True
        assert match_at_level("0101210010", "0101210020", "exact") is False

    def test_chapter_match(self):
        assert match_at_level("0101210010", "0102290000", "chapter") is True
        assert match_at_level("0101210010", "0202300000", "chapter") is False

    def test_heading_match(self):
        assert match_at_level("0101210010", "0101290000", "heading") is True
        assert match_at_level("0101210010", "0102210000", "heading") is False

    def test_subheading_match(self):
        assert match_at_level("0101210010", "0101210020", "subheading") is True
        assert match_at_level("0101210010", "0101290000", "subheading") is False

    def test_invalid_level(self):
        with pytest.raises(ValueError, match="Unknown level"):
            match_at_level("0101", "0101", "invalid")
