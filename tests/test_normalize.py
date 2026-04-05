"""Tests for data normalization: dedup, length filters, validation."""

from hts_lora.data.ingest import RawExample
from hts_lora.data.normalize import normalize_examples
from hts_lora.utils.config import NormalizationConfig


def _make_example(desc: str, code: str = "0101.21.0010") -> RawExample:
    return RawExample(description=desc, hts_code=code, source="test")


class TestNormalize:
    def test_removes_invalid_codes(self):
        examples = [
            _make_example("Valid product", "0101.21.0010"),
            _make_example("Invalid code product", "XXXX"),
        ]
        config = NormalizationConfig()
        result = normalize_examples(examples, config)
        assert len(result) == 1
        assert result[0].hts_code == "0101210010"

    def test_length_filter_too_short(self):
        examples = [
            _make_example("OK", "0101.21.0010"),  # Too short (2 chars)
            _make_example("A valid product description", "0101.21.0010"),
        ]
        config = NormalizationConfig(min_description_length=10)
        result = normalize_examples(examples, config)
        assert len(result) == 1

    def test_length_filter_too_long(self):
        examples = [
            _make_example("x" * 3000, "0101.21.0010"),
            _make_example("A valid product description", "0101.21.0010"),
        ]
        config = NormalizationConfig(max_description_length=2048)
        result = normalize_examples(examples, config)
        assert len(result) == 1

    def test_exact_dedup(self):
        examples = [
            _make_example("Live horses for breeding", "0101.21.0010"),
            _make_example("Live horses for breeding", "0101.21.0010"),
            _make_example("live horses for breeding", "0101.21.0010"),  # Case-insensitive dup
        ]
        config = NormalizationConfig()
        result = normalize_examples(examples, config)
        assert len(result) == 1

    def test_keeps_different_descriptions(self):
        examples = [
            _make_example("Live horses for breeding purposes", "0101.21.0010"),
            _make_example("Frozen beef cuts for export trade", "0202.30.0000"),
        ]
        config = NormalizationConfig()
        result = normalize_examples(examples, config)
        assert len(result) == 2

    def test_normalizes_codes(self):
        examples = [_make_example("A valid product description", "0101.21.0010")]
        config = NormalizationConfig()
        result = normalize_examples(examples, config)
        assert result[0].hts_code == "0101210010"  # Digits only
