"""Tests for hierarchical matching and evaluation metrics (v2 ParsedPrediction)."""

from hts_lora.evaluation.metrics import compute_metrics
from hts_lora.inference.parse_output import ParsedPrediction


def _make_prediction(gt_code: str, pred_code: str, abstain: bool = False, parse_ok: bool = True):
    """Helper to create a v2 prediction record."""
    if abstain:
        pred = ParsedPrediction(
            is_abstention=True,
            abstention_reason="Test abstention",
            parse_ok=True,
            raw="Cannot classify: test",
        )
    else:
        pred = ParsedPrediction(
            hts_code=pred_code,
            parse_ok=True,
            raw=f"HTS Code: {pred_code}",
        )
    return {
        "hts_code": gt_code,
        "prediction": pred if parse_ok else None,
        "parse_ok": parse_ok,
        "abstain": False,  # ground truth is not abstain (unless specified)
    }


class TestComputeMetrics:
    def test_perfect_predictions(self):
        preds = [
            _make_prediction("0101210010", "0101.21.0010"),
            _make_prediction("0202300000", "0202.30.0000"),
        ]
        metrics = compute_metrics(preds)
        assert metrics["exact_match"] == 1.0
        assert metrics["chapter_match"] == 1.0
        assert metrics["heading_match"] == 1.0
        assert metrics["subheading_match"] == 1.0

    def test_wrong_chapter(self):
        preds = [
            _make_prediction("0101210010", "0202.30.0000"),
        ]
        metrics = compute_metrics(preds)
        assert metrics["exact_match"] == 0.0
        assert metrics["chapter_match"] == 0.0

    def test_right_chapter_wrong_heading(self):
        preds = [
            _make_prediction("0101210010", "0102.29.0000"),
        ]
        metrics = compute_metrics(preds)
        assert metrics["exact_match"] == 0.0
        assert metrics["chapter_match"] == 1.0
        assert metrics["heading_match"] == 0.0

    def test_right_heading_wrong_subheading(self):
        preds = [
            _make_prediction("0101210010", "0101.29.0000"),
        ]
        metrics = compute_metrics(preds)
        assert metrics["heading_match"] == 1.0
        assert metrics["subheading_match"] == 0.0
        assert metrics["exact_match"] == 0.0

    def test_parse_rate(self):
        preds = [
            _make_prediction("0101210010", "0101.21.0010", parse_ok=True),
            _make_prediction("0202300000", "0202.30.0000", parse_ok=False),
        ]
        metrics = compute_metrics(preds)
        assert metrics["parse_rate"] == 0.5

    def test_empty_predictions(self):
        metrics = compute_metrics([])
        assert metrics["total"] == 0

    def test_abstain_handling(self):
        preds = [
            {
                "hts_code": "__ABSTAIN__",
                "prediction": ParsedPrediction(
                    is_abstention=True,
                    abstention_reason="Too vague",
                    parse_ok=True,
                    raw="Cannot classify: Too vague",
                ),
                "parse_ok": True,
                "abstain": True,
            },
        ]
        metrics = compute_metrics(preds)
        assert metrics["abstain_rate"] == 1.0

    def test_mixed_results(self):
        preds = [
            _make_prediction("0101210010", "0101.21.0010"),  # Exact
            _make_prediction("0202300000", "0201.30.0000"),  # Wrong heading, right chapter
            _make_prediction("8544300000", "8544.30.0000"),  # Exact
        ]
        metrics = compute_metrics(preds)
        assert metrics["exact_match"] == 2 / 3
        assert metrics["chapter_match"] == 3 / 3

    def test_hierarchy_consistency(self):
        pred = ParsedPrediction(
            chapter_code="85",
            heading_code="85.44",
            subheading_code="8544.30",
            hts_code="8544.30.0000",
            parse_ok=True,
            raw="test",
        )
        preds = [{
            "hts_code": "8544300000",
            "prediction": pred,
            "parse_ok": True,
            "abstain": False,
        }]
        metrics = compute_metrics(preds)
        assert metrics["hierarchy_consistency"] == 1.0

    def test_hierarchy_inconsistency(self):
        pred = ParsedPrediction(
            chapter_code="73",  # Wrong chapter for 8544
            heading_code="85.44",
            subheading_code="8544.30",
            hts_code="8544.30.0000",
            parse_ok=True,
            raw="test",
        )
        preds = [{
            "hts_code": "8544300000",
            "prediction": pred,
            "parse_ok": True,
            "abstain": False,
        }]
        metrics = compute_metrics(preds)
        assert metrics["hierarchy_consistency"] == 0.0
