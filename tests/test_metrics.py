"""Tests for hierarchical matching and evaluation metrics."""

from hts_lora.evaluation.metrics import compute_metrics


def _make_prediction(gt_code: str, pred_code: str, abstain: bool = False, parse_ok: bool = True):
    """Helper to create a prediction record."""
    pred = {
        "predicted_code": None if abstain else pred_code,
        "confidence": 0.0 if abstain else 0.9,
        "rationale": "test",
        "abstain": abstain,
    }
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

    def test_json_parse_rate(self):
        preds = [
            _make_prediction("0101210010", "0101.21.0010", parse_ok=True),
            _make_prediction("0202300000", "0202.30.0000", parse_ok=False),
        ]
        metrics = compute_metrics(preds)
        assert metrics["json_parse_rate"] == 0.5

    def test_empty_predictions(self):
        metrics = compute_metrics([])
        assert metrics["total"] == 0

    def test_abstain_handling(self):
        preds = [
            {
                "hts_code": "__ABSTAIN__",
                "prediction": {"predicted_code": None, "confidence": 0.0, "abstain": True},
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
