"""Hierarchical HTS classification metrics."""

from __future__ import annotations

import math
from typing import Any

from hts_lora.utils.hts_codes import match_at_level, normalize_code, validate_code
from hts_lora.utils.logging import get_logger

logger = get_logger("evaluation.metrics")


def compute_metrics(
    predictions: list[dict[str, Any]],
    top_k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute all evaluation metrics from a list of prediction results.

    Each prediction dict should have:
        - hts_code: ground truth code
        - prediction: parsed model output dict (or None)
        - parse_ok: whether JSON parsing succeeded
        - abstain: whether the example is an abstention case
    """
    if top_k_values is None:
        top_k_values = [1, 3, 5]

    total = len(predictions)
    if total == 0:
        return {"total": 0}

    # JSON parse rate
    parse_ok = sum(1 for p in predictions if p.get("parse_ok", False))

    # Filter to parseable predictions with valid predicted codes
    valid_preds = []
    for p in predictions:
        if not p.get("parse_ok") or p.get("prediction") is None:
            continue
        pred = p["prediction"]
        if pred.get("abstain"):
            valid_preds.append(p)
        elif pred.get("predicted_code") and validate_code(str(pred["predicted_code"])):
            valid_preds.append(p)

    # Separate abstention and classification examples
    abstain_examples = [p for p in predictions if p.get("abstain", False)]
    classify_examples = [p for p in predictions if not p.get("abstain", False)]

    # Abstain accuracy: model should abstain on abstain examples
    abstain_correct = 0
    for p in abstain_examples:
        if p.get("parse_ok") and p.get("prediction", {}).get("abstain"):
            abstain_correct += 1
    abstain_rate = abstain_correct / len(abstain_examples) if abstain_examples else 0.0

    # Classification metrics (non-abstain examples)
    classify_valid = [
        p for p in classify_examples
        if p.get("parse_ok")
        and p.get("prediction") is not None
        and p["prediction"].get("predicted_code")
        and validate_code(str(p["prediction"]["predicted_code"]))
    ]

    exact = 0
    chapter_match = 0
    heading_match = 0
    subheading_match = 0

    for p in classify_valid:
        gt = p["hts_code"]
        pred_code = str(p["prediction"]["predicted_code"])
        if match_at_level(pred_code, gt, "exact"):
            exact += 1
        if match_at_level(pred_code, gt, "chapter"):
            chapter_match += 1
        if match_at_level(pred_code, gt, "heading"):
            heading_match += 1
        if match_at_level(pred_code, gt, "subheading"):
            subheading_match += 1

    n_classify = len(classify_examples) or 1

    # Top-k accuracy (for rerank mode with rankings)
    top_k_acc = {}
    rerank_preds = [
        p for p in classify_valid
        if p["prediction"].get("rankings")
    ]
    for k in top_k_values:
        correct = 0
        for p in rerank_preds:
            gt = p["hts_code"]
            rankings = p["prediction"]["rankings"]
            top_codes = [r["code"] for r in sorted(rankings, key=lambda r: r["rank"])[:k]]
            if any(validate_code(c) and match_at_level(c, gt, "exact") for c in top_codes):
                correct += 1
        top_k_acc[f"top_{k}_accuracy"] = correct / len(rerank_preds) if rerank_preds else 0.0

    # Confidence calibration (simple ECE approximation)
    calibration = _expected_calibration_error(classify_valid)

    metrics: dict[str, float] = {
        "total": total,
        "json_parse_rate": parse_ok / total,
        "exact_match": exact / n_classify,
        "chapter_match": chapter_match / n_classify,
        "heading_match": heading_match / n_classify,
        "subheading_match": subheading_match / n_classify,
        "abstain_rate": abstain_rate,
        "abstain_count": len(abstain_examples),
        "confidence_ece": calibration,
        **top_k_acc,
    }

    logger.info(
        f"Metrics: exact={metrics['exact_match']:.3f}, "
        f"chapter={metrics['chapter_match']:.3f}, "
        f"heading={metrics['heading_match']:.3f}, "
        f"parse_rate={metrics['json_parse_rate']:.3f}"
    )
    return metrics


def _expected_calibration_error(
    predictions: list[dict[str, Any]],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) using confidence scores."""
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]

    for p in predictions:
        pred = p.get("prediction", {})
        confidence = pred.get("confidence", 0.5)
        if not isinstance(confidence, (int, float)):
            continue
        confidence = max(0.0, min(1.0, float(confidence)))
        gt = p["hts_code"]
        pred_code = str(pred.get("predicted_code", ""))
        correct = validate_code(pred_code) and match_at_level(pred_code, gt, "exact")
        bin_idx = min(int(confidence * n_bins), n_bins - 1)
        bins[bin_idx].append((confidence, correct))

    ece = 0.0
    total = sum(len(b) for b in bins)
    if total == 0:
        return 0.0

    for bin_items in bins:
        if not bin_items:
            continue
        avg_conf = sum(c for c, _ in bin_items) / len(bin_items)
        avg_acc = sum(1 for _, correct in bin_items if correct) / len(bin_items)
        ece += len(bin_items) / total * abs(avg_conf - avg_acc)

    return round(ece, 4)
