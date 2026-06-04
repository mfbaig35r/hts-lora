"""Hierarchical HTS classification metrics (v2 structured text output)."""

from __future__ import annotations

from typing import Any

from hts_lora.inference.parse_output import ParsedPrediction
from hts_lora.utils.hts_codes import (
    match_at_level,
    normalize_code,
    validate_code,
)
from hts_lora.utils.logging import get_logger

logger = get_logger("evaluation.metrics")


def compute_metrics(
    predictions: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute all evaluation metrics from a list of v2 prediction results.

    Each prediction dict should have:
        - hts_code: ground truth. Either a single code (str) or a list of
          acceptable codes (list[str]). List-shape is used for benchmarks
          like ATLAS where some examples accept any of several codes.
        - prediction: ParsedPrediction (or dict with same fields)
        - parse_ok: whether structured text parsing succeeded
        - abstain: whether the example is an abstention case (ground truth)
    """
    total = len(predictions)
    if total == 0:
        return {"total": 0}

    # Parse rate
    parse_ok_count = sum(1 for p in predictions if p.get("parse_ok", False))

    # Separate abstention and classification examples (ground truth)
    abstain_examples = [p for p in predictions if p.get("abstain", False)]
    classify_examples = [p for p in predictions if not p.get("abstain", False)]

    # Abstain accuracy: model should abstain on abstain examples
    abstain_correct = 0
    for p in abstain_examples:
        pred = _get_pred(p)
        if p.get("parse_ok") and pred is not None and _is_abstention(pred):
            abstain_correct += 1
    abstain_rate = abstain_correct / len(abstain_examples) if abstain_examples else 0.0

    # Classification metrics (non-abstain examples with valid predicted codes)
    classify_valid = []
    for p in classify_examples:
        pred = _get_pred(p)
        if not p.get("parse_ok") or pred is None:
            continue
        pred_code = _get_hts_code(pred)
        if pred_code and validate_code(pred_code):
            classify_valid.append(p)

    exact = 0
    chapter_match = 0
    heading_match = 0
    subheading_match = 0

    for p in classify_valid:
        gold = _gold_codes(p)
        pred_code = _get_hts_code(_get_pred(p))
        if _match_any(pred_code, gold, "exact"):
            exact += 1
        if _match_any(pred_code, gold, "chapter"):
            chapter_match += 1
        if _match_any(pred_code, gold, "heading"):
            heading_match += 1
        if _match_any(pred_code, gold, "subheading"):
            subheading_match += 1

    n_classify = len(classify_examples) or 1

    # Hierarchy consistency: do parsed chapter/heading/subheading match the HTS code?
    consistency_total = 0
    consistency_ok = 0
    for p in classify_valid:
        pred = _get_pred(p)
        pred_code = _get_hts_code(pred)
        if pred_code and validate_code(pred_code):
            consistency_total += 1
            if _check_hierarchy_consistency(pred, pred_code):
                consistency_ok += 1

    metrics: dict[str, float] = {
        "total": total,
        "parse_rate": parse_ok_count / total,
        "exact_match": exact / n_classify,
        "chapter_match": chapter_match / n_classify,
        "heading_match": heading_match / n_classify,
        "subheading_match": subheading_match / n_classify,
        "abstain_rate": abstain_rate,
        "abstain_count": len(abstain_examples),
        "hierarchy_consistency": (
            consistency_ok / consistency_total if consistency_total else 0.0
        ),
    }

    logger.info(
        f"Metrics: exact={metrics['exact_match']:.3f}, "
        f"chapter={metrics['chapter_match']:.3f}, "
        f"heading={metrics['heading_match']:.3f}, "
        f"parse_rate={metrics['parse_rate']:.3f}"
    )
    return metrics


def _gold_codes(p: dict[str, Any]) -> list[str]:
    """Return the ground-truth codes for a record as a list.

    Accepts either `hts_code: str` (single gold) or `hts_code: list[str]`
    (any-of-set gold, e.g. ATLAS multi-code rows).
    """
    raw = p.get("hts_code", "")
    if isinstance(raw, list):
        return [c for c in raw if c]
    return [raw] if raw else []


def _match_any(pred_code: str, gold: list[str], level: str) -> bool:
    """True if pred_code matches any of the gold codes at the given level."""
    for g in gold:
        try:
            if match_at_level(pred_code, g, level):
                return True
        except ValueError:
            continue
    return False


def _get_pred(p: dict[str, Any]) -> Any:
    """Get the prediction object (ParsedPrediction or dict)."""
    return p.get("prediction")


def _get_hts_code(pred: Any) -> str | None:
    """Extract HTS code from a ParsedPrediction or dict."""
    if pred is None:
        return None
    if isinstance(pred, ParsedPrediction):
        return pred.hts_code
    return pred.get("hts_code")


def _is_abstention(pred: Any) -> bool:
    """Check if prediction is an abstention."""
    if isinstance(pred, ParsedPrediction):
        return pred.is_abstention
    return pred.get("is_abstention", False)


def _check_hierarchy_consistency(pred: Any, hts_code: str) -> bool:
    """Check if parsed chapter/heading/subheading are consistent with the HTS code."""
    try:
        digits = normalize_code(hts_code)
    except ValueError:
        return False

    expected_chap = digits[:2]
    expected_head = digits[:4]
    expected_sub = digits[:6]

    if isinstance(pred, ParsedPrediction):
        pred_chap = pred.chapter_code
        pred_head = pred.heading_code
        pred_sub = pred.subheading_code
    else:
        pred_chap = pred.get("chapter_code")
        pred_head = pred.get("heading_code")
        pred_sub = pred.get("subheading_code")

    # Only check fields that are present
    if pred_chap and pred_chap.lstrip("0") != expected_chap.lstrip("0"):
        return False
    if pred_head:
        head_digits = pred_head.replace(".", "")
        if head_digits != expected_head:
            return False
    if pred_sub:
        sub_digits = pred_sub.replace(".", "")
        if sub_digits != expected_sub:
            return False

    return True
