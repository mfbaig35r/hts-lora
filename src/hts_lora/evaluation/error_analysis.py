"""Error analysis: bucket failures by type and sample for review (v2 ParsedPrediction)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from hts_lora.inference.parse_output import ParsedPrediction
from hts_lora.utils.hts_codes import match_at_level, validate_code
from hts_lora.utils.logging import get_logger

logger = get_logger("evaluation.error_analysis")

# Error bucket definitions
WRONG_CHAPTER = "wrong_chapter"
RIGHT_CHAPTER_WRONG_HEADING = "right_chapter_wrong_heading"
RIGHT_HEADING_WRONG_SUBHEADING = "right_heading_wrong_subheading"
RIGHT_SUBHEADING_WRONG_FULL = "right_subheading_wrong_full"
HALLUCINATED_CODE = "hallucinated_code"
PARSE_FAILURE = "parse_failure"
MISSING_PREDICTION = "missing_prediction"
FALSE_ABSTAIN = "false_abstain"
MISSED_ABSTAIN = "missed_abstain"


def _get_hts_code(pred: Any) -> str:
    """Extract HTS code from ParsedPrediction or dict."""
    if isinstance(pred, ParsedPrediction):
        return pred.hts_code or ""
    return str(pred.get("hts_code", "") or "")


def _is_abstention(pred: Any) -> bool:
    """Check if prediction is an abstention."""
    if isinstance(pred, ParsedPrediction):
        return pred.is_abstention
    return pred.get("is_abstention", False)


def analyze_errors(
    predictions: list[dict[str, Any]],
    max_samples_per_bucket: int = 5,
) -> dict[str, Any]:
    """Categorize prediction errors into buckets with sample failures."""
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for p in predictions:
        gt_code = p.get("hts_code", "")
        is_abstain = p.get("abstain", False)
        parse_ok = p.get("parse_ok", False)
        pred = p.get("prediction")

        sample = {
            "description": p.get("description", "")[:200],
            "ground_truth": gt_code,
        }

        if not parse_ok:
            sample["raw"] = p.get("raw", "")[:200]
            buckets[PARSE_FAILURE].append(sample)
            continue

        if pred is None:
            buckets[MISSING_PREDICTION].append(sample)
            continue

        pred_abstain = _is_abstention(pred)
        pred_code = _get_hts_code(pred)

        # Abstention errors
        if is_abstain and not pred_abstain:
            sample["predicted"] = pred_code
            buckets[MISSED_ABSTAIN].append(sample)
            continue
        if not is_abstain and pred_abstain:
            buckets[FALSE_ABSTAIN].append(sample)
            continue

        # Skip abstain examples that were correctly handled
        if is_abstain and pred_abstain:
            continue

        # Classification errors
        if not pred_code or not validate_code(pred_code):
            sample["predicted"] = pred_code
            buckets[HALLUCINATED_CODE].append(sample)
            continue

        if match_at_level(pred_code, gt_code, "exact"):
            continue  # Correct

        sample["predicted"] = pred_code

        if not match_at_level(pred_code, gt_code, "chapter"):
            buckets[WRONG_CHAPTER].append(sample)
        elif not match_at_level(pred_code, gt_code, "heading"):
            buckets[RIGHT_CHAPTER_WRONG_HEADING].append(sample)
        elif not match_at_level(pred_code, gt_code, "subheading"):
            buckets[RIGHT_HEADING_WRONG_SUBHEADING].append(sample)
        else:
            buckets[RIGHT_SUBHEADING_WRONG_FULL].append(sample)

    # Build summary
    summary: dict[str, Any] = {
        "bucket_counts": {k: len(v) for k, v in buckets.items()},
        "total_errors": sum(len(v) for v in buckets.values()),
        "buckets": {},
    }

    for bucket_name, samples in buckets.items():
        summary["buckets"][bucket_name] = {
            "count": len(samples),
            "samples": samples[:max_samples_per_bucket],
        }

    logger.info(f"Error analysis: {summary['total_errors']} total errors across {len(buckets)} buckets")
    return summary
