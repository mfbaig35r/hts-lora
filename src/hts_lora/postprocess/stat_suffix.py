"""Layer A: stat-suffix validation and completion.

Post-processes a ParsedPrediction by validating its 10-digit HTS code
against a known-valid index. When the code is unrecognized but its
8-digit subheading IS in the index, the code is replaced with the
most-frequent valid 10-digit completion observed under that subheading.

The index is produced by scripts/build_stat_suffix_index.py.

Layer A only touches the `hts_code` field. The chapter / heading /
subheading parsed from the model output are left intact. The audit log
records every change so we can report a stat_suffix_corrected_rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from hts_lora.inference.parse_output import ParsedPrediction
from hts_lora.utils.io import read_json
from hts_lora.utils.logging import get_logger

logger = get_logger("postprocess.stat_suffix")

PickerStrategy = Literal["most_common", "first_valid"]


@dataclass
class StatSuffixIndex:
    """Loaded stat-suffix index. Cheap dict lookups; load once at startup."""

    valid_10digit: dict[str, int]                # code -> training frequency
    completions_by_8digit: dict[str, list[str]]  # 8-digit prefix -> sorted children
    meta: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> StatSuffixIndex:
        raw = read_json(path)
        return cls(
            valid_10digit=raw.get("valid_10digit", {}),
            completions_by_8digit=raw.get("completions_by_8digit", {}),
            meta=raw.get("_meta", {}),
        )

    def is_valid_10digit(self, code: str) -> bool:
        return _strip(code) in self.valid_10digit

    def valid_completions(self, eight_digit: str) -> list[str]:
        return self.completions_by_8digit.get(_strip(eight_digit), [])

    def has_8digit(self, eight_digit: str) -> bool:
        return _strip(eight_digit) in self.completions_by_8digit


@dataclass
class StatSuffixResult:
    """Outcome of applying Layer A to one ParsedPrediction."""

    prediction: ParsedPrediction
    changed: bool
    original_hts_code: str | None
    chosen_hts_code: str | None
    reason: str   # "valid", "completed", "no_8digit_match", "no_change_possible", "skipped_abstain", "skipped_no_code"


def validate_and_complete(
    prediction: ParsedPrediction,
    index: StatSuffixIndex,
    strategy: PickerStrategy = "most_common",
) -> StatSuffixResult:
    """Validate the 10-digit code in `prediction` against `index`.

    If the code is already valid: no change.
    If the code is unrecognized but its 8-digit prefix is in the index:
    replace with the picker's chosen completion.
    If the 8-digit prefix is unknown: leave alone (we can't fix what we
    can't anchor).
    """
    if prediction.is_abstention:
        return StatSuffixResult(
            prediction=prediction,
            changed=False,
            original_hts_code=prediction.hts_code,
            chosen_hts_code=prediction.hts_code,
            reason="skipped_abstain",
        )

    original = prediction.hts_code
    if not original:
        return StatSuffixResult(
            prediction=prediction,
            changed=False,
            original_hts_code=None,
            chosen_hts_code=None,
            reason="skipped_no_code",
        )

    stripped = _strip(original)
    # Codes shorter than 10 chars are padded with 0 on the right
    # (matches hts_codes.normalize_code).
    if len(stripped) < 10:
        stripped = stripped.ljust(10, "0")
    elif len(stripped) > 10:
        stripped = stripped[:10]

    if index.is_valid_10digit(stripped):
        return StatSuffixResult(
            prediction=prediction,
            changed=False,
            original_hts_code=original,
            chosen_hts_code=original,
            reason="valid",
        )

    eight = stripped[:8]
    completions = index.valid_completions(eight)
    if not completions:
        return StatSuffixResult(
            prediction=prediction,
            changed=False,
            original_hts_code=original,
            chosen_hts_code=original,
            reason="no_8digit_match",
        )

    chosen = _pick(completions, strategy, index)
    if not chosen or chosen == stripped:
        return StatSuffixResult(
            prediction=prediction,
            changed=False,
            original_hts_code=original,
            chosen_hts_code=original,
            reason="no_change_possible",
        )

    # Replace the code; format with dots to match parser output convention.
    new_code = f"{chosen[:4]}.{chosen[4:6]}.{chosen[6:10]}"
    new_pred = _replace_hts_code(prediction, new_code)
    return StatSuffixResult(
        prediction=new_pred,
        changed=True,
        original_hts_code=original,
        chosen_hts_code=new_code,
        reason="completed",
    )


def _pick(
    completions: list[str],
    strategy: PickerStrategy,
    index: StatSuffixIndex,
) -> str | None:
    if not completions:
        return None
    if strategy == "first_valid":
        return completions[0]
    # "most_common": completions are already sorted by descending freq in the
    # index (see build_stat_suffix_index.py), so just take the head.
    return completions[0]


def _strip(code: str) -> str:
    return code.replace(".", "").replace("-", "").strip() if code else ""


def _replace_hts_code(pred: ParsedPrediction, new_code: str) -> ParsedPrediction:
    """Return a new ParsedPrediction with `hts_code` replaced. All other
    fields preserved. Used so the caller's original object isn't mutated."""
    return ParsedPrediction(
        chapter_code=pred.chapter_code,
        chapter_desc=pred.chapter_desc,
        heading_code=pred.heading_code,
        heading_desc=pred.heading_desc,
        subheading_code=pred.subheading_code,
        subheading_desc=pred.subheading_desc,
        hts_code=new_code,
        reasoning=pred.reasoning,
        provides_for=pred.provides_for,
        is_abstention=pred.is_abstention,
        abstention_reason=pred.abstention_reason,
        parse_ok=pred.parse_ok,
        raw=pred.raw,
    )
