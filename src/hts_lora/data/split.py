"""Stratified train/valid/test split with leakage prevention.

Supports two split modes:
- "row": Split individual examples (original behavior)
- "ruling": Group by ruling_number, assign entire ruling to one split
"""

from __future__ import annotations

import random
from collections import defaultdict

from datasketch import MinHash, MinHashLSH

from hts_lora.data.build_examples import TrainingExample
from hts_lora.utils.config import SplitConfig
from hts_lora.utils.hts_codes import chapter
from hts_lora.utils.logging import get_logger

logger = get_logger("data.split")


class DataSplit:
    """Container for train/valid/test splits."""

    def __init__(
        self,
        train: list[TrainingExample],
        valid: list[TrainingExample],
        test: list[TrainingExample],
    ):
        self.train = train
        self.valid = valid
        self.test = test

    def summary(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "valid": len(self.valid),
            "test": len(self.test),
            "total": len(self.train) + len(self.valid) + len(self.test),
        }


def stratified_split(
    examples: list[TrainingExample],
    config: SplitConfig,
) -> DataSplit:
    """Split examples stratified by chapter with leakage detection.

    When split_by="ruling", entire rulings are kept together in one split.
    Synthetic examples (no ruling_number) go to train only.
    """
    if config.split_by == "ruling":
        return _ruling_level_split(examples, config)
    return _row_level_split(examples, config)


def _ruling_level_split(
    examples: list[TrainingExample],
    config: SplitConfig,
) -> DataSplit:
    """Group by ruling_number, then assign entire groups to splits."""
    rng = random.Random(config.seed)

    # Separate: with ruling_number vs without (synthetic → train only)
    ruling_groups: dict[str, list[TrainingExample]] = defaultdict(list)
    no_ruling: list[TrainingExample] = []

    for ex in examples:
        if ex.ruling_number:
            ruling_groups[ex.ruling_number].append(ex)
        else:
            no_ruling.append(ex)

    # Group rulings by chapter for stratification
    chapter_rulings: dict[str, list[str]] = defaultdict(list)
    for ruling_num, group in ruling_groups.items():
        # Use chapter of first non-abstain example
        chap = "__other__"
        for ex in group:
            if not ex.abstain:
                try:
                    chap = chapter(ex.hts_code) if config.stratify_by == "chapter" else ex.hts_code
                except ValueError:
                    pass
                break
        chapter_rulings[chap].append(ruling_num)

    train: list[TrainingExample] = []
    valid: list[TrainingExample] = []
    test: list[TrainingExample] = []

    for chap_key, ruling_nums in chapter_rulings.items():
        rng.shuffle(ruling_nums)
        n = len(ruling_nums)
        n_val = max(1, int(n * config.val)) if n >= 3 else 0
        n_test = max(1, int(n * config.test)) if n >= 3 else 0
        n_train = n - n_val - n_test

        if n_train <= 0:
            for rn in ruling_nums:
                train.extend(ruling_groups[rn])
            continue

        for rn in ruling_nums[:n_train]:
            train.extend(ruling_groups[rn])
        for rn in ruling_nums[n_train:n_train + n_val]:
            valid.extend(ruling_groups[rn])
        for rn in ruling_nums[n_train + n_val:]:
            test.extend(ruling_groups[rn])

    # Synthetic examples → train only
    train.extend(no_ruling)

    rng.shuffle(train)
    rng.shuffle(valid)
    rng.shuffle(test)

    split = DataSplit(train=train, valid=valid, test=test)
    logger.info(f"Ruling-level split: {split.summary()}")

    # Verify no ruling leaks across splits
    _verify_ruling_isolation(train, valid, test)

    # Check for text leakage
    leaks = detect_leakage(train, valid, test)
    if leaks:
        logger.warning(f"Potential data leakage detected: {leaks}")
    else:
        logger.info("No cross-split leakage detected")

    return split


def _verify_ruling_isolation(
    train: list[TrainingExample],
    valid: list[TrainingExample],
    test: list[TrainingExample],
) -> None:
    """Verify no ruling_number appears in more than one split."""
    train_rulings = {e.ruling_number for e in train if e.ruling_number}
    valid_rulings = {e.ruling_number for e in valid if e.ruling_number}
    test_rulings = {e.ruling_number for e in test if e.ruling_number}

    train_valid = train_rulings & valid_rulings
    train_test = train_rulings & test_rulings
    valid_test = valid_rulings & test_rulings

    if train_valid or train_test or valid_test:
        logger.error(
            f"Ruling isolation violated! "
            f"train∩valid={len(train_valid)}, train∩test={len(train_test)}, valid∩test={len(valid_test)}"
        )
    else:
        logger.info("Ruling isolation verified: no ruling in multiple splits")


def _row_level_split(
    examples: list[TrainingExample],
    config: SplitConfig,
) -> DataSplit:
    """Split individual examples stratified by chapter (original behavior)."""
    rng = random.Random(config.seed)

    groups: dict[str, list[TrainingExample]] = defaultdict(list)
    for ex in examples:
        if ex.abstain:
            key = "__abstain__"
        else:
            key = chapter(ex.hts_code) if config.stratify_by == "chapter" else ex.hts_code
        groups[key].append(ex)

    train: list[TrainingExample] = []
    valid: list[TrainingExample] = []
    test: list[TrainingExample] = []

    for key, group in groups.items():
        rng.shuffle(group)
        n = len(group)
        n_val = max(1, int(n * config.val)) if n >= 3 else 0
        n_test = max(1, int(n * config.test)) if n >= 3 else 0
        n_train = n - n_val - n_test

        if n_train <= 0:
            train.extend(group)
            continue

        train.extend(group[:n_train])
        valid.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(valid)
    rng.shuffle(test)

    split = DataSplit(train=train, valid=valid, test=test)
    logger.info(f"Row-level split: {split.summary()}")

    leaks = detect_leakage(train, valid, test)
    if leaks:
        logger.warning(f"Potential data leakage detected: {leaks}")
    else:
        logger.info("No cross-split leakage detected")

    return split


def detect_leakage(
    train: list[TrainingExample],
    valid: list[TrainingExample],
    test: list[TrainingExample],
    threshold: float = 0.85,
    num_perm: int = 128,
) -> dict[str, int]:
    """Detect near-duplicate descriptions across splits using MinHash LSH."""
    leaks: dict[str, int] = {}

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i, ex in enumerate(train):
        mh = _make_minhash(ex.description, num_perm)
        try:
            lsh.insert(f"train_{i}", mh)
        except ValueError:
            continue

    valid_leaks = 0
    for ex in valid:
        mh = _make_minhash(ex.description, num_perm)
        if lsh.query(mh):
            valid_leaks += 1
    if valid_leaks:
        leaks["valid_vs_train"] = valid_leaks

    test_leaks = 0
    for ex in test:
        mh = _make_minhash(ex.description, num_perm)
        if lsh.query(mh):
            test_leaks += 1
    if test_leaks:
        leaks["test_vs_train"] = test_leaks

    return leaks


def _make_minhash(text: str, num_perm: int) -> MinHash:
    m = MinHash(num_perm=num_perm)
    words = text.lower().split()
    for i in range(len(words) - 2):
        shingle = " ".join(words[i:i + 3])
        m.update(shingle.encode("utf-8"))
    return m
