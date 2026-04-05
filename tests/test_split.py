"""Tests for stratified splitting, ruling-level split, and leakage detection."""

import random

from hts_lora.data.build_examples import TrainingExample
from hts_lora.data.split import detect_leakage, stratified_split
from hts_lora.utils.config import SplitConfig


def _make_examples(n: int, seed: int = 42, with_rulings: bool = False) -> list[TrainingExample]:
    """Generate n examples spread across a few chapters."""
    rng = random.Random(seed)
    chapters = ["01", "02", "06", "17", "39", "61", "69", "73", "85", "94"]
    examples = []
    for i in range(n):
        chap = rng.choice(chapters)
        code = f"{chap}{rng.randint(10, 99)}{rng.randint(10, 99)}{rng.randint(1000, 9999)}"
        ruling = f"N{100000 + i // 2}" if with_rulings else None  # 2 products per ruling
        examples.append(
            TrainingExample(
                description=f"Product {i} from chapter {chap} with details {rng.randint(1000, 9999)}",
                hts_code=code,
                task_type="hierarchical_classify",
                ruling_number=ruling,
            )
        )
    return examples


class TestRowLevelSplit:
    def test_split_sizes(self):
        examples = _make_examples(100)
        config = SplitConfig(train=0.8, val=0.1, test=0.1, seed=42, split_by="row")
        split = stratified_split(examples, config)

        total = split.summary()["total"]
        assert total == 100
        assert split.summary()["train"] >= 60
        assert split.summary()["val"] >= 5
        assert split.summary()["test"] >= 5

    def test_no_overlap(self):
        examples = _make_examples(50)
        config = SplitConfig(seed=42, split_by="row")
        split = stratified_split(examples, config)

        train_descs = {e.description for e in split.train}
        val_descs = {e.description for e in split.val}
        test_descs = {e.description for e in split.test}

        assert len(train_descs & val_descs) == 0
        assert len(train_descs & test_descs) == 0
        assert len(val_descs & test_descs) == 0

    def test_deterministic(self):
        examples = _make_examples(50)
        config = SplitConfig(seed=42, split_by="row")
        split1 = stratified_split(examples, config)
        split2 = stratified_split(examples, config)

        assert [e.description for e in split1.train] == [e.description for e in split2.train]

    def test_handles_small_groups(self):
        examples = _make_examples(5)
        config = SplitConfig(seed=42, split_by="row")
        split = stratified_split(examples, config)
        assert split.summary()["total"] == 5


class TestRulingLevelSplit:
    def test_ruling_isolation(self):
        """No ruling_number should appear in more than one split."""
        examples = _make_examples(100, with_rulings=True)
        config = SplitConfig(seed=42, split_by="ruling")
        split = stratified_split(examples, config)

        train_rulings = {e.ruling_number for e in split.train if e.ruling_number}
        val_rulings = {e.ruling_number for e in split.val if e.ruling_number}
        test_rulings = {e.ruling_number for e in split.test if e.ruling_number}

        assert len(train_rulings & val_rulings) == 0
        assert len(train_rulings & test_rulings) == 0
        assert len(val_rulings & test_rulings) == 0

    def test_all_examples_preserved(self):
        examples = _make_examples(100, with_rulings=True)
        config = SplitConfig(seed=42, split_by="ruling")
        split = stratified_split(examples, config)
        assert split.summary()["total"] == 100

    def test_no_ruling_examples_go_to_train(self):
        """Examples without ruling_number should go to train only."""
        with_ruling = _make_examples(20, with_rulings=True)
        without_ruling = _make_examples(10, seed=99, with_rulings=False)
        examples = with_ruling + without_ruling
        config = SplitConfig(seed=42, split_by="ruling")
        split = stratified_split(examples, config)

        # All no-ruling examples should be in train
        val_no_ruling = [e for e in split.val if e.ruling_number is None]
        test_no_ruling = [e for e in split.test if e.ruling_number is None]
        assert len(val_no_ruling) == 0
        assert len(test_no_ruling) == 0

    def test_deterministic(self):
        examples = _make_examples(50, with_rulings=True)
        config = SplitConfig(seed=42, split_by="ruling")
        split1 = stratified_split(examples, config)
        split2 = stratified_split(examples, config)

        rulings1 = sorted(e.ruling_number for e in split1.val if e.ruling_number)
        rulings2 = sorted(e.ruling_number for e in split2.val if e.ruling_number)
        assert rulings1 == rulings2


class TestLeakageDetection:
    def test_no_leakage_on_distinct_data(self):
        train = [
            TrainingExample(description="Unique training product A", hts_code="0101210010", task_type="hierarchical_classify"),
            TrainingExample(description="Unique training product B", hts_code="0202300000", task_type="hierarchical_classify"),
        ]
        val = [
            TrainingExample(description="Completely different validation item", hts_code="8544300000", task_type="hierarchical_classify"),
        ]
        test = [
            TrainingExample(description="Another totally different test item", hts_code="6109100012", task_type="hierarchical_classify"),
        ]
        leaks = detect_leakage(train, val, test)
        assert leaks == {}

    def test_detects_exact_duplicate_leak(self):
        train = [
            TrainingExample(description="Live horses for breeding purposes in agricultural settings", hts_code="0101210010", task_type="hierarchical_classify"),
        ]
        val = [
            TrainingExample(description="Live horses for breeding purposes in agricultural settings", hts_code="0101210010", task_type="hierarchical_classify"),
        ]
        leaks = detect_leakage(train, val, [], threshold=0.5)
        assert leaks.get("val_vs_train", 0) > 0
