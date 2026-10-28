import re
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray

from dataeval import Metadata
from dataeval.data import DatasetSplits, TrainValSplit, split_dataset
from dataeval.data._split import (
    _build_multilabel_matrix,
    _calculate_validation_fraction,
    _item_groups,
    _IterativeStratifiedKFold,
    _multilabel_find_best_split,
    _multilabel_make_splits,
    _multilabel_single_split,
    _single_split,
    _validate_groupable,
    _validate_labels,
    _validate_stratifiable,
)


@pytest.fixture(scope="module")
def _base() -> NDArray[np.intp]:
    return np.arange(5, dtype=np.intp)


@pytest.fixture(scope="module")
def labels(_base):
    """Creates 500 labels with values [0-4]."""
    return np.repeat(_base, 20).astype(np.intp)  # ex. [0, 0, 1, 1, ...]


@pytest.fixture(scope="module")
def groups(_base, RNG):
    """Creates 100 labels and corresponding groups."""
    values = np.tile(_base, 20)  # ex. [0, 1, 0, 1, ...]
    return {"Discrete": values, "Binned": RNG.choice(_base, size=100).astype(np.intp)}


def get_metadata(labels, groups=None) -> Metadata:
    metadata = MagicMock(spec=Metadata)
    metadata.class_labels = labels
    metadata.multi_target = False  # classification, as a real Metadata would declare
    if groups is not None:
        metadata.factor_names = ["Discrete", "Binned"]
        metadata.factor_data = np.column_stack(tuple(groups.values()))
    return metadata


def check_sample_leakage(splits: DatasetSplits):
    test_inds = set(splits.test)

    print("\nChecking for Sample Leakage")
    for fold in splits.folds:
        train_inds = set(fold.train)
        val_inds = set(fold.val)
        assert test_inds.isdisjoint(train_inds), "common elements between train and test"
        assert test_inds.isdisjoint(val_inds), "common elements between val and test"
        assert val_inds.isdisjoint(train_inds), "common elements between train and val"

        assert len(test_inds) + len(train_inds) + len(val_inds) == 100


def check_stratification(labels: NDArray[np.intp], splits: DatasetSplits, tolerance: float):
    """Checks that all folds and optional test split have all labels and tolerable label frequencies."""
    unique_labels, class_counts = np.unique(labels, return_counts=True)
    class_freqs = class_counts / np.sum(class_counts)

    def check_tolerance(labels):
        uniques, counts = np.unique(labels, return_counts=True)
        freq = counts / np.sum(counts)

        assert (uniques == unique_labels).all(), "Mismatch between split labels and all labels"
        assert np.allclose(
            freq,
            class_freqs,
            atol=tolerance,  # , atol=1 / len(class_freqs)
        ), "Frequency difference greater than tolerance"

    if splits.test.size > 0:
        check_tolerance(labels[splits.test])

    for fold in splits.folds:
        check_tolerance(labels[fold.train])


def check_group_leakage(splits: DatasetSplits, metadata: dict[str, NDArray], split_on: list[str]):
    relevant_metadata = np.stack([metadata[key] for key in split_on], axis=1)
    _, groups = np.unique(relevant_metadata, axis=0, return_inverse=True)
    test_groups = set(groups[splits.test])
    for fold in splits.folds:
        train_groups = set(groups[fold.train])
        val_groups = set(groups[fold.val])
        assert test_groups.isdisjoint(train_groups), "common groups between train and test"
        assert test_groups.isdisjoint(val_groups), "common groups between val and test"
        assert val_groups.isdisjoint(train_groups), "common groups between train and val"


####################################################################################################
# Test Bad Input Arguments #######################################
####################################################################################################


@pytest.mark.required
class TestInputValidation:
    """Tests the boundaries of the inputs to split dataset."""

    def test_not_stratifiable(self):
        """Tests case where lowest label count is less than partitions."""
        with pytest.raises(ValueError, match="Unable to stratify due to label frequency"):
            assert not _validate_stratifiable(np.array([0, 1, 1]), 2)

    def test_stratifiable(self):
        """Tests that equal lowest label count and partitions is valid."""
        _validate_stratifiable(np.array([0, 0, 1, 1]), 2)
        pass

    def test_continuous_labels(self, labels):
        """Tests that validate labels raises error with continuous labels."""
        cont_labels = labels.astype(np.float64) + np.random.uniform(-1, 1, size=labels.shape)
        error_statement = "Detected continuous labels. Labels must be discrete for proper stratification"
        with pytest.raises(ValueError, match=error_statement):
            _validate_labels(cont_labels, total_partitions=2)

    def test_too_many_partitions(self):
        """Tests that an error is raised if there are more partitions than number of labels."""
        error_statement = (
            r"Total number of labels must be greater than the total number of partitions. "
            r"Got 1 labels and 1 total \[train, val, test\] partitions."
        )

        with pytest.raises(ValueError, match=error_statement):
            _validate_labels(np.array([0]), total_partitions=1)

    @pytest.mark.parametrize("folds", [-1, 0])
    def test_invalid_folds(self, folds):
        """Tests that negative and 0 folds are invalid."""
        error_msg = f"Number of folds must be greater than or equal to 1, got {folds}"
        with pytest.raises(ValueError, match=error_msg):
            _calculate_validation_fraction(num_folds=folds, test_frac=0.0, val_frac=0.0)

    @pytest.mark.parametrize("fraction", [-0.001, 1.001])
    def test_invalid_val_frac_values(self, fraction):
        """Tests that 1 fold and val frac is given, but raises error due to val fraction out of bounds [0-1]."""
        error_msg = f"val_frac out of bounds. Must be between 0.0 and 1.0, got {fraction}"
        with pytest.raises(ValueError, match=error_msg):
            _calculate_validation_fraction(num_folds=1, test_frac=0.0, val_frac=fraction)

    @pytest.mark.parametrize("fraction", [-0.001, 1.001])
    def test_invalid_test_frac_values(self, fraction):
        """Tests that an error is raised if test fraction is not between 0.0 and 1.0."""
        error_msg = f"test_frac out of bounds. Must be between 0.0 and 1.0, got {fraction}"
        with pytest.raises(ValueError, match=error_msg):
            _calculate_validation_fraction(num_folds=1, test_frac=fraction, val_frac=0.0)

    def test_invalid_multi_fold_and_val_frac(self):
        """Tests that an error is raised when both >1 fold and val fraction are given."""
        error_msg = "Can only specify val_frac when num_folds equals 1"
        with pytest.raises(ValueError, match=error_msg):
            _calculate_validation_fraction(num_folds=5, test_frac=0.0, val_frac=0.2)

    def test_invalid_one_fold_no_val_frac(self):
        """Tests that 1 fold without specifying val fraction raises error."""
        error_msg = "If num_folds equals 1, must assign a value to val_frac"
        with pytest.raises(ValueError, match=error_msg):
            _calculate_validation_fraction(num_folds=1, test_frac=0.0, val_frac=0.0)

    @pytest.mark.parametrize(
        ("tfrac", "vfrac", "expected"),
        [(0.0, 0.1, 0.1), (0.0, 1.0, 1.0), (0.5, 0.5, 0.25), (1.0, 0.5, 0.0), (0.4, 0.4, 0.6 * 0.4)],
    )
    def test_one_fold_valid_fractions(self, tfrac, vfrac, expected):
        """Tests that input fractions correctly calculates validation fraction."""
        validation_fraction = _calculate_validation_fraction(
            1,
            test_frac=tfrac,
            val_frac=vfrac,
        )
        assert validation_fraction == expected

    @pytest.mark.parametrize(
        ("folds", "tfrac"),
        [(2, 0.1), (2, 1.0), (2, 0.5), (5, 0.5), (6, 0.4)],
    )
    def test_multi_fold_valid_fractions(self, folds, tfrac):
        """Tests validation calculation with multiple folds."""
        expected = (1.0 / folds) * (1.0 - tfrac)
        result = _calculate_validation_fraction(folds, test_frac=tfrac, val_frac=0.0)

        assert expected == result


@pytest.mark.required
class TestGroupData:
    def test_too_few_unique_groups(self, groups):
        """Tests that too many folds over small groups of data raises error and is not groupable."""
        values = groups["Discrete"]
        uniques, group_ids = np.unique(values, axis=0, return_inverse=True)

        error_msg = f"Unique groups ({len(uniques)}) must be greater than num partitions ({len(uniques) + 1})."
        with pytest.raises(ValueError, match=re.escape(error_msg)):
            _validate_groupable(group_ids, len(uniques) + 1)

    def test_single_unique_group(self):
        """Tests that a single unique group is not groupable."""
        error_msg = "Unique groups (1) must be greater than 1."
        with pytest.raises(ValueError, match=re.escape(error_msg)):
            _validate_groupable(np.ones(shape=(100, 1), dtype=np.intp), 1000)

    def test_no_valid_groups(self, labels, groups):
        """Names that resolve to no factor are refused, and the message says which.

        Previously this fell through to "Unique groups (1) must be greater than 1", which
        names neither the factor nor the reason — the grouping had already been silently
        reduced to nothing by then.
        """
        metadata = get_metadata(labels, groups)
        keys = ["Foo", "Bar"]
        with pytest.raises(ValueError, match=re.escape("['Bar', 'Foo']")):
            split_dataset(
                metadata,
                num_folds=1,
                stratify=False,
                split_on=keys,
                val_frac=0.2,
                test_frac=0.5,
            )


@pytest.mark.required
@pytest.mark.parametrize("val_frac", [0.1, 0.7, 0.99])
def test_split_dataset(labels, val_frac) -> None:
    """Tests no sample leakage at varying validation fractions."""
    metadata = get_metadata(labels)
    splits = split_dataset(metadata, val_frac=val_frac)
    check_sample_leakage(splits)


@pytest.mark.required
@pytest.mark.parametrize("num_folds", [1, 5])
@pytest.mark.parametrize("test_frac", [0.0, 0.25])
class TestFunctionalSplits:
    """Tests split dataset for label miscounts, group and sample leakage, and proper stratification."""

    def test_stratification(self, labels, num_folds, test_frac):
        """Tests stratification with no grouping."""
        metadata = get_metadata(labels)
        val_frac = 0.1 if num_folds == 1 else 0.0
        splits = split_dataset(metadata, num_folds=num_folds, stratify=True, val_frac=val_frac, test_frac=test_frac)
        check_sample_leakage(splits)
        check_stratification(labels, splits, tolerance=0.01)

    def test_grouping(self, labels, groups, num_folds, test_frac):
        """Tests grouping with no stratification."""
        metadata = get_metadata(labels, groups)
        val_frac = 0.1 if num_folds == 1 else 0.0
        keys: list[str] = list(groups.keys())
        splits = split_dataset(
            metadata,
            num_folds=num_folds,
            stratify=False,
            split_on=keys,
            val_frac=val_frac,
            test_frac=test_frac,
        )
        check_sample_leakage(splits)
        check_group_leakage(splits, groups, keys)

    def test_grouped_stratification(self, labels, groups, test_frac, num_folds):
        """Tests grouping and stratification."""
        metadata = get_metadata(labels, groups)
        val_frac = 0.1 if num_folds == 1 else 0.0
        keys: list[str] = list(groups.keys())
        splits = split_dataset(
            metadata,
            num_folds=num_folds,
            stratify=True,
            split_on=keys,
            val_frac=val_frac,
            test_frac=test_frac,
        )
        check_sample_leakage(splits)
        check_group_leakage(splits, groups, keys)
        check_stratification(labels, splits, tolerance=0.15)


####################################################################################################
# Regression / OD Split Tests #####################################
####################################################################################################


def _make_od_metadata(
    n_images: int,
    n_classes: int,
    detections_per_image: int = 3,
    seed: int = 42,
) -> MagicMock:
    """Create a mock Metadata object simulating an OD dataset."""
    rng = np.random.default_rng(seed)
    n_detections = n_images * detections_per_image
    item_indices = np.repeat(np.arange(n_images, dtype=np.intp), detections_per_image)
    class_labels = rng.integers(0, n_classes, size=n_detections).astype(np.intp)

    metadata = MagicMock(spec=Metadata)
    metadata.class_labels = class_labels
    metadata.item_indices = item_indices
    metadata.item_count = n_images
    return metadata


class TestMaxFoldsRegression:
    """Regression tests for the unique_groups=2 bug that capped max_folds at 2."""

    def test_single_split_respects_split_frac(self):
        """single_split should produce splits close to the requested fraction, not 50/50."""
        labels = np.repeat(np.arange(5, dtype=np.intp), 40)  # 200 labels, 5 classes
        index = np.arange(len(labels))
        split = _single_split(index, labels, split_frac=0.15, stratified=True)
        ratio = len(split.val) / len(index)
        # Should be within 5% of target, NOT the old 50%
        assert abs(ratio - 0.15) < 0.05, f"Split ratio {ratio:.3f} too far from target 0.15"

    def test_single_split_small_frac(self):
        """10% split should not produce 50/50."""
        labels = np.repeat(np.arange(5, dtype=np.intp), 40)
        index = np.arange(len(labels))
        split = _single_split(index, labels, split_frac=0.10, stratified=True)
        ratio = len(split.val) / len(index)
        assert abs(ratio - 0.10) < 0.05

    def test_single_split_large_frac(self):
        """30% split should not produce 50/50."""
        labels = np.repeat(np.arange(5, dtype=np.intp), 40)
        index = np.arange(len(labels))
        split = _single_split(index, labels, split_frac=0.30, stratified=True)
        ratio = len(split.val) / len(index)
        assert abs(ratio - 0.30) < 0.05

    def test_split_dataset_proportions_with_test_frac(self):
        """split_dataset with test_frac=0.15 should not put all data in test."""
        labels = np.repeat(np.arange(5, dtype=np.intp), 20)  # 100 labels
        metadata = get_metadata(labels)
        splits = split_dataset(metadata, num_folds=1, stratify=True, test_frac=0.15, val_frac=0.15)
        n = len(labels)

        # Test set should be ~15%, NOT 100%
        test_frac_actual = len(splits.test) / n
        assert test_frac_actual < 0.25, f"Test fraction {test_frac_actual:.2f} is too large"
        assert test_frac_actual > 0.05, f"Test fraction {test_frac_actual:.2f} is too small"

        # Train and val should be non-empty
        assert len(splits.folds[0].train) > 0, "Train set is empty"
        assert len(splits.folds[0].val) > 0, "Val set is empty"


class TestBuildMultilabelMatrix:
    """Tests for _build_multilabel_matrix."""

    def test_basic(self):
        class_labels = np.array([0, 1, 2, 0, 1], dtype=np.intp)
        item_indices = np.array([0, 0, 0, 1, 1], dtype=np.intp)
        matrix = _build_multilabel_matrix(class_labels, item_indices, n_images=2)
        expected = np.array([[1, 1, 1], [1, 1, 0]], dtype=np.int8)
        np.testing.assert_array_equal(matrix, expected)

    def test_single_detection_per_image(self):
        class_labels = np.array([0, 1, 2], dtype=np.intp)
        item_indices = np.array([0, 1, 2], dtype=np.intp)
        matrix = _build_multilabel_matrix(class_labels, item_indices, n_images=3)
        expected = np.eye(3, dtype=np.int8)
        np.testing.assert_array_equal(matrix, expected)

    def test_image_with_no_detections(self):
        """Images with no detections should have all-zero rows."""
        class_labels = np.array([0, 1], dtype=np.intp)
        item_indices = np.array([0, 2], dtype=np.intp)  # image 1 has no detections
        matrix = _build_multilabel_matrix(class_labels, item_indices, n_images=3)
        np.testing.assert_array_equal(matrix[1], [0, 0])

    def test_duplicate_class_in_image(self):
        """Multiple detections of same class in one image should still be 1."""
        class_labels = np.array([0, 0, 0], dtype=np.intp)
        item_indices = np.array([0, 0, 0], dtype=np.intp)
        matrix = _build_multilabel_matrix(class_labels, item_indices, n_images=1)
        np.testing.assert_array_equal(matrix, [[1]])


class TestIterativeStratifiedKFold:
    """Tests for _IterativeStratifiedKFold."""

    def test_fold_sizes_balanced(self):
        """All folds should have roughly equal sizes."""
        rng = np.random.default_rng(0)
        n, c = 100, 5
        y = rng.integers(0, 2, size=(n, c)).astype(np.int8)
        splitter = _IterativeStratifiedKFold(n_splits=5)
        fold_sizes = [len(test) for _, test in splitter.split(np.arange(n), y)]
        assert sum(fold_sizes) == n
        assert max(fold_sizes) - min(fold_sizes) <= 2, f"Fold sizes too unbalanced: {fold_sizes}"

    def test_no_overlap_between_folds(self):
        """Each sample should appear in exactly one fold's test set."""
        rng = np.random.default_rng(1)
        n, c = 50, 3
        y = rng.integers(0, 2, size=(n, c)).astype(np.int8)
        splitter = _IterativeStratifiedKFold(n_splits=3)
        all_test = []
        for _, test in splitter.split(np.arange(n), y):
            all_test.extend(test.tolist())
        assert sorted(all_test) == list(range(n))

    def test_train_test_complementary(self):
        """Train and test indices should be complementary for each fold."""
        n = 30
        y = np.eye(5, dtype=np.int8)[:3].T  # 5 samples, 3 classes
        y = np.tile(y, (6, 1))  # 30 samples
        splitter = _IterativeStratifiedKFold(n_splits=3)
        for train, test in splitter.split(np.arange(n), y):
            assert len(train) + len(test) == n
            assert set(train.tolist()).isdisjoint(set(test.tolist()))

    def test_rare_labels_distributed(self):
        """Rare labels should appear in multiple folds, not all in one."""
        n = 100
        y = np.zeros((n, 4), dtype=np.int8)
        y[:, 0] = 1  # common label: all 100 images
        y[:4, 1] = 1  # rare label: only 4 images
        y[:50, 2] = 1  # medium label
        y[:20, 3] = 1  # less common
        splitter = _IterativeStratifiedKFold(n_splits=4)
        for _, test in splitter.split(np.arange(n), y):
            # Rare label (4 total) should have exactly 1 per fold
            assert y[test, 1].sum() >= 1


class TestMultilabelSplitFunctions:
    """Tests for _multilabel_single_split, _multilabel_make_splits, _multilabel_find_best_split."""

    @pytest.fixture
    def multilabel_data(self):
        rng = np.random.default_rng(42)
        n_images, n_classes = 200, 8
        ml = rng.integers(0, 2, size=(n_images, n_classes)).astype(np.int8)
        # Ensure every class appears in at least 10 images
        for c in range(n_classes):
            if ml[:, c].sum() < 10:
                ml[rng.choice(n_images, 10, replace=False), c] = 1
        return ml

    def test_single_split_no_leakage(self, multilabel_data):
        ml = multilabel_data
        index = np.arange(len(ml))
        split = _multilabel_single_split(index, ml, split_frac=0.2)
        assert set(split.train.tolist()).isdisjoint(set(split.val.tolist()))
        assert len(split.train) + len(split.val) == len(ml)

    def test_single_split_respects_fraction(self, multilabel_data):
        ml = multilabel_data
        index = np.arange(len(ml))
        split = _multilabel_single_split(index, ml, split_frac=0.2)
        ratio = len(split.val) / len(index)
        assert abs(ratio - 0.2) < 0.05

    def test_make_splits_covers_all_indices(self, multilabel_data):
        ml = multilabel_data
        index = np.arange(len(ml))
        splits = _multilabel_make_splits(index, ml, n_folds=5)
        assert len(splits) == 5
        for split in splits:
            assert len(split.train) + len(split.val) == len(ml)

    def test_find_best_split_prefers_balanced(self):
        """find_best_split should prefer splits with balanced per-label frequencies."""
        n = 100
        ml = np.zeros((n, 2), dtype=np.int8)
        ml[:50, 0] = 1  # class 0: first 50
        ml[50:, 1] = 1  # class 1: last 50

        good_split = TrainValSplit(np.arange(0, 80, dtype=np.intp), np.arange(80, 100, dtype=np.intp))
        bad_split = TrainValSplit(np.arange(50, 100, dtype=np.intp), np.arange(0, 50, dtype=np.intp))
        best = _multilabel_find_best_split(ml, [good_split, bad_split], 0.2)
        # good_split has both classes in train; bad_split puts all of class 0 in val
        assert np.array_equal(best.train, good_split.train)


class TestODSplitDataset:
    """Integration tests for split_dataset with OD (multi-label) datasets."""

    def test_od_returns_image_level_indices(self):
        """All returned indices must be valid image indices, not detection indices."""
        meta = _make_od_metadata(n_images=200, n_classes=10)
        splits = split_dataset(meta, num_folds=1, stratify=True, test_frac=0.15, val_frac=0.15)
        n_images = 200
        all_indices = np.concatenate([splits.test, splits.folds[0].train, splits.folds[0].val])
        assert all_indices.max() < n_images, "Index exceeds number of images"
        assert all_indices.min() >= 0

    def test_od_no_leakage(self):
        """No image should appear in more than one split."""
        meta = _make_od_metadata(n_images=200, n_classes=10)
        splits = split_dataset(meta, num_folds=1, stratify=True, test_frac=0.15, val_frac=0.15)
        train = set(splits.folds[0].train.tolist())
        val = set(splits.folds[0].val.tolist())
        test = set(splits.test.tolist())
        assert train.isdisjoint(val), "Train/Val overlap"
        assert train.isdisjoint(test), "Train/Test overlap"
        assert val.isdisjoint(test), "Val/Test overlap"
        assert len(train | val | test) == 200

    def test_od_split_proportions(self):
        """OD split proportions should approximate requested fractions."""
        n_images = 500
        meta = _make_od_metadata(n_images=n_images, n_classes=10)
        splits = split_dataset(meta, num_folds=1, stratify=True, test_frac=0.15, val_frac=0.15)
        test_frac = len(splits.test) / n_images
        train_frac = len(splits.folds[0].train) / n_images
        val_frac = len(splits.folds[0].val) / n_images
        assert abs(test_frac - 0.15) < 0.05, f"Test frac {test_frac:.3f} not ~0.15"
        assert train_frac > 0.5, f"Train frac {train_frac:.3f} too small"
        assert val_frac > 0.05, f"Val frac {val_frac:.3f} too small"

    def test_od_no_stratify(self):
        """OD split without stratification should still use image-level indices."""
        meta = _make_od_metadata(n_images=200, n_classes=10)
        splits = split_dataset(meta, num_folds=1, stratify=False, test_frac=0.15, val_frac=0.15)
        all_indices = np.concatenate([splits.test, splits.folds[0].train, splits.folds[0].val])
        assert all_indices.max() < 200
        assert len(set(all_indices.tolist())) == 200

    def test_od_multi_fold(self):
        """OD split with multiple folds should work correctly."""
        meta = _make_od_metadata(n_images=200, n_classes=10)
        splits = split_dataset(meta, num_folds=3, stratify=True, test_frac=0.15)
        assert len(splits.folds) == 3
        for fold in splits.folds:
            all_idx = np.concatenate([splits.test, fold.train, fold.val])
            assert len(set(all_idx.tolist())) == 200

    def test_od_class_representation_in_splits(self):
        """Every class present in the dataset should appear in the test split."""
        meta = _make_od_metadata(n_images=500, n_classes=10, detections_per_image=5, seed=0)
        splits = split_dataset(meta, num_folds=1, stratify=True, test_frac=0.2, val_frac=0.2)
        ml = _build_multilabel_matrix(meta.class_labels, meta.item_indices, 500)
        test_classes = ml[splits.test].sum(axis=0)
        # All classes should appear in the test split
        assert (test_classes > 0).all(), f"Missing classes in test: {test_classes}"

    def test_ic_dataset_unchanged(self, labels):
        """IC datasets should still use the standard single-label path."""
        metadata = get_metadata(labels)
        splits = split_dataset(metadata, num_folds=1, stratify=True, val_frac=0.2, test_frac=0.2)
        check_sample_leakage(splits)
        check_stratification(labels, splits, tolerance=0.02)


@pytest.mark.filterwarnings("ignore:`patient_id` was dropped:UserWarning")
@pytest.mark.required
class TestSplitOnNamesAFactorTheMetadataDoesNotCarry:
    """``split_on`` exists to keep rows that belong together on one side of a split.

    Grouping by a *subset* of what was asked for hands back folds that do not honor the
    request, and nothing downstream can tell — so an unresolvable name is refused rather
    than skipped.
    """

    @staticmethod
    def _md(n=120):
        from dataeval import Metadata

        rng = np.random.default_rng(0)
        return Metadata.from_factors(
            {
                # Near-unique, so Metadata drops it as an identifier — and a per-entity id
                # is exactly what a caller reaches for first when grouping a split.
                "patient_id": np.array([f"p{i % 80:03d}" for i in range(n)]),
                "site": np.array([f"s{i % 6}" for i in range(n)]),
            },
            class_labels=rng.integers(0, 2, n),
        )

    def test_a_dropped_factor_is_refused_not_silently_skipped(self):
        md = self._md()
        assert "patient_id" not in md.factor_names

        with pytest.raises(ValueError, match="patient_id"):
            split_dataset(md, num_folds=2, split_on=["patient_id", "site"])

    def test_the_message_says_why_the_factor_is_missing(self):
        """A dropped factor and a typo are different mistakes and read differently."""
        with pytest.raises(ValueError, match="cardinality_over_budget"):
            split_dataset(self._md(), num_folds=2, split_on=["patient_id"])

    def test_a_misspelled_factor_is_refused(self):
        with pytest.raises(ValueError, match=r"\['sight'\]"):
            split_dataset(self._md(), num_folds=2, split_on=["sight"])

    def test_a_factor_the_metadata_carries_still_groups(self):
        assert split_dataset(self._md(), num_folds=2, split_on=["site"]) is not None


def _videos(classes: ArrayLike, empty_frame: int | None = None, extra: Sequence[list[list[tuple[int, int]]]] = ()):
    """One 20-frame video per ``(a, b)``: track 1 of class ``a`` and track 2 of class ``b`` in every frame.

    ``empty_frame`` names a frame left without detections in every such video. ``extra`` appends
    videos given frame by frame, as lists of ``(label, track_id)`` detections.
    """
    from dataeval.data import SequenceFrames
    from tests.data.conftest import FactorVideos

    pairs = np.asarray(classes).reshape(-1, 2).tolist()
    videos = [[[] if f == empty_frame else [(a, 1), (b, 2)] for f in range(20)] for a, b in pairs] + list(extra)
    return SequenceFrames(cast(Any, FactorVideos(videos, [{} for _ in videos])))


def _partitions(splits: DatasetSplits) -> list[tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.intp]]]:
    """``(train, val, test)`` for every fold."""
    return [(fold.train, fold.val, splits.test) for fold in splits.folds]


def _assert_groups_disjoint(splits: DatasetSplits, group: NDArray[np.intp]) -> None:
    for parts in _partitions(splits):
        train, val, test = (set(group[p].tolist()) for p in parts)
        assert train.isdisjoint(val), "a group is in both train and val"
        assert train.isdisjoint(test), "a group is in both train and test"
        assert val.isdisjoint(test), "a group is in both val and test"


@pytest.mark.required
class TestSplitOnForDetectionAndFrames:
    """``split_on`` on detection and frame data keeps each group in one partition; v1.1 ignored it (#1619)."""

    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize("num_folds", [1, 2])
    def test_split_on_sequence_keeps_every_video_whole(self, seed, num_folds):
        rng = np.random.default_rng(seed)
        frames = _videos(rng.integers(2, size=(4 + seed % 3, 2)))
        val_frac = 0.2 if num_folds == 1 else 0.0
        splits = split_dataset(
            frames, num_folds=num_folds, stratify=False, split_on=["sequence"], test_frac=0.2, val_frac=val_frac
        )
        assert len(splits.folds) == num_folds
        _assert_groups_disjoint(splits, np.asarray(frames.frame_map)[:, 0])

    def test_a_detection_free_frame_stays_with_its_video(self):
        frames = _videos([(0, 1)] * 5, empty_frame=5)
        splits = split_dataset(frames, split_on=["sequence"], test_frac=0.2, val_frac=0.2)
        for parts in _partitions(splits):
            partition_of = {int(i): k for k, p in enumerate(parts) for i in p}
            for v in range(5):
                assert partition_of.get(v * 20 + 5) == partition_of[v * 20], f"frame 5 of video {v} left its video"

    @pytest.mark.parametrize("num_folds", [1, 2])
    def test_every_frame_is_assigned_once(self, num_folds):
        frames = _videos([(0, 1)] * 6, empty_frame=5)
        val_frac = 0.2 if num_folds == 1 else 0.0
        splits = split_dataset(frames, num_folds=num_folds, split_on=["sequence"], test_frac=0.2, val_frac=val_frac)
        for parts in _partitions(splits):
            np.testing.assert_array_equal(np.sort(np.concatenate(parts)), np.arange(len(frames)))

    def test_a_factor_varying_within_an_item_raises(self):
        md = Metadata(_videos([(0, 1)] * 4))
        md.add_factors({"side": np.arange(md.level_counts[md.label_level]) % 2}, level=md.label_level)
        with pytest.raises(ValueError, match="'side' varies within"):
            split_dataset(md, split_on=["side"], test_frac=0.2, val_frac=0.2)

    def test_too_few_groups_raises(self):
        with pytest.raises(ValueError, match=re.escape("Got 2 groups and 3 total [train, val, test] partitions")):
            split_dataset(_videos([(0, 1)] * 2), num_folds=2, split_on=["sequence"], test_frac=0.2)

    def test_too_few_groups_for_a_single_split_counts_the_validation_partition(self):
        with pytest.raises(ValueError, match=re.escape("Got 2 groups and 3 total [train, val, test] partitions")):
            split_dataset(_videos([(0, 1)] * 2), split_on=["sequence"], test_frac=0.2, val_frac=0.2)

    @pytest.mark.filterwarnings("ignore:`sequence` was binned automatically:UserWarning")
    @pytest.mark.filterwarnings("ignore:Declared encodings are finer:UserWarning")
    def test_past_20_videos_groups_are_runs_of_videos_unless_levels_are_declared(self):
        n = 25
        frames = _videos([(0, 1)] * n)
        video = np.asarray(frames.frame_map)[:, 0]

        binned = _item_groups(Metadata(frames), ["sequence"], len(frames))
        assert len(np.unique(binned)) < n
        for v in range(n):  # a video is never split, binned or not
            assert len(np.unique(binned[video == v])) == 1

        declared = Metadata(frames, factor_levels={"sequence": range(n)})
        assert len(np.unique(_item_groups(declared, ["sequence"], len(frames)))) == n
        splits = split_dataset(declared, split_on=["sequence"], test_frac=0.2, val_frac=0.2)
        _assert_groups_disjoint(splits, video)
        assert len(set(video[splits.test].tolist())) == round(0.2 * n)

    def test_stratified_grouped_split_keeps_each_class(self):
        """Sorted single-class videos, so unstratified blocks would miss a class.

        Class 1 reaches exactly as many videos as there are partitions only by counting the last
        video, whose class-1 detections are all in frames 10-19: a group holds a class if any of
        its frames does.
        """
        late = [[(0, 1), (0 if f < 10 else 1, 2)] for f in range(20)]
        frames = _videos([(0, 0)] * 6 + [(1, 1)] * 2, extra=[late])
        classes_of = [{0}] * 6 + [{1}] * 2 + [{0, 1}]
        video = np.asarray(frames.frame_map)[:, 0]
        splits = split_dataset(frames, stratify=True, split_on=["sequence"], test_frac=0.2, val_frac=0.2)
        for parts in _partitions(splits):
            assert all(set().union(*(classes_of[v] for v in set(video[p].tolist()))) == {0, 1} for p in parts)
        _assert_groups_disjoint(splits, video)

    def test_a_class_in_fewer_groups_than_partitions_raises(self):
        """Class 1 in 2 of 10 videos cannot reach train, val and test; this once returned an empty train."""
        frames = _videos([(0, 1)] * 2 + [(0, 0)] * 8)
        message = "the rarest class has only 2 groups, fewer than the 3 partitions requested"
        with pytest.raises(ValueError, match=re.escape(message)):
            split_dataset(frames, stratify=True, split_on=["sequence"], test_frac=0.2, val_frac=0.2)

    def test_a_grouped_split_that_empties_a_partition_raises(self):
        """Class 1 in 3 of 6 videos passes the count, but holding out half leaves its train side none to split."""
        frames = _videos([(0, 1)] * 3 + [(0, 0)] * 3)
        message = "Splitting 6 groups left a partition empty (test 3, fold 0 train 0, fold 0 val 3 groups)"
        with pytest.raises(ValueError, match=re.escape(message)):
            split_dataset(frames, stratify=True, split_on=["sequence"], test_frac=0.5, val_frac=0.2)

    def test_a_sparse_frame_view_is_split_by_frame(self):
        """Fewer detections than frames once sent a frame view down the classification path, which split rows."""
        sparse = [[[] if f % 4 == 3 else [(0, 1)] for f in range(20)] for _ in range(5)]
        frames = _videos([], extra=sparse)  # 75 detections over 100 frames
        splits = split_dataset(frames, split_on=["sequence"], test_frac=0.2, val_frac=0.2)
        _assert_groups_disjoint(splits, np.asarray(frames.frame_map)[:, 0])
        for parts in _partitions(splits):
            np.testing.assert_array_equal(np.sort(np.concatenate(parts)), np.arange(len(frames)))

    def test_a_factor_the_frames_do_not_carry_is_refused(self):
        with pytest.raises(ValueError, match=re.escape("['weather']")):
            split_dataset(_videos([(0, 1)] * 4), split_on=["weather"], test_frac=0.2, val_frac=0.2)

    def test_a_metadata_container_is_refused_on_detections(self):
        """A bare container has no item-level rows, so a detection-free item could not be placed."""
        md = Metadata(_videos([(0, 1)] * 4))
        container = SimpleNamespace(
            class_labels=md.class_labels,
            factor_names=md.factor_names,
            factor_data=md.factor_data,
            is_binned=[True] * len(md.factor_names),
            item_indices=md.item_indices,
            item_count=md.item_count,
        )
        with pytest.raises(ValueError, match="Metadata"):
            split_dataset(cast(Any, container), split_on=["sequence"], test_frac=0.2, val_frac=0.2)

    def test_split_on_for_image_classification_is_unchanged(self):
        site = np.arange(60) % 6
        md = Metadata.from_factors({"site": site.astype(str)}, class_labels=np.arange(60) % 2)
        _assert_groups_disjoint(split_dataset(md, split_on=["site"], test_frac=0.2, val_frac=0.2), site)

    def test_an_image_classification_dataset_does_not_take_the_detection_path(self, monkeypatch):
        """Its label level differs from its item level too, so routing on levels would send it down the grouped path."""
        from tests.data.conftest import MockICDataset

        md = Metadata(cast(Any, MockICDataset([i % 2 for i in range(60)], {0: "a", 1: "b"})))
        site = np.arange(60) % 6
        md.add_factors({"site": site}, level=md.item_level)

        def detection_path(*args, **kwargs):
            raise AssertionError("an image classification dataset took the detection path")

        monkeypatch.setattr("dataeval.data._split._split_od", detection_path)
        _assert_groups_disjoint(split_dataset(md, split_on=["site"], test_frac=0.2, val_frac=0.2), site)
