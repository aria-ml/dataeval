import re
from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.typing import NDArray

from dataeval.data._metadata import Metadata
from dataeval.data._split import (
    calculate_validation_fraction,
    split_dataset,
    validate_groupable,
    validate_labels,
    validate_stratifiable,
)
from dataeval.outputs._utils import SplitDatasetOutput


@pytest.fixture(scope="module")
def _base() -> NDArray[np.intp]:
    return np.arange(5, dtype=np.intp)


@pytest.fixture(scope="module")
def labels(_base):
    """Creates 500 labels with values [0-4]"""

    return np.repeat(_base, 20).astype(np.intp)  # ex. [0, 0, 1, 1, ...]


@pytest.fixture(scope="module")
def groups(_base, RNG):
    """Creates 100 labels and corresponding groups"""

    values = np.tile(_base, 20)  # ex. [0, 1, 0, 1, ...]
    return {"Discrete": values, "Binned": RNG.choice(_base, size=100).astype(np.intp)}


def get_metadata(labels, groups=None) -> Metadata:
    metadata = MagicMock(spec=Metadata)
    metadata.class_labels = labels
    if groups is not None:
        metadata.factor_names = ["Discrete", "Binned"]
        metadata.binned_data = np.column_stack(tuple(groups.values()))
    return metadata


def check_sample_leakage(splits: SplitDatasetOutput):
    test_inds = set(splits.test)

    print("\nChecking for Sample Leakage")
    for fold in splits.folds:
        train_inds = set(fold.train)
        val_inds = set(fold.val)
        assert test_inds.isdisjoint(train_inds), "common elements between train and test"
        assert test_inds.isdisjoint(val_inds), "common elements between val and test"
        assert val_inds.isdisjoint(train_inds), "common elements between train and val"

        assert len(test_inds) + len(train_inds) + len(val_inds) == 100


def check_stratification(labels: NDArray[np.int_], splits: SplitDatasetOutput, tolerance: float):
    """Checks that all folds and optional test split have all labels and tolerable label frequencies"""
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


def check_group_leakage(splits: SplitDatasetOutput, metadata: dict[str, NDArray], split_on: list[str]):
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
################################### Test Bad Input Arguments #######################################
####################################################################################################


@pytest.mark.required
class TestInputValidation:
    """Tests the boundaries of the inputs to split dataset"""

    def test_not_stratifiable(self):
        """Tests case where lowest label count is less than partitions"""

        with pytest.raises(ValueError):
            assert not validate_stratifiable(np.array([0, 1, 1]), 2)

    def test_stratifiable(self):
        """Tests that equal lowest label count and partitions is valid"""

        validate_stratifiable(np.array([0, 0, 1, 1]), 2)
        pass

    def test_continuous_labels(self, labels):
        """Tests that validate labels raises error with continuous labels"""

        cont_labels = labels.astype(np.float64) + np.random.uniform(-1, 1, size=labels.shape)
        error_statement = "Detected continuous labels. Labels must be discrete for proper stratification"
        with pytest.raises(ValueError, match=error_statement):
            validate_labels(cont_labels, total_partitions=2)

    def test_too_many_partitions(self):
        """Tests that an error is raised if there are more partitions than number of labels"""

        error_statement = (
            r"Total number of labels must be greater than the total number of partitions. "
            r"Got 1 labels and 1 total \[train, val, test\] partitions."
        )

        with pytest.raises(ValueError, match=error_statement):
            validate_labels(np.array([0]), total_partitions=1)

    @pytest.mark.parametrize("folds", (-1, 0))
    def test_invalid_folds(self, folds):
        """Tests that negative and 0 folds are invalid"""

        error_msg = f"Number of folds must be greater than or equal to 1, got {folds}"
        with pytest.raises(ValueError, match=error_msg):
            calculate_validation_fraction(num_folds=folds, test_frac=0.0, val_frac=0.0)

    @pytest.mark.parametrize("fraction", (-0.001, 1.001))
    def test_invalid_val_frac_values(self, fraction):
        """Tests that 1 fold and val frac is given, but raises error due to val fraction out of bounds [0-1]"""

        error_msg = f"val_frac out of bounds. Must be between 0.0 and 1.0, got {fraction}"
        with pytest.raises(ValueError, match=error_msg):
            calculate_validation_fraction(num_folds=1, test_frac=0.0, val_frac=fraction)

    @pytest.mark.parametrize("fraction", (-0.001, 1.001))
    def test_invalid_test_frac_values(self, fraction):
        """Tests that an error is raised if test fraction is not between 0.0 and 1.0"""

        error_msg = f"test_frac out of bounds. Must be between 0.0 and 1.0, got {fraction}"
        with pytest.raises(ValueError, match=error_msg):
            calculate_validation_fraction(num_folds=1, test_frac=fraction, val_frac=0.0)

    def test_invalid_multi_fold_and_val_frac(self):
        """Tests that an error is raised when both >1 fold and val fraction are given"""

        error_msg = "Can only specify val_frac when num_folds equals 1"
        with pytest.raises(ValueError, match=error_msg):
            calculate_validation_fraction(num_folds=5, test_frac=0.0, val_frac=0.2)

    def test_invalid_one_fold_no_val_frac(self):
        """Tests that 1 fold without specifying val fraction raises error"""

        error_msg = "If num_folds equals 1, must assign a value to val_frac"
        with pytest.raises(ValueError, match=error_msg):
            calculate_validation_fraction(num_folds=1, test_frac=0.0, val_frac=0.0)

    @pytest.mark.parametrize(
        "tfrac, vfrac, expected",
        [(0.0, 0.1, 0.1), (0.0, 1.0, 1.0), (0.5, 0.5, 0.25), (1.0, 0.5, 0.0), (0.4, 0.4, 0.6 * 0.4)],
    )
    def test_one_fold_valid_fractions(self, tfrac, vfrac, expected):
        """Tests that input fractions correctly calculates validation fraction"""

        validation_fraction = calculate_validation_fraction(
            1,
            test_frac=tfrac,
            val_frac=vfrac,
        )
        assert validation_fraction == expected

    @pytest.mark.parametrize(
        "folds, tfrac",
        [(2, 0.1), (2, 1.0), (2, 0.5), (5, 0.5), (6, 0.4)],
    )
    def test_multi_fold_valid_fractions(self, folds, tfrac):
        """Tests validation calculation with multiple folds"""

        expected = (1.0 / folds) * (1.0 - tfrac)
        result = calculate_validation_fraction(folds, test_frac=tfrac, val_frac=0.0)

        assert expected == result


@pytest.mark.required
class TestGroupData:
    def test_too_few_unique_groups(self, groups):
        """Tests that too many folds over small groups of data raises error and is not groupable"""

        values = groups["Discrete"]
        uniques, group_ids = np.unique(values, axis=0, return_inverse=True)

        error_msg = f"Unique groups ({len(uniques)}) must be greater than num partitions ({len(uniques) + 1})."
        with pytest.raises(ValueError, match=re.escape(error_msg)):
            validate_groupable(group_ids, len(uniques) + 1)

    def test_single_unique_group(self):
        """Tests that a single unique group is not groupable"""
        error_msg = "Unique groups (1) must be greater than 1."
        with pytest.raises(ValueError, match=re.escape(error_msg)):
            validate_groupable(np.ones(shape=(100, 1), dtype=np.intp), 1000)

    def test_no_valid_groups(self, labels, groups):
        metadata = get_metadata(labels, groups)
        keys = ["Foo", "Bar"]
        error_msg = "Unique groups (1) must be greater than 1."
        with pytest.raises(ValueError, match=re.escape(error_msg)):
            split_dataset(
                metadata,
                num_folds=1,
                stratify=False,
                split_on=keys,
                val_frac=0.2,
                test_frac=0.5,
            )


@pytest.mark.required
@pytest.mark.parametrize("val_frac", (0.1, 0.7, 0.99))
def test_split_dataset(labels, val_frac) -> None:
    """Tests no sample leakage at varying validation fractions"""
    metadata = get_metadata(labels)
    splits = split_dataset(metadata, val_frac=val_frac)
    check_sample_leakage(splits)


@pytest.mark.required
@pytest.mark.parametrize("num_folds", [1, 5])
@pytest.mark.parametrize("test_frac", [0.0, 0.25])
class TestFunctionalSplits:
    """Tests split dataset for label miscounts, group and sample leakage, and proper stratification"""

    def test_stratification(self, labels, num_folds, test_frac):
        """Tests stratification with no grouping"""

        metadata = get_metadata(labels)
        val_frac = 0.1 if num_folds == 1 else 0.0
        splits = split_dataset(metadata, num_folds=num_folds, stratify=True, val_frac=val_frac, test_frac=test_frac)
        check_sample_leakage(splits)
        check_stratification(labels, splits, tolerance=0.01)

    def test_grouping(self, labels, groups, num_folds, test_frac):
        """Tests grouping with no stratification"""

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
        """Tests grouping and stratification"""

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
