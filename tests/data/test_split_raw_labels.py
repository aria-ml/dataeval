"""Tests for split_dataset accepting a bare array/sequence of class labels."""

import numpy as np
import pytest

from dataeval.data import split_dataset
from dataeval.data._split import DatasetSplits


def _assert_valid_disjoint(splits: DatasetSplits, n: int) -> None:
    """Every returned index is a valid item index and the partitions are disjoint."""
    test = set(splits.test.tolist())
    for fold in splits.folds:
        train = set(fold.train.tolist())
        val = set(fold.val.tolist())

        # Disjoint partitions
        assert test.isdisjoint(train), "train/test overlap"
        assert test.isdisjoint(val), "val/test overlap"
        assert train.isdisjoint(val), "train/val overlap"

        # Valid, in-range, integer indices
        all_idx = np.concatenate([splits.test, fold.train, fold.val])
        assert all_idx.min() >= 0
        assert all_idx.max() < n
        assert len(set(all_idx.tolist())) == n  # partitions cover every item exactly once


@pytest.mark.required
class TestSplitRawLabels:
    """split_dataset should accept a bare sequence/array of integer class labels."""

    def test_stratified_ndarray_labels(self):
        """A raw NDArray of labels can be split with stratification."""
        labels = np.repeat(np.arange(5, dtype=np.intp), 20)  # 100 labels, 5 classes
        splits = split_dataset(labels, num_folds=1, stratify=True, val_frac=0.2, test_frac=0.2)
        _assert_valid_disjoint(splits, n=len(labels))
        assert len(splits.test) > 0
        assert len(splits.folds) == 1
        assert len(splits.folds[0].val) > 0

    def test_stratified_python_list_labels(self):
        """A raw Python list of labels is also accepted (bare Sequence[int])."""
        labels = [i % 4 for i in range(80)]  # 80 labels, 4 classes
        splits = split_dataset(labels, num_folds=3, stratify=True, test_frac=0.2)
        _assert_valid_disjoint(splits, n=len(labels))
        assert len(splits.folds) == 3

    def test_simple_nonstratified_labels(self):
        """Non-stratified simple split also works on raw labels."""
        labels = np.repeat(np.arange(5, dtype=np.intp), 20)
        splits = split_dataset(labels, num_folds=1, stratify=False, val_frac=0.3)
        _assert_valid_disjoint(splits, n=len(labels))

    def test_split_on_with_raw_labels_raises(self):
        """Requesting grouping via split_on with only raw labels raises a clear ValueError."""
        labels = np.repeat(np.arange(5, dtype=np.intp), 20)
        with pytest.raises(ValueError, match="split_on"):
            split_dataset(labels, num_folds=1, stratify=True, split_on=["scene"], val_frac=0.2)
