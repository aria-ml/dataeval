"""Validation tests for merge_datasets (vacuous index2label guard + datum shape)."""

import warnings

import numpy as np
import pytest

from dataeval.data import merge_datasets
from dataeval.protocols import DatasetMetadata


class _LabeledDataset:
    """AnnotatedDataset that exposes an index2label vocabulary."""

    def __init__(self, tag: str, n: int, index2label: dict[int, str]) -> None:
        self._items = [(np.zeros((3, 2, 2), dtype=np.float32), i) for i in range(n)]
        self.metadata = DatasetMetadata(id=tag, index2label=dict(index2label))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int):
        return self._items[index]


class _NoVocabDataset:
    """AnnotatedDataset whose metadata omits index2label entirely."""

    def __init__(self, tag: str, n: int) -> None:
        self._items = [(np.zeros((3, 2, 2), dtype=np.float32), i) for i in range(n)]
        self.metadata: DatasetMetadata = DatasetMetadata(id=tag)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int):
        return self._items[index]


class _TripleDatumDataset:
    """AnnotatedDataset whose datum tuple has a different arity (3-tuple vs 2-tuple)."""

    def __init__(self, tag: str, n: int, index2label: dict[int, str]) -> None:
        self._items = [(np.zeros((3, 2, 2), dtype=np.float32), i, {"id": i}) for i in range(n)]
        self.metadata = DatasetMetadata(id=tag, index2label=dict(index2label))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int):
        return self._items[index]


@pytest.mark.required
class TestMergeValidation:
    def test_no_vocab_datasets_warn(self):
        """Merging datasets that both lack index2label surfaces the vacuous-equality case."""
        with pytest.warns(UserWarning, match="index2label"):
            merged = merge_datasets(_NoVocabDataset("a", 2), _NoVocabDataset("b", 3))
        assert len(merged) == 5

    def test_valid_merge_no_warning(self):
        """Existing valid merges (shared vocabulary) still succeed with no warning."""
        i2l = {0: "cat", 1: "dog"}
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning becomes an error
            merged = merge_datasets(
                _LabeledDataset("a", 2, i2l),
                _LabeledDataset("b", 3, i2l),
            )
        assert len(merged) == 5
        assert merged.metadata.get("index2label") == i2l

    def test_mismatched_vocab_still_raises(self):
        """Disagreeing vocabularies remain a hard error."""
        with pytest.raises(ValueError, match="index2label"):
            merge_datasets(
                _LabeledDataset("a", 1, {0: "cat"}),
                _LabeledDataset("b", 1, {0: "dog"}),
            )

    def test_some_vocab_some_not_raises(self):
        """If some expose index2label and others don't, keep the hard error."""
        with pytest.raises(ValueError, match="index2label"):
            merge_datasets(
                _LabeledDataset("a", 1, {0: "cat"}),
                _NoVocabDataset("b", 1),
            )

    def test_datum_shape_mismatch_warns(self):
        """Obvious datum-arity mismatches emit a warning but still merge.

        Both datasets share a vocabulary, so only the datum-shape warning fires.
        """
        i2l = {0: "cat", 1: "dog"}
        with pytest.warns(UserWarning, match="datum"):
            merged = merge_datasets(_LabeledDataset("a", 2, i2l), _TripleDatumDataset("b", 2, i2l))
        assert len(merged) == 4
