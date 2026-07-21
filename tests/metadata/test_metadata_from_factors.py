"""Tests for Metadata.from_factors — building Metadata from raw factor arrays.

These cover the "minimal data" construction path: a user with only a factor
table (and optional class labels) can build a dataeval.Metadata without owning a
full MAITE image dataset.
"""

import numpy as np
import pytest

from dataeval import Metadata
from dataeval.bias import Balance, Diversity, Parity
from dataeval.exceptions import ShapeMismatchError
from dataeval.protocols import MetadataLike


class TestMetadataFromFactors:
    def test_basic_discrete(self):
        factors = {
            "age_bin": np.array([0, 1, 0, 2, 1, 0]),
            "gender": np.array([1, 0, 1, 0, 1, 0]),
        }
        labels = np.array([0, 1, 0, 1, 0, 1])
        md = Metadata.from_factors(factors, labels)

        assert isinstance(md, Metadata)
        assert isinstance(md, MetadataLike)
        assert sorted(md.factor_names) == ["age_bin", "gender"]
        assert md.factor_data.shape == (6, 2)
        np.testing.assert_array_equal(md.class_labels, labels)
        assert len(list(md.is_discrete)) == 2
        assert all(md.is_discrete)

    def test_array_interface(self):
        factors = {"f": np.array([0, 1, 2, 0, 1])}
        md = Metadata.from_factors(factors, np.array([0, 0, 1, 1, 0]))
        # Array protocol must work without a bound dataset
        assert len(md) == 5
        assert md.shape == (5, 1)
        assert md.ndim == 2
        arr = np.asarray(md)
        assert arr.shape[0] == 5

    def test_no_class_labels_defaults_single_class(self):
        factors = {"f": np.array([0, 1, 0, 1])}
        md = Metadata.from_factors(factors)
        np.testing.assert_array_equal(md.class_labels, np.zeros(4, dtype=np.intp))

    def test_index2label_passthrough(self):
        factors = {"f": np.array([0, 1, 0])}
        md = Metadata.from_factors(factors, np.array([0, 1, 0]), index2label={0: "cat", 1: "dog"})
        assert md.index2label[0] == "cat"
        assert md.index2label[1] == "dog"

    def test_continuous_factor_binning(self):
        rng = np.random.default_rng(0)
        factors = {"brightness": rng.normal(size=50)}
        labels = rng.integers(0, 2, size=50)
        md = Metadata.from_factors(factors, labels, continuous_factor_bins={"brightness": 5})
        # continuous factor should be binned into ints, marked not-discrete
        assert md.factor_data.dtype == np.int64
        assert list(md.is_discrete) == [False]
        # digitize_data returns 1-indexed bin ids in [1, n_bins]
        assert md.factor_data[:, 0].max() <= 5
        assert len(np.unique(md.factor_data[:, 0])) <= 5

    def test_length_mismatch_raises(self):
        factors = {"a": np.array([0, 1, 2]), "b": np.array([0, 1])}
        with pytest.raises(ShapeMismatchError):
            Metadata.from_factors(factors)

    def test_labels_length_mismatch_raises(self):
        factors = {"a": np.array([0, 1, 2])}
        with pytest.raises(ShapeMismatchError):
            Metadata.from_factors(factors, np.array([0, 1]))

    def test_item_indices_custom(self):
        # OD-style: multiple detections mapping to fewer source images
        factors = {"conf_bin": np.array([0, 1, 2, 0, 1])}
        labels = np.array([0, 1, 0, 1, 0])
        item_indices = np.array([0, 0, 1, 1, 2])
        md = Metadata.from_factors(factors, labels, item_indices=item_indices)
        np.testing.assert_array_equal(md.item_indices, item_indices)
        assert md.factor_data.shape == (5, 1)

    @pytest.mark.parametrize("evaluator", [Balance, Diversity, Parity])
    def test_end_to_end_bias_evaluators(self, evaluator):
        rng = np.random.default_rng(1)
        n = 200
        factors = {
            "a": rng.integers(0, 3, size=n),
            "b": rng.integers(0, 4, size=n),
        }
        labels = rng.integers(0, 2, size=n)
        md = Metadata.from_factors(factors, labels)
        result = evaluator().evaluate(md)
        assert result is not None
