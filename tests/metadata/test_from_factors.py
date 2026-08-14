"""Tests for Metadata.from_factors — building Metadata from raw factor arrays.

These cover the "minimal data" construction path: a user with only a factor
table (and optional class labels) can build a dataeval.Metadata without owning a
full MAITE image dataset.
"""

import numpy as np
import polars as pl
import pytest

from dataeval import Metadata
from dataeval.bias import Balance, Diversity, Parity
from dataeval.exceptions import ShapeMismatchError
from dataeval.protocols import MetadataLike
from dataeval.types import SourceIndex


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


@pytest.mark.required
class TestFromFactorsBuildsTheStore:
    """FE-6 issue 6.2. ``from_factors`` is the entry point for tabular and array users,
    and it must reach the same normalized store the dataset path does rather than a
    stacked frame that happens to answer the same questions.
    """

    def test_bare_arrays_build_one_level(self):
        """Nothing in bare arrays distinguishes an item from a label, so there is one level."""
        metadata = Metadata.from_factors({"a": np.arange(6.0), "b": np.arange(6)})
        assert metadata.levels == ("unit",)
        assert dict(metadata._store.counts) == {"unit": 6}
        assert not metadata._store.links, "a single level has no edges"

    def test_a_source_index_builds_both_levels_and_the_edge(self):
        """Values as well as shape: a placement that scatters them passes a count check."""
        index = [
            SourceIndex(0, None),
            SourceIndex(1, None),
            SourceIndex(0, 0),
            SourceIndex(0, 1),
            SourceIndex(1, 0),
        ]
        metadata = Metadata.from_factors({"m": np.arange(len(index), dtype=np.float64)}, source_index=index)
        assert dict(metadata._store.counts) == {"unit": 2, "instance": 3}
        assert ("instance", "unit") in metadata._store.links
        assert metadata._store.positions_from("instance", "unit").tolist() == [0, 0, 1]
        assert metadata._store.frame("unit")["unit_m"].to_list() == [0.0, 1.0]
        assert metadata._store.frame("instance")["instance_m"].to_list() == [2.0, 3.0, 4.0]

    def test_each_half_lands_at_its_own_level_only(self):
        """Deliberately not one-to-one, so the downward gather has something to get wrong.

        With one instance per unit the gather is the identity and any permutation of it
        passes; two instances under the first unit and one under the second makes the
        expected column a repeat that only a correct gather produces.
        """
        index = [SourceIndex(0, None), SourceIndex(1, None), SourceIndex(0, 0), SourceIndex(0, 1), SourceIndex(1, 0)]
        metadata = Metadata.from_factors({"m": np.arange(len(index), dtype=np.float64)}, source_index=index)
        assert "unit_m" in metadata._store.frame("unit").columns
        assert "unit_m" not in metadata._store.frame("instance").columns
        assert "instance_m" in metadata._store.frame("instance").columns
        # Read from the finer rows by gather rather than from a stored copy: unit 0's value
        # appears on both of its instances, unit 1's on its one.
        assert metadata._store.column("instance", "unit_m").to_list() == [0.0, 0.0, 1.0]

    def test_the_flat_frame_is_derived_from_the_store(self):
        """Heights alone would pass for a frame built by concatenation independent of the
        store, which is the implementation this class exists to rule out. Compare values,
        and check that a write to the store shows up in the frame.
        """
        index = [SourceIndex(0, None), SourceIndex(0, 0), SourceIndex(0, 1)]
        metadata = Metadata.from_factors({"m": np.arange(len(index), dtype=np.float64)}, source_index=index)
        flat = metadata.dataframe
        assert flat.height == sum(metadata._store.counts.values())
        for level in metadata.levels:
            rows = metadata.rows_at(level)
            assert rows.height == metadata._store.height(level)
            assert rows.equals(flat.filter(pl.col("level") == level)), level
        # Derived, not stored: a later write reaches the frame without anything rebuilding it.
        metadata.add_factors({"late": np.arange(metadata.level_counts["unit"], dtype=np.float64)}, level="unit")
        assert "late" not in flat.columns
        assert "late" in metadata.dataframe.columns
