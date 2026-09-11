"""Unit tests for dataeval.core._parity module."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from dataeval.core._parity import parity

pytestmark = pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning")


@pytest.mark.required
class TestParity:
    """Test cases for the parity function."""

    def test_basic_functionality(self):
        """Test basic parity calculation with valid inputs."""
        binned_data = np.array([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.intp)
        class_labels = np.array([0, 0, 1, 1], dtype=np.intp)

        result = parity(binned_data, class_labels)
        chi_scores = result["scores"]
        p_values = result["p_values"]

        assert isinstance(chi_scores, np.ndarray)
        assert isinstance(p_values, np.ndarray)
        assert chi_scores.shape == (2,)
        assert p_values.shape == (2,)
        assert chi_scores.dtype == np.float64
        assert p_values.dtype == np.float64

    def test_single_factor(self):
        """Test with single factor column."""
        binned_data = np.array([[0], [1], [0], [1]], dtype=np.intp)
        class_labels = np.array([0, 0, 1, 1], dtype=np.intp)

        result = parity(binned_data, class_labels)

        assert result["scores"].shape == (1,)
        assert result["p_values"].shape == (1,)

    def test_high_correlation(self):
        """Test with highly correlated factor and labels.

        Note: Uses Bergsma bias-corrected Cramér's V (G-test), which adjusts
        for positive bias in small samples. For n=10 with perfect association,
        the corrected value is ~0.8.
        """
        data = [0] * 5 + [1] * 5
        binned_data = np.array([data], dtype=np.intp).T
        class_labels = np.array(data, dtype=np.intp)

        result = parity(binned_data, class_labels)

        # Bias-corrected value for n=10, perfect association in 2x2 table
        assert 0.8 <= result["scores"][0] <= 0.85
        assert result["p_values"][0] < 0.05

    def test_no_correlation(self):
        """Test with uncorrelated factor and labels."""
        binned_data = np.array([[0], [0], [0], [0]], dtype=np.intp)
        class_labels = np.array([0, 0, 1, 1], dtype=np.intp)

        result = parity(binned_data, class_labels)

        assert_array_almost_equal(result["scores"], [0.0])
        assert_array_almost_equal(result["p_values"], [1.0])

    def test_return_types(self):
        """Test return types."""
        binned_data = np.array([[0], [1]], dtype=np.intp)
        class_labels = np.array([0, 1], dtype=np.intp)

        result = parity(binned_data, class_labels)

        assert len(result) == 3
        chi_scores = result["scores"]
        p_values = result["p_values"]
        insufficient_data = result["insufficient_data"]
        assert isinstance(chi_scores, np.ndarray)
        assert isinstance(p_values, np.ndarray)
        assert isinstance(insufficient_data, dict)

    def test_insufficient_data_detection(self):
        """Test detection of cells with counts < 5."""
        binned_data = np.array([[0], [1], [0]], dtype=np.intp)
        class_labels = np.array([0, 1, 0], dtype=np.intp)

        result = parity(binned_data, class_labels)
        insufficient_data = result["insufficient_data"]

        assert 0 in insufficient_data
        assert len(insufficient_data[0]) > 0

    def test_multiple_factors(self):
        """Test with multiple factors."""
        binned_data = np.array([[0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0]], dtype=np.intp)
        class_labels = np.array([0, 1, 0, 1], dtype=np.intp)

        result = parity(binned_data, class_labels)

        assert result["scores"].shape == (3,)
        assert result["p_values"].shape == (3,)

    def test_zero_row_removal(self):
        """Test that zero-only rows are properly handled."""
        # Create data where one factor value never appears
        binned_data = np.array([[0], [0], [0], [0]], dtype=np.intp)
        class_labels = np.array([0, 0, 1, 1], dtype=np.intp)

        result = parity(binned_data, class_labels)

        # Should not raise error despite having only one factor value
        assert len(result["scores"]) == 1
        assert len(result["p_values"]) == 1

    def test_single_sample_zero_score(self):
        """A single sample yields a contingency total of 1 (n <= 1), giving a 0.0 score."""
        binned_data = np.array([[0]], dtype=np.intp)
        class_labels = np.array([0], dtype=np.intp)

        result = parity(binned_data, class_labels)

        assert result["scores"][0] == 0.0

    def test_single_class(self):
        """Test with only one class label."""
        binned_data = np.array([[0], [1], [0], [1]], dtype=np.intp)
        class_labels = np.array([0, 0, 0, 0], dtype=np.intp)

        result = parity(binned_data, class_labels)

        assert_array_almost_equal(result["scores"], [0.0])
        assert_array_almost_equal(result["p_values"], [1.0])

    def test_empty_arrays(self):
        """Test behavior with empty arrays."""
        binned_data = np.array([], dtype=np.intp).reshape(0, 1)
        class_labels = np.array([], dtype=np.intp)

        with pytest.raises((ValueError, IndexError)):
            parity(binned_data, class_labels)

    def test_mismatched_lengths(self):
        """Test error handling for mismatched input lengths."""
        binned_data = np.array([[0], [1]], dtype=np.intp)
        class_labels = np.array([0, 1, 2], dtype=np.intp)  # Different length

        with pytest.raises((ValueError, IndexError)):
            parity(binned_data, class_labels)

    def test_large_factor_values(self):
        """Test with large factor values.

        Note: Uses larger sample size (n=20) to avoid bias correction
        reducing the score to 0 in very small samples.
        """
        # Increase sample size to avoid bias correction zeroing out the score
        binned_data = np.array([[100] * 5 + [200] * 5], dtype=np.intp).T
        class_labels = np.array([0] * 5 + [1] * 5, dtype=np.intp)

        result = parity(binned_data, class_labels)

        assert result["scores"][0] > 0
        assert 0 <= result["p_values"][0] <= 1

    def test_dtype_consistency(self):
        """Test that output arrays have correct dtypes."""
        binned_data = np.array([[0], [1]], dtype=np.intp)
        class_labels = np.array([0, 1], dtype=np.intp)

        result = parity(binned_data, class_labels)

        assert result["scores"].dtype == np.float64
        assert result["p_values"].dtype == np.float64

    def test_class_labels_as_list(self):
        """Test with class_labels as different sequence types."""
        binned_data = np.array([[0], [1]], dtype=np.intp)

        # Test with list
        result1 = parity(binned_data, [0, 1])  # type: ignore

        # Test with tuple
        result2 = parity(binned_data, (0, 1))  # type: ignore

        # Test with numpy array
        result3 = parity(binned_data, np.array([0, 1]))

        assert_array_almost_equal(result1["scores"], result2["scores"])
        assert_array_almost_equal(result2["scores"], result3["scores"])
        assert_array_almost_equal(result1["p_values"], result2["p_values"])
        assert_array_almost_equal(result2["p_values"], result3["p_values"])


@pytest.mark.required
class TestParityEffectiveN:
    """The G-test's evidence is the entities behind the table, not always its rows."""

    @staticmethod
    def _replicated(seed, entities=60, fan_out=40):
        """One factor and a label axis, independent, each value repeated `fan_out` times."""
        rng = np.random.default_rng(seed)
        factor = rng.integers(0, 3, entities)
        labels = rng.integers(0, 3, entities)
        return (
            np.repeat(factor, fan_out)[:, None].astype(np.intp),
            np.repeat(labels, fan_out).astype(np.intp),
            factor[:, None].astype(np.intp),
            labels.astype(np.intp),
            entities,
        )

    def test_replication_alone_rejects_independence_without_it(self):
        """The defect this argument exists for, pinned so it cannot come back quietly.

        The statistic is linear in the table's total, so a fan-out of forty multiplies the
        evidence for a difference that is not there and the p-value collapses.
        """
        wide, wide_labels, _, _, _ = self._replicated(0)
        assert float(parity(wide, wide_labels)["p_values"][0]) < 1e-20

    def test_scaling_to_the_entities_restores_the_honest_test(self):
        """Both statistics land exactly where one row per entity puts them."""
        for seed in range(6):
            wide, wide_labels, narrow, narrow_labels, entities = self._replicated(seed)
            scaled = parity(wide, wide_labels, [entities, entities])
            honest = parity(narrow, narrow_labels)
            assert float(scaled["p_values"][0]) == pytest.approx(float(honest["p_values"][0]), abs=1e-12)
            assert float(scaled["scores"][0]) == pytest.approx(float(honest["scores"][0]), abs=1e-12)

    def test_none_leaves_every_output_where_it_was(self):
        """The ordinary single-level call is not to be disturbed by the argument existing."""
        rng = np.random.default_rng(0)
        factors = rng.integers(0, 3, (500, 2)).astype(np.intp)
        labels = rng.integers(0, 2, 500).astype(np.intp)
        without, with_none = parity(factors, labels), parity(factors, labels, None)
        assert_array_almost_equal(without["p_values"], with_none["p_values"])
        assert_array_almost_equal(without["scores"], with_none["scores"])

    def test_counts_at_or_above_the_rows_change_nothing(self):
        """A count that cannot be met is clamped, not applied."""
        rng = np.random.default_rng(0)
        factors = rng.integers(0, 3, (200, 1)).astype(np.intp)
        labels = rng.integers(0, 2, 200).astype(np.intp)
        assert float(parity(factors, labels, [200, 900])["p_values"][0]) == pytest.approx(
            float(parity(factors, labels)["p_values"][0])
        )

    def test_a_thin_cell_stays_flagged_once_it_is_scaled(self):
        """Replication is what hides a thin cell, so the flag has to read the scaled table."""
        factor = np.repeat(np.array([0, 0, 1, 1, 1, 1]), 40)[:, None].astype(np.intp)
        labels = np.repeat(np.array([0, 1, 0, 1, 0, 1]), 40).astype(np.intp)
        # Unscaled, every cell holds 40 or 80 and nothing is flagged.
        assert not parity(factor, labels)["insufficient_data"]
        flagged = parity(factor, labels, [6, 6])["insufficient_data"]
        assert flagged, "the six entities behind this table leave cells below five"
        counts = [c for cats in flagged.values() for classes in cats.values() for c in classes.values()]
        assert all(isinstance(count, float) for count in counts), "scaled counts are entities, not rows"

    def test_the_length_must_match_the_columns_plus_the_axis(self):
        """One entry per factor plus one for the labels, which is what the pairs index into."""
        rng = np.random.default_rng(0)
        factors = rng.integers(0, 3, (50, 2)).astype(np.intp)
        labels = rng.integers(0, 2, 50).astype(np.intp)
        with pytest.raises(ValueError, match="2 entries for 3 columns"):
            parity(factors, labels, [50, 50])
