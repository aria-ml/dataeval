"""Unit tests for dataeval.core._parity module."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from dataeval.core._parity import parity


@pytest.mark.required
class TestParity:
    """Test cases for the parity function."""

    def test_basic_functionality(self):
        """Test basic parity calculation with valid inputs."""
        binned_data = np.array([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.intp)
        class_labels = np.array([0, 0, 1, 1], dtype=np.intp)

        result = parity(binned_data, class_labels)
        chi_scores = result["chi_scores"]
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

        assert result["chi_scores"].shape == (1,)
        assert result["p_values"].shape == (1,)

    def test_perfect_correlation(self):
        """Test with perfectly correlated factor and labels."""
        data = [0] * 5 + [1] * 5
        binned_data = np.array([data], dtype=np.intp).T
        class_labels = np.array(data, dtype=np.intp)

        result = parity(binned_data, class_labels)

        assert result["chi_scores"][0] > 0
        assert result["p_values"][0] < 0.05

    def test_no_correlation(self):
        """Test with uncorrelated factor and labels."""
        binned_data = np.array([[0], [0], [0], [0]], dtype=np.intp)
        class_labels = np.array([0, 0, 1, 1], dtype=np.intp)

        result = parity(binned_data, class_labels)

        assert_array_almost_equal(result["chi_scores"], [0.0])
        assert_array_almost_equal(result["p_values"], [1.0])

    def test_return_insufficient_data_false(self):
        """Test return type when return_insufficient_data=False."""
        binned_data = np.array([[0], [1]], dtype=np.intp)
        class_labels = np.array([0, 1], dtype=np.intp)

        result = parity(binned_data, class_labels, return_insufficient_data=False)

        assert len(result) == 2
        assert isinstance(result["chi_scores"], np.ndarray)
        assert isinstance(result["p_values"], np.ndarray)

    def test_return_insufficient_data_true(self):
        """Test return type when return_insufficient_data=True."""
        binned_data = np.array([[0], [1]], dtype=np.intp)
        class_labels = np.array([0, 1], dtype=np.intp)

        result = parity(binned_data, class_labels, return_insufficient_data=True)

        assert len(result) == 3
        chi_scores = result["chi_scores"]
        p_values = result["p_values"]
        insufficient_data = result["insufficient_data"]
        assert isinstance(chi_scores, np.ndarray)
        assert isinstance(p_values, np.ndarray)
        assert isinstance(insufficient_data, dict)

    def test_insufficient_data_detection(self):
        """Test detection of cells with counts < 5."""
        binned_data = np.array([[0], [1], [0]], dtype=np.intp)
        class_labels = np.array([0, 1, 0], dtype=np.intp)

        result = parity(binned_data, class_labels, return_insufficient_data=True)
        insufficient_data = result["insufficient_data"]

        assert 0 in insufficient_data
        assert len(insufficient_data[0]) > 0

    def test_multiple_factors(self):
        """Test with multiple factors."""
        binned_data = np.array([[0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0]], dtype=np.intp)
        class_labels = np.array([0, 1, 0, 1], dtype=np.intp)

        result = parity(binned_data, class_labels)

        assert result["chi_scores"].shape == (3,)
        assert result["p_values"].shape == (3,)

    def test_zero_row_removal(self):
        """Test that zero-only rows are properly handled."""
        # Create data where one factor value never appears
        binned_data = np.array([[0], [0], [0], [0]], dtype=np.intp)
        class_labels = np.array([0, 0, 1, 1], dtype=np.intp)

        result = parity(binned_data, class_labels)

        # Should not raise error despite having only one factor value
        assert len(result["chi_scores"]) == 1
        assert len(result["p_values"]) == 1

    def test_single_class(self):
        """Test with only one class label."""
        binned_data = np.array([[0], [1], [0], [1]], dtype=np.intp)
        class_labels = np.array([0, 0, 0, 0], dtype=np.intp)

        result = parity(binned_data, class_labels)

        assert_array_almost_equal(result["chi_scores"], [0.0])
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
        """Test with large factor values."""
        binned_data = np.array([[100], [200], [100], [200]], dtype=np.intp)
        class_labels = np.array([0, 1, 0, 1], dtype=np.intp)

        result = parity(binned_data, class_labels)

        assert result["chi_scores"][0] > 0
        assert 0 <= result["p_values"][0] <= 1

    def test_dtype_consistency(self):
        """Test that output arrays have correct dtypes."""
        binned_data = np.array([[0], [1]], dtype=np.intp)
        class_labels = np.array([0, 1], dtype=np.intp)

        result = parity(binned_data, class_labels)

        assert result["chi_scores"].dtype == np.float64
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

        assert_array_almost_equal(result1["chi_scores"], result2["chi_scores"])
        assert_array_almost_equal(result2["chi_scores"], result3["chi_scores"])
        assert_array_almost_equal(result1["p_values"], result2["p_values"])
        assert_array_almost_equal(result2["p_values"], result3["p_values"])
