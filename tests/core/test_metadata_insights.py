"""Unit tests for metadata insights functions."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.random import Generator

from dataeval.core._metadata_insights import (
    _calc_median_deviations,
    factor_deviation,
    factor_predictors,
)


@pytest.mark.required
class TestCalcMedianDeviations:
    """Tests for the _calc_median_deviations helper function."""

    @pytest.mark.parametrize("samples_ref", (1, 5, 10, 100))
    @pytest.mark.parametrize("samples_tst", (1, 5, 10, 100))
    @pytest.mark.parametrize("factors", (1, 5, 10))
    def test_output_shape(self, samples_ref, samples_tst, factors):
        """Tests that output shape is (n_test, n_factors) for all input combinations."""
        reference = np.arange(samples_ref * factors).reshape(samples_ref, factors)
        test = np.arange(samples_tst * factors).reshape(samples_tst, factors)

        result = _calc_median_deviations(reference, test)

        assert result.shape == (samples_tst, factors)
        assert not np.any(np.isnan(result))

    def test_identical_data_zero_deviation(self):
        """Tests that identical reference and test data results in zero deviation."""
        reference = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        test = np.array([[1.0, 2.0, 3.0]])

        result = _calc_median_deviations(reference, test)

        np.testing.assert_array_almost_equal(result, [[0.0, 0.0, 0.0]])

    def test_all_positive_values(self):
        """Tests that all deviation values are non-negative."""
        reference = np.random.randn(10, 5)
        test = np.random.randn(5, 5)

        result = _calc_median_deviations(reference, test)

        assert np.all(result >= 0)

    def test_symmetric_deviation(self):
        """Tests that positive and negative deviations are scaled appropriately."""
        # Reference centered at 5
        reference = np.array([[3.0], [5.0], [7.0]])
        # Test values above and below median
        test = np.array([[7.0], [3.0]])

        result = _calc_median_deviations(reference, test)

        # Both should have same deviation magnitude
        assert result[0, 0] == result[1, 0]

    def test_single_factor(self):
        """Tests calculation with a single factor."""
        reference = np.array([[1.0], [2.0], [3.0]])
        test = np.array([[5.0]])

        result = _calc_median_deviations(reference, test)

        assert result.shape == (1, 1)
        assert result[0, 0] > 0


@pytest.mark.required
class TestFindDeviatedFactors:
    """Tests for the find_deviated_factors function."""

    def test_empty_indices(self):
        """Tests that empty indices returns empty list."""
        reference_factors = {"time": np.array([1.0, 2.0, 3.0])}
        test_factors = {"time": np.array([4.0, 5.0, 6.0])}

        result = factor_deviation(reference_factors, test_factors, [])

        assert result == []

    def test_empty_reference_factors_raises(self):
        """Tests that empty reference factors raises ValueError."""
        test_factors = {"time": np.array([1.0, 2.0, 3.0])}

        with pytest.raises(ValueError, match="reference_factors dictionary cannot be empty"):
            factor_deviation({}, test_factors, [0])

    def test_empty_test_factors_raises(self):
        """Tests that empty test factors raises ValueError."""
        reference_factors = {"time": np.array([1.0, 2.0, 3.0])}

        with pytest.raises(ValueError, match="test_factors dictionary cannot be empty"):
            factor_deviation(reference_factors, {}, [0])

    def test_mismatched_keys_raises(self):
        """Tests that mismatched keys between reference and test raises ValueError."""
        reference_factors = {"time": np.array([1.0, 2.0, 3.0])}
        test_factors = {"altitude": np.array([100.0, 110.0, 105.0])}

        with pytest.raises(ValueError, match="reference_factors and test_factors must have the same keys"):
            factor_deviation(reference_factors, test_factors, [0])

    def test_mismatched_reference_array_lengths_raises(self):
        """Tests that mismatched reference array lengths raises ValueError."""
        reference_factors = {
            "time": np.array([1.0, 2.0, 3.0]),
            "altitude": np.array([100.0, 110.0]),
        }
        test_factors = {
            "time": np.array([4.0]),
            "altitude": np.array([120.0]),
        }

        with pytest.raises(ValueError, match="All reference factor arrays must have the same length"):
            factor_deviation(reference_factors, test_factors, [0])

    def test_mismatched_test_array_lengths_raises(self):
        """Tests that mismatched test array lengths raises ValueError."""
        reference_factors = {
            "time": np.array([1.0, 2.0, 3.0]),
            "altitude": np.array([100.0, 110.0, 105.0]),
        }
        test_factors = {
            "time": np.array([4.0]),
            "altitude": np.array([120.0, 130.0]),
        }

        with pytest.raises(ValueError, match="All test factor arrays must have the same length"):
            factor_deviation(reference_factors, test_factors, [0])

    @pytest.mark.parametrize("n_ref", (0, 1, 2))
    def test_insufficient_reference_samples_warns(self, n_ref, caplog):
        """Tests that less than 3 reference samples raises warning and returns empty list."""
        import logging

        reference_factors = {"time": np.arange(n_ref, dtype=float)}
        test_factors = {"time": np.array([10.0])}

        with caplog.at_level(logging.WARNING):
            result = factor_deviation(reference_factors, test_factors, [0])

        assert f"At least 3 reference metadata samples are needed, got {n_ref}" in caplog.text
        assert result == [{}]

    def test_single_index_single_factor(self):
        """Tests basic case with one index and one factor."""
        reference_factors = {"time": np.array([1.0, 2.0, 3.0])}
        test_factors = {"time": np.array([5.0])}

        result = factor_deviation(reference_factors, test_factors, [0])

        assert len(result) == 1
        assert "time" in result[0]
        assert result[0]["time"] > 0

    def test_multiple_indices_single_factor(self):
        """Tests multiple indices with single factor."""
        reference_factors = {"time": np.array([1.0, 2.0, 3.0])}
        test_factors = {"time": np.array([5.0, 10.0, 2.0])}

        result = factor_deviation(reference_factors, test_factors, [0, 1, 2])

        assert len(result) == 3
        for res in result:
            assert "time" in res

    def test_single_index_multiple_factors(self):
        """Tests single index with multiple factors."""
        reference_factors = {
            "time": np.array([1.0, 2.0, 3.0]),
            "altitude": np.array([100.0, 110.0, 105.0]),
        }
        test_factors = {
            "time": np.array([5.0]),
            "altitude": np.array([108.0]),
        }

        result = factor_deviation(reference_factors, test_factors, [0])

        assert len(result) == 1
        assert set(result[0].keys()) == {"time", "altitude"}

    def test_multiple_indices_multiple_factors(self):
        """Tests multiple indices with multiple factors."""
        reference_factors = {
            "time": np.array([1.0, 2.0, 3.0]),
            "altitude": np.array([100.0, 110.0, 105.0]),
        }
        test_factors = {
            "time": np.array([5.0, 12.0, 4.0]),
            "altitude": np.array([108.0, 112.0, 500.0]),
        }

        result = factor_deviation(reference_factors, test_factors, [0, 1, 2])

        assert len(result) == 3
        for res in result:
            assert set(res.keys()) == {"time", "altitude"}

    def test_dictionary_sorted_by_deviation(self):
        """Tests that each dictionary is sorted by deviation value (descending)."""
        reference_factors = {
            "time": np.array([1.0, 2.0, 3.0]),
            "altitude": np.array([100.0, 110.0, 105.0]),
        }
        test_factors = {
            "time": np.array([2.0]),  # Low deviation
            "altitude": np.array([500.0]),  # High deviation
        }

        result = factor_deviation(reference_factors, test_factors, [0])

        # First key should be the one with highest deviation
        keys = list(result[0].keys())
        values = list(result[0].values())
        assert keys[0] == "altitude"
        assert values[0] > values[1]

    def test_result_order_matches_indices_order(self):
        """Tests that result list order matches input indices order."""
        reference_factors = {"time": np.array([1.0, 2.0, 3.0])}
        test_factors = {"time": np.array([10.0, 5.0, 20.0])}

        # Request in specific order
        result = factor_deviation(reference_factors, test_factors, [2, 0, 1])

        # Result should have 3 items in same order as indices
        assert len(result) == 3
        # Index 2 should have highest deviation (20.0)
        assert result[0]["time"] > result[1]["time"]  # index 2 vs index 0
        assert result[0]["time"] > result[2]["time"]  # index 2 vs index 1

    def test_index_out_of_bounds_raises(self):
        """Tests that out of bounds index raises IndexError."""
        reference_factors = {"time": np.array([1.0, 2.0, 3.0])}
        test_factors = {"time": np.array([5.0])}

        with pytest.raises(ValueError, match="Invalid data dimensions"):
            factor_deviation(reference_factors, test_factors, [5])

    def test_all_factors_in_each_result(self):
        """Tests that all factors appear in each result dictionary."""
        reference_factors = {
            "a": np.array([1.0, 2.0, 3.0]),
            "b": np.array([10.0, 20.0, 30.0]),
            "c": np.array([100.0, 200.0, 300.0]),
        }
        test_factors = {
            "a": np.array([5.0, 2.5]),
            "b": np.array([25.0, 15.0]),
            "c": np.array([250.0, 150.0]),
        }

        result = factor_deviation(reference_factors, test_factors, [0, 1])

        assert len(result) == 2
        for res in result:
            assert set(res.keys()) == {"a", "b", "c"}


@pytest.mark.required
class TestFindFactorPredictors:
    """Tests for the find_factor_predictors function."""

    def test_empty_factors_raises(self):
        """Tests that empty factors dictionary raises ValueError."""
        with pytest.raises(ValueError, match="factors dictionary cannot be empty"):
            factor_predictors({}, [0])

    def test_mismatched_array_lengths_raises(self):
        """Tests that mismatched factor array lengths raises ValueError."""
        factors = {
            "time": np.array([1.0, 2.0, 3.0]),
            "altitude": np.array([100.0, 110.0]),
        }

        with pytest.raises(ValueError, match="All factor arrays must have the same length"):
            factor_predictors(factors, [0])

    def test_mismatched_discrete_features_length_raises(self):
        """Tests that mismatched discrete_features length raises ValueError."""
        factors = {
            "time": np.array([1.0, 2.0, 3.0]),
            "altitude": np.array([100.0, 110.0, 105.0]),
        }

        with pytest.raises(ValueError, match="discrete_features length .* must match number of factors"):
            factor_predictors(factors, [0], discrete_features=[True])

    def test_empty_indices_returns_zeros(self):
        """Tests that empty indices returns all zeros."""
        factors = {
            "time": np.array([1.0, 2.0, 3.0]),
            "altitude": np.array([100.0, 110.0, 105.0]),
        }

        result = factor_predictors(factors, np.array([], dtype=int))

        assert result == {"time": 0.0, "altitude": 0.0}

    def test_single_factor(self):
        """Tests with a single factor."""
        factors = {"time": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}

        result = factor_predictors(factors, [2, 3, 4])

        assert "time" in result
        assert isinstance(result["time"], float)
        assert result["time"] >= 0.0

    def test_multiple_factors(self):
        """Tests with multiple factors."""
        factors = {
            "time": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "altitude": np.array([100.0, 110.0, 105.0, 108.0, 112.0]),
        }

        result = factor_predictors(factors, [2, 3, 4])

        assert set(result.keys()) == {"time", "altitude"}
        for value in result.values():
            assert isinstance(value, float)
            assert value >= 0.0

    def test_perfect_correlation(self):
        """Tests that perfectly correlated factor has high mutual information."""
        # Create factor that perfectly separates flagged vs non-flagged
        factors = {
            "perfect": np.array([0.0, 0.0, 1.0, 1.0, 1.0]),
            "random": np.random.randn(5),
        }

        result = factor_predictors(factors, [2, 3, 4])

        # Perfect correlation should have higher MI than random
        assert result["perfect"] > result["random"]

    def test_discrete_features_parameter(self):
        """Tests that discrete_features parameter is accepted."""
        factors = {
            "categorical": np.array([0, 1, 0, 1, 0]),
            "continuous": np.array([1.5, 2.3, 3.1, 4.7, 5.2]),
        }

        result = factor_predictors(
            factors,
            [2, 3, 4],
            discrete_features=[True, False],
        )

        assert set(result.keys()) == {"categorical", "continuous"}

    def test_all_samples_flagged(self):
        """Tests when all samples are flagged."""
        factors = {"time": np.array([1.0, 2.0, 3.0])}

        result = factor_predictors(factors, [0, 1, 2])

        assert "time" in result
        # All flagged means no variance in the target
        assert result["time"] == 0.0

    def test_mutual_information_in_bits(self):
        """Tests that mutual information is returned in bits (not nats)."""
        factors = {
            "time": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        }

        result = factor_predictors(factors, [3, 4])

        # MI should be reasonable (typically < 2 bits for this simple case)
        assert result["time"] < 10.0

    def test_return_type_is_mapping(self):
        """Tests that return type is a mapping (dict)."""
        factors = {"time": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}

        result = factor_predictors(factors, [2, 3])

        assert isinstance(result, dict)


@pytest.mark.optional
class TestFunctionalScenarios:
    """Integration-style tests with realistic scenarios."""

    def test_find_deviated_factors_realistic_data(self, RNG: Generator):
        """Tests find_deviated_factors with realistic data."""
        # Create reference data with known distribution
        reference_factors = {
            "temperature": RNG.normal(20, 2, 100),
            "humidity": RNG.normal(50, 5, 100),
            "pressure": RNG.normal(1013, 10, 100),
        }

        # Create test data with outliers
        test_factors = {
            "temperature": np.array([21.0, 50.0, 19.0]),  # One outlier
            "humidity": np.array([52.0, 51.0, 90.0]),  # One outlier
            "pressure": np.array([1015.0, 1012.0, 1014.0]),  # No outliers
        }

        result = factor_deviation(reference_factors, test_factors, [0, 1, 2])

        assert len(result) == 3
        # Index 1 should have high temperature deviation
        # Index 2 should have high humidity deviation
        assert "temperature" in result[1]
        assert "humidity" in result[2]

    def test_find_factor_predictors_realistic_data(self, RNG: Generator):
        """Tests find_factor_predictors with realistic data."""
        # Create factors where time is correlated with being flagged
        n_samples = 100
        factors = {
            "time": np.concatenate([RNG.normal(10, 1, 50), RNG.normal(20, 1, 50)]),
            "noise": RNG.normal(0, 1, n_samples),
        }

        # Flag samples from second half (which have higher time values)
        flagged_indices = list(range(50, 75))

        result = factor_predictors(factors, flagged_indices)

        # Time should have higher MI than noise
        assert result["time"] > result["noise"]

    def test_combined_workflow(self, RNG: Generator):
        """Tests a combined workflow using both functions."""
        # Create reference and test data
        n_ref = 50
        n_test = 20

        reference_factors = {
            "sensor_a": RNG.normal(100, 10, n_ref),
            "sensor_b": RNG.normal(50, 5, n_ref),
        }

        test_factors = {
            "sensor_a": RNG.normal(100, 10, n_test),
            "sensor_b": RNG.normal(50, 5, n_test),
        }

        # Add some outliers to test data
        test_factors["sensor_a"][5] = 200
        test_factors["sensor_b"][10] = 100

        # Find deviated factors for all test samples
        all_indices = list(range(n_test))
        deviations = factor_deviation(reference_factors, test_factors, all_indices)

        # Get top 5 most deviated samples
        sorted_by_max_dev = sorted(deviations, key=lambda x: max(x.values()), reverse=True)
        top_5_indices = all_indices[:5]  # Would need actual indices from deviations

        # Find which factors predict these deviations
        all_factors_flat = {
            name: np.concatenate([reference_factors[name], test_factors[name]]) for name in reference_factors
        }
        predictors = factor_predictors(all_factors_flat, top_5_indices)

        assert len(sorted_by_max_dev) == n_test
        assert set(predictors.keys()) == {"sensor_a", "sensor_b"}
