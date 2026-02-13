"""
Tests for ResultAggregator class.

These tests verify proper initialization, storage, and retrieval
of metrics across multiple runs and substeps.
"""

import numpy as np
from numpy.testing import assert_array_equal

from dataeval.performance._aggregator import ResultAggregator


class TestResultAggregatorInitialization:
    """Test aggregator initialization."""

    def test_initializes_with_runs_and_substeps(self):
        """Verify aggregator stores run and substep counts."""
        aggregator = ResultAggregator(runs=3, substeps=5)

        assert aggregator.runs == 3
        assert aggregator.substeps == 5

    def test_initializes_empty_storage(self):
        """Verify aggregator starts with empty storage."""
        aggregator = ResultAggregator(runs=3, substeps=5)

        assert aggregator.get_results() == {}


class TestResultAggregatorScalarMetrics:
    """Test accumulation of scalar metrics."""

    def test_stores_scalar_metric(self):
        """Verify storing single scalar value."""
        aggregator = ResultAggregator(runs=2, substeps=3)

        aggregator.add_result(run=0, step=0, metric_name="accuracy", value=0.95)

        results = aggregator.get_results()
        assert "accuracy" in results
        assert results["accuracy"].shape == (2, 3)
        assert results["accuracy"][0, 0] == 0.95

    def test_accumulates_multiple_scalar_values(self):
        """Verify accumulating scalar values across runs and steps."""
        aggregator = ResultAggregator(runs=2, substeps=2)

        # Run 0
        aggregator.add_result(0, 0, "accuracy", 0.5)
        aggregator.add_result(0, 1, "accuracy", 0.7)

        # Run 1
        aggregator.add_result(1, 0, "accuracy", 0.6)
        aggregator.add_result(1, 1, "accuracy", 0.8)

        results = aggregator.get_results()
        assert_array_equal(results["accuracy"], [[0.5, 0.7], [0.6, 0.8]])

    def test_handles_multiple_scalar_metrics(self):
        """Verify handling multiple different scalar metrics."""
        aggregator = ResultAggregator(runs=1, substeps=2)

        aggregator.add_result(0, 0, "accuracy", 0.9)
        aggregator.add_result(0, 0, "precision", 0.85)
        aggregator.add_result(0, 1, "accuracy", 0.95)
        aggregator.add_result(0, 1, "precision", 0.90)

        results = aggregator.get_results()
        assert "accuracy" in results
        assert "precision" in results
        assert results["accuracy"].shape == (1, 2)
        assert results["precision"].shape == (1, 2)


class TestResultAggregatorArrayMetrics:
    """Test accumulation of array-valued metrics (e.g., per-class)."""

    def test_stores_array_metric(self):
        """Verify storing array-valued metric (e.g., per-class accuracy)."""
        aggregator = ResultAggregator(runs=1, substeps=2)

        # Per-class accuracy for 3 classes
        aggregator.add_result(0, 0, "class_accuracy", np.array([0.9, 0.8, 0.7]))

        results = aggregator.get_results()
        assert "class_accuracy" in results
        assert results["class_accuracy"].shape == (1, 2, 3)  # runs × substeps × classes

    def test_accumulates_array_values(self):
        """Verify accumulating array metrics across runs."""
        aggregator = ResultAggregator(runs=2, substeps=1)

        # Run 0
        aggregator.add_result(0, 0, "per_class", np.array([1.0, 2.0]))

        # Run 1
        aggregator.add_result(1, 0, "per_class", np.array([3.0, 4.0]))

        results = aggregator.get_results()
        expected = np.array(
            [
                [[1.0, 2.0]],  # Run 0, step 0
                [[3.0, 4.0]],  # Run 1, step 0
            ],
        )
        assert_array_equal(results["per_class"], expected)

    def test_handles_mixed_scalar_and_array(self):
        """Verify handling both scalar and array metrics simultaneously."""
        aggregator = ResultAggregator(runs=1, substeps=1)

        aggregator.add_result(0, 0, "accuracy", 0.95)  # Scalar
        aggregator.add_result(0, 0, "per_class", np.array([0.9, 0.8]))  # Array

        results = aggregator.get_results()
        assert results["accuracy"].shape == (1, 1)
        assert results["per_class"].shape == (1, 1, 2)


class TestResultAggregatorAutoShapeDetection:
    """Test automatic detection of metric shapes."""

    def test_auto_detects_scalar_shape(self):
        """Verify shape detection for scalar metrics."""
        aggregator = ResultAggregator(runs=2, substeps=3)

        # First value determines shape
        aggregator.add_result(0, 0, "loss", 0.5)

        results = aggregator.get_results()
        # Scalar → shape (runs, substeps)
        assert results["loss"].shape == (2, 3)

    def test_auto_detects_array_shape(self):
        """Verify shape detection for array metrics."""
        aggregator = ResultAggregator(runs=2, substeps=3)

        # First value determines shape
        aggregator.add_result(0, 0, "scores", np.array([1.0, 2.0, 3.0, 4.0]))

        results = aggregator.get_results()
        # Array of length 4 → shape (runs, substeps, 4)
        assert results["scores"].shape == (2, 3, 4)

    def test_handles_1d_array_as_scalar(self):
        """Verify 1-element arrays treated as scalars."""
        aggregator = ResultAggregator(runs=1, substeps=2)

        # 1-element array should be treated as scalar
        aggregator.add_result(0, 0, "metric", np.array([0.95]))

        results = aggregator.get_results()
        # Should be (runs, substeps), not (runs, substeps, 1)
        assert results["metric"].shape == (1, 2)
        assert results["metric"][0, 0] == 0.95


class TestResultAggregatorEdgeCases:
    """Test edge cases and error conditions."""

    def test_handles_zero_values_gracefully(self):
        """Verify handling zero values doesn't break storage."""
        aggregator = ResultAggregator(runs=1, substeps=1)

        aggregator.add_result(0, 0, "loss", 0.0)

        results = aggregator.get_results()
        assert results["loss"][0, 0] == 0.0

    def test_handles_negative_values(self):
        """Verify handling negative values (e.g., for loss)."""
        aggregator = ResultAggregator(runs=1, substeps=1)

        aggregator.add_result(0, 0, "error", -0.5)

        results = aggregator.get_results()
        assert results["error"][0, 0] == -0.5
