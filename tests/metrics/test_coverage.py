import math
import warnings

import numpy as np
import pytest

from dataeval.utils._array import flatten

try:
    from matplotlib.figure import Figure
except ImportError:
    Figure = type(None)

from dataeval.metrics.bias._coverage import coverage


@pytest.mark.required
class TestCoverageUnit:
    def test_fails_with_invalid_radius_type(self):
        embs = np.zeros((100, 2))
        with pytest.raises(ValueError), warnings.catch_warnings():
            coverage(embs, radius_type="NOT_A_RADIUS_TYPE")  # type: ignore

    def test_n_too_small(self):
        embs = np.zeros((3, 2))
        with pytest.raises(ValueError), warnings.catch_warnings():
            coverage(embs, num_observations=3)

    def test_naive(self):
        """Checks pvals, crit, rho are all acceptable values"""
        embs = np.zeros((3, 2))
        result = coverage(embs, "naive", num_observations=1)
        assert abs(result.coverage_radius - math.sqrt(2 / 3) / math.sqrt(math.pi)) < 0.01

    def test_adaptive(self):
        """Checks pvals, crit, rho are all acceptable values"""
        embs = np.zeros((100, 2))
        result = coverage(embs, "adaptive", num_observations=1)
        np.testing.assert_array_equal(result.critical_value_radii, np.zeros(100))

    def test_high_dim_data_valueerror(self):
        """High dimensional data should raise valueerror"""
        embs = np.random.random(size=(100, 16, 16))
        with pytest.raises(ValueError):
            coverage(embs)

    def test_non_unit_interval(self):
        embs = np.random.random(size=(100, 16, 16)) * 2
        with pytest.raises(ValueError):
            coverage(embs)


@pytest.mark.requires_all
class TestCoveragePlot:
    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_plot(self, n):
        images = np.zeros((20, 3, 28, 28), dtype=np.float64)
        images[1] += 0.3
        images[3] += 0.8
        images[5] += 0.9
        images[7] += 0.8
        images[9] += 0.4
        result = coverage(flatten(images), num_observations=10, percent=0.2)
        output = result.plot(images, n)
        assert isinstance(output, Figure)

    def test_plot_raises(self):
        images = np.zeros((20, 3, 28, 28), dtype=np.float64)
        result = coverage(flatten(images), num_observations=10, percent=0.15)
        bad_images = np.ones((5, 28), dtype=np.intp)
        with pytest.raises(ValueError):
            result.plot(bad_images, 3)


@pytest.mark.optional
class TestCoverageFunctional:
    def test_naive_answer_edge(self):
        embs = np.zeros((101, 2))
        embs[-1] = 1
        result = coverage(embs, "naive", num_observations=100)
        # all indices should be uncovered
        assert len(result.uncovered_indices) == 101
        assert result.uncovered_indices[-1] == 100
        assert result.critical_value_radii[0] == pytest.approx(1.41421356)
        assert result.critical_value_radii[100] == pytest.approx(1.41421356)

    def test_adaptive_answer_edge(self):
        embs = np.zeros((101, 2))
        embs[-1] = 1
        result = coverage(embs, "adaptive", num_observations=100)
        # because the adaptive only returns the top k percent of results
        # and the default is 1% only one indice is uncovered even though
        # all indices have the same value and the 100 index is returned
        # based on the way the values are sorted
        assert len(result.uncovered_indices) == 1
        assert result.uncovered_indices[0] == 100
        assert result.critical_value_radii[0] == pytest.approx(1.41421356)
        assert result.critical_value_radii[100] == pytest.approx(1.41421356)

    def test_naive_answer(self):
        embs = np.zeros((101, 2))
        embs[-1] = 1
        result = coverage(embs, "naive", num_observations=20)
        assert result.uncovered_indices[0] == 100
        assert result.critical_value_radii[100] == pytest.approx(1.41421356)

    def test_adaptive_answer(self):
        embs = np.zeros((101, 2))
        embs[-1] = 1
        result = coverage(embs, "adaptive", num_observations=20)
        assert result.uncovered_indices[0] == 100
        assert result.critical_value_radii[100] == pytest.approx(1.41421356)
