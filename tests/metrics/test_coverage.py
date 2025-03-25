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
    def test_base_plotting(self):
        images = np.zeros((20, 3, 28, 28), dtype=np.float64)
        images[1] += 0.3
        images[5] += 0.9
        images[7] += 0.8
        result = coverage(flatten(images), num_observations=10, percent=0.15)
        output = result.plot(images, 3)
        assert isinstance(output, Figure)
        images = np.ones((10, 28), dtype=np.intp)
        with pytest.raises(ValueError):
            result.plot(images, 3)


@pytest.mark.optional
class TestCoverageFunctional:
    def test_naive_answer(self):
        embs = np.zeros((100, 2))
        embs = np.concatenate((embs, np.ones((1, 2))))
        result = coverage(embs, "naive", num_observations=20)
        assert result.uncovered_indices[0] == 100
        assert result.critical_value_radii[100] == pytest.approx(1.41421356)

    def test_adaptive_answer(self):
        embs = np.zeros((100, 2))
        embs = np.concatenate((embs, np.ones((1, 2))))
        result = coverage(embs, "adaptive", num_observations=20)
        assert result.uncovered_indices[0] == 100
        assert result.critical_value_radii[100] == pytest.approx(1.41421356)
