import math
import warnings

import numpy as np
import pytest

from dataeval._internal.metrics.coverage import coverage


class TestCoverageUnit:
    def test_fails_with_invalid_radius_type(self):
        embs = np.zeros((100, 2))
        with pytest.raises(ValueError), warnings.catch_warnings():
            coverage(embs, radius_type="NOT_A_RADIUS_TYPE")  # type: ignore

    def test_n_too_small(self):
        embs = np.zeros((3, 2))
        with pytest.raises(ValueError), warnings.catch_warnings():
            coverage(embs, k=3)

    def test_naive(self):
        """Checks pvals, crit, rho are all acceptable values"""
        embs = np.zeros((3, 2))
        result = coverage(embs, "naive", k=1)
        assert abs(result.critical_value - math.sqrt(2 / 3) / math.sqrt(math.pi)) < 0.01

    def test_adaptive(self):
        """Checks pvals, crit, rho are all acceptable values"""
        embs = np.zeros((100, 2))
        result = coverage(embs, "adaptive", k=1)
        np.testing.assert_array_equal(result.radii, np.zeros(100))

    def test_high_dim_data(self):
        """High dimensional data should not affect calculations"""
        embs = np.random.random(size=(100, 16, 16))
        x = coverage(embs)
        x_flat = coverage(embs.reshape((100, -1)))

        assert x.critical_value == x_flat.critical_value
        np.testing.assert_array_equal(x.indices, x_flat.indices)
        np.testing.assert_array_equal(x.radii, x_flat.radii)


class TestCoverageFunctional:
    def test_naive_answer(self):
        embs = np.zeros((100, 2))
        embs = np.concatenate((embs, np.ones((1, 2))))
        result = coverage(embs, "naive", k=20)
        assert result.indices[0] == 100
        assert result.radii[100] == pytest.approx(1.41421356)

    def test_adaptive_answer(self):
        embs = np.zeros((100, 2))
        embs = np.concatenate((embs, np.ones((1, 2))))
        result = coverage(embs, "adaptive", k=20)
        assert result.indices[0] == 100
        assert result.radii[100] == pytest.approx(1.41421356)
