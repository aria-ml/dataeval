import math

import numpy as np
import pytest

from dataeval.core._coverage import coverage_adaptive, coverage_naive


@pytest.mark.required
class TestCoverageUnit:
    def test_n_too_small(self):
        embs = np.zeros((3, 2))
        with pytest.raises(ValueError, match="less than or equal to"):
            coverage_naive(embs, num_observations=3)

    def test_naive(self):
        """Checks pvals, crit, rho are all acceptable values."""
        embs = np.zeros((3, 2))
        result = coverage_naive(embs, num_observations=1)
        assert abs(result["coverage_radius"] - math.sqrt(2 / 3) / math.sqrt(math.pi)) < 0.01

    def test_adaptive(self):
        """Checks pvals, crit, rho are all acceptable values."""
        embs = np.zeros((100, 2))
        result = coverage_adaptive(embs, num_observations=1, percent=0.01)
        np.testing.assert_array_equal(result["critical_value_radii"], np.zeros(100))

    def test_high_dim_data_valueerror(self):
        """High dimensional data should raise valueerror."""
        embs = np.random.random(size=(100, 16, 16))
        with pytest.raises(ValueError, match="expected 2"):
            coverage_naive(embs, 20)

    def test_non_unit_interval(self):
        embs = np.random.random(size=(16, 16)) * 2
        with pytest.raises(ValueError, match="must be unit"):
            coverage_naive(embs, 20)


@pytest.mark.optional
class TestCoverageFunctional:
    def test_naive_answer_edge(self):
        embs = np.zeros((101, 2))
        embs[-1] = 1
        result = coverage_naive(embs, num_observations=100)
        # all indices should be uncovered
        assert len(result["uncovered_indices"]) == 101
        assert result["uncovered_indices"][-1] == 100
        assert result["critical_value_radii"][0] == pytest.approx(1.41421356)
        assert result["critical_value_radii"][100] == pytest.approx(1.41421356)

    def test_adaptive_answer_edge(self):
        embs = np.zeros((101, 2))
        embs[-1] = 1
        result = coverage_adaptive(embs, num_observations=100, percent=0.01)
        # because the adaptive only returns the top k percent of results
        # and the default is 1% only one indice is uncovered even though
        # all indices have the same value and the 100 index is returned
        # based on the way the values are sorted
        assert len(result["uncovered_indices"]) == 1
        assert result["uncovered_indices"][0] == 100
        assert result["critical_value_radii"][0] == pytest.approx(1.41421356)
        assert result["critical_value_radii"][100] == pytest.approx(1.41421356)

    def test_naive_answer(self):
        embs = np.zeros((101, 2))
        embs[-1] = 1
        result = coverage_naive(embs, num_observations=20)
        assert result["uncovered_indices"][0] == 100
        assert result["critical_value_radii"][100] == pytest.approx(1.41421356)

    def test_adaptive_answer(self):
        embs = np.zeros((101, 2))
        embs[-1] = 1
        result = coverage_adaptive(embs, num_observations=20, percent=0.01)
        assert result["uncovered_indices"][0] == 100
        assert result["critical_value_radii"][100] == pytest.approx(1.41421356)
