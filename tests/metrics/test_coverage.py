import math
import warnings

import numpy as np
import pytest

from dataeval.metrics import coverage


class TestCoverageUnit:
    def test_fails_with_invalid_radius_type(self):
        embs = np.zeros((100, 2))
        with pytest.raises(ValueError), warnings.catch_warnings():
            coverage(embs, radius_type="new")  # type: ignore

    def test_n_too_small(self):
        embs = np.zeros((3, 2))
        with pytest.raises(ValueError), warnings.catch_warnings():
            coverage(embs, radius_type="naive")

    def test_naive(self):
        embs = np.zeros((3, 2))
        result = coverage(embs, "naive", k=1)
        assert abs(result[2] - math.sqrt(2 / 3) / math.sqrt(math.pi)) < 0.01

    def test_adaptive(self):
        embs = np.zeros((100, 2))
        _, crit, _ = coverage(embs, "adaptive", k=1)
        assert (crit == np.zeros(100)).all()


class TestCoverageFunctional:
    def test_naive_answer(self):
        embs = np.zeros((100, 2))
        embs = np.concatenate((embs, np.ones((1, 2))))
        pvals, dists, _ = coverage(embs, "naive", k=20)
        assert pvals[0] == 100
        assert dists[100] == pytest.approx(1.41421356)

    def test_adaptive_answer(self):
        embs = np.zeros((100, 2))
        embs = np.concatenate((embs, np.ones((1, 2))))
        pvals, dists, _ = coverage(embs, "adaptive", k=20)
        assert pvals[0] == 100
        assert dists[100] == pytest.approx(1.41421356)
