import math
import warnings

import numpy as np
import pytest

from dataeval.metrics import Coverage


class TestCoverageUnit:
    def test_fails_with_invalid_radius_type(self):
        embs = np.zeros((100, 2))
        with pytest.raises(ValueError), warnings.catch_warnings():
            metric = Coverage(embs, "new")  # type: ignore
            metric.evaluate()

    def test_n_too_small(self):
        embs = np.zeros((3, 2))
        with pytest.raises(ValueError), warnings.catch_warnings():
            metric = Coverage(embs, "naive")
            metric.evaluate()

    def test_naive(self):
        embs = np.zeros((3, 2))
        metric = Coverage(embs, "naive", k=1)
        metric.evaluate()
        assert abs(metric.rho - math.sqrt(2 / 3) / math.sqrt(math.pi)) < 0.01

    def test_adaptive(self):
        embs = np.zeros((100, 2))
        metric = Coverage(embs, "adaptive", k=1)
        _, crit = metric.evaluate()
        assert (crit == np.zeros(100)).all()


class TestCoverageFunctional:
    def test_naive_answer(self):
        embs = np.zeros((100, 2))
        embs = np.concatenate((embs, np.ones((1, 2))))
        metric = Coverage(embs, "naive", k=20)
        pvals, dists = metric.evaluate()
        assert pvals[0] == 100
        assert dists[100] == pytest.approx(1.41421356)

    def test_adaptive_answer(self):
        embs = np.zeros((100, 2))
        embs = np.concatenate((embs, np.ones((1, 2))))
        metric = Coverage(embs, "adaptive", k=20)
        pvals, dists = metric.evaluate()
        assert pvals[0] == 100
        assert dists[100] == pytest.approx(1.41421356)
