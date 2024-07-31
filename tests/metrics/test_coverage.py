import math
import warnings

import numpy as np
import numpy.testing as npt
import pytest

from dataeval._internal.functional.coverage import sort_neighbors
from dataeval.metrics import Coverage


class TestAPICoverage:
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


class TestFunctionalCoverage:
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

    def test_naive_radius(self):
        pass

    def test_adaptive_radius(self):
        pass

    def test_sort_neighbors(self):
        embs = np.arange(100)
        k = 20
        crit = sort_neighbors(embs, k=k)

        npt.assert_array_equal(crit, np.arange(k + 1))

    def test_coverage(self):
        pass
