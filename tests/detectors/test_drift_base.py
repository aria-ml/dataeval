"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

from itertools import product
from unittest.mock import MagicMock

import numpy as np
import pytest

from dataeval.detectors.drift._base import BaseDriftUnivariate, preprocess_x
from dataeval.detectors.drift.updates import LastSeenUpdate, ReservoirSamplingUpdate


class TestUpdateReference:
    n = [3, 50]
    n_features = [1, 10]
    update_method = [LastSeenUpdate, ReservoirSamplingUpdate]
    tests_update = list(product(n, n_features, update_method))
    n_tests_update = len(tests_update)

    @pytest.fixture(scope="class")
    def update_params(self, request):
        return self.tests_update[request.param]

    @pytest.mark.parametrize("update_params", list(range(n_tests_update)), indirect=True)
    def test_update_reference(self, update_params):
        n, n_features, update_method = update_params
        n_ref = np.random.randint(1, n)
        n_test = np.random.randint(1, 2 * n)
        X_ref = np.random.rand(n_ref * n_features).reshape(n_ref, n_features)
        X = np.random.rand(n_test * n_features).reshape(n_test, n_features)
        update_method = update_method(n)
        X_ref_new = update_method(X_ref, X, n)

        assert X_ref_new.shape[0] <= n
        if isinstance(update_method, LastSeenUpdate):
            assert (X_ref_new[-1] == X[-1]).all()


def test_base_init_preprocess_fn_valueerror():
    with pytest.raises(ValueError):
        BaseDriftUnivariate(np.empty([]), preprocess_fn="NotCallable")  # type: ignore


def test_base_init_update_x_ref_valueerror():
    with pytest.raises(ValueError):
        BaseDriftUnivariate(np.empty([]), update_x_ref="invalid")  # type: ignore


def test_base_init_correction_valueerror():
    with pytest.raises(ValueError):
        BaseDriftUnivariate(np.empty([]), n_features=2, correction="invalid")  # type: ignore


def test_base_init_set_n_features():
    base = BaseDriftUnivariate(np.zeros(1), n_features=1)
    assert base.n_features == 1


def test_base_predict_correction_valueerror():
    base = BaseDriftUnivariate(np.zeros(1), n_features=1)
    mock_score = MagicMock()
    mock_score.return_value = (np.array(0.5), np.array(0.5))
    base.score = mock_score
    base.correction = "invalid"
    with pytest.raises(ValueError):
        base.predict(np.empty([]))


def test_base_preprocess_infer_features():
    base = BaseDriftUnivariate(np.zeros((3, 3)), preprocess_fn=lambda x: x)
    assert base.n_features == 3


def test_base_preprocess():
    base = BaseDriftUnivariate(np.zeros(3), n_features=1, preprocess_fn=lambda x: x)
    np.testing.assert_equal(base._preprocess(base._x_ref), np.zeros(3))
    np.testing.assert_equal(base._preprocess(np.ones(3)), np.ones(3))


class TestPreprocessDecorator:
    _x_refcount = 0
    _x: np.ndarray | None

    def _preprocess(self, x: np.ndarray) -> np.ndarray:
        return x

    @preprocess_x
    def recursive_preprocess(self, x: np.ndarray, n: int, depth: int) -> int:
        if n < depth:
            n = n + 1
            assert n == self._x_refcount
            assert self._x == x
            return self.recursive_preprocess(x, n, depth)
        else:
            return n

    def test_preprocess_decorator(self):
        result = self.recursive_preprocess(np.array([10]), 0, 10)
        assert result == 10
        assert self._x_refcount == 0
        assert not hasattr(self, "_x")
