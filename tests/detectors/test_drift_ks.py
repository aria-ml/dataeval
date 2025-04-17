"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from itertools import product
from unittest.mock import MagicMock

import numpy as np
import pytest

from dataeval.detectors.drift._ks import DriftKS
from dataeval.detectors.drift.updates import LastSeenUpdate, ReservoirSamplingUpdate
from dataeval.utils.data._embeddings import Embeddings


@pytest.mark.required
class TestKSDrift:
    n, n_hidden, n_classes = 200, 10, 5
    n_features = [1, 10]
    alternative = ["two-sided", "less", "greater"]
    correction = ["bonferroni", "fdr"]
    update_strategy = [LastSeenUpdate(1000), ReservoirSamplingUpdate(1000)]
    tests_ksdrift = list(
        product(
            n_features,
            alternative,
            correction,
            update_strategy,
        )
    )
    n_tests = len(tests_ksdrift)

    def get_embeddings(self, n: int = 100, n_features: int = 10) -> Embeddings:
        mock = MagicMock(sepc=Embeddings)
        mock._data = np.random.randn(n * n_features).reshape(n, n_features).astype(np.float32)
        mock.to_numpy.return_value = mock._data
        mock.__getitem__.side_effect = lambda idx: mock._data[idx]
        mock.__len__.return_value = n
        setattr(mock, "__class__", Embeddings)
        return mock

    @pytest.fixture(scope="class")
    def ksdrift_params(self, request):
        return self.tests_ksdrift[request.param]

    @pytest.mark.parametrize("ksdrift_params", list(range(n_tests)), indirect=True)
    def test_ksdrift(self, ksdrift_params):
        (
            n_features,
            alternative,
            correction,
            update_strategy,
        ) = ksdrift_params
        np.random.seed(0)
        data = self.get_embeddings(self.n, n_features)

        cd = DriftKS(
            data=data,
            p_val=0.05,
            update_strategy=update_strategy,
            correction=correction,
            alternative=alternative,
        )
        preds = cd.predict(data)
        assert not preds.drifted
        assert cd.n == self.n + self.n
        assert cd.x_ref.shape[0] == min(update_strategy.n, self.n + self.n)  # type: ignore
        assert preds.feature_drift.shape[0] == cd.n_features
        assert (preds.feature_drift == (preds.p_vals < cd.p_val)).all()  # type: ignore
        assert preds.feature_threshold == cd.p_val

        np.random.seed(0)
        X_randn = np.random.randn(self.n * n_features).reshape(self.n, n_features).astype("float32")
        mu, sigma = 5, 5
        X_low = MagicMock(spec=Embeddings)
        X_low.to_numpy.return_value = sigma * X_randn - mu
        X_high = MagicMock(spec=Embeddings)
        X_high.to_numpy.return_value = sigma * X_randn + mu

        preds_high = cd.predict(X_high)
        if alternative != "less":
            assert preds_high.drifted

        preds_low = cd.predict(X_low)
        if alternative != "greater":
            assert preds_low.drifted

        assert preds_low.distances.min() >= 0.0  # type: ignore

        if correction == "bonferroni":
            assert preds_low.threshold == cd.p_val / cd.n_features
