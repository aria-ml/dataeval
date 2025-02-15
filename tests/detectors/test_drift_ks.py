"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from itertools import product

import numpy as np
import pytest

from dataeval.detectors.drift._ks import DriftKS
from dataeval.detectors.drift.updates import LastSeenUpdate, ReservoirSamplingUpdate


@pytest.mark.required
class TestKSDrift:
    n, n_hidden, n_classes = 200, 10, 5
    n_features = [1, 10]
    alternative = ["two-sided", "less", "greater"]
    correction = ["bonferroni", "fdr"]
    update_x_ref = [LastSeenUpdate(1000), ReservoirSamplingUpdate(1000)]
    tests_ksdrift = list(
        product(
            n_features,
            alternative,
            correction,
            update_x_ref,
        )
    )
    n_tests = len(tests_ksdrift)

    @pytest.fixture(scope="class")
    def ksdrift_params(self, request):
        return self.tests_ksdrift[request.param]

    @pytest.mark.parametrize("ksdrift_params", list(range(n_tests)), indirect=True)
    def test_ksdrift(self, ksdrift_params):
        (
            n_features,
            alternative,
            correction,
            update_x_ref,
        ) = ksdrift_params
        np.random.seed(0)
        x_ref = np.random.randn(self.n * n_features).reshape(self.n, n_features).astype(np.float32)

        cd = DriftKS(
            x_ref=x_ref,
            p_val=0.05,
            update_x_ref=update_x_ref,
            preprocess_fn=None,
            correction=correction,
            alternative=alternative,
        )
        x = x_ref.copy()
        preds = cd.predict(x)
        assert not preds.drifted
        assert cd.n == x.shape[0] + x_ref.shape[0]
        assert cd.x_ref.shape[0] == min(update_x_ref.n, x.shape[0] + x_ref.shape[0])  # type: ignore
        assert preds.feature_drift.shape[0] == cd.n_features
        assert (preds.feature_drift == (preds.p_vals < cd.p_val)).all()  # type: ignore
        assert preds.feature_threshold == cd.p_val

        np.random.seed(0)
        X_randn = np.random.randn(self.n * n_features).reshape(self.n, n_features).astype("float32")
        mu, sigma = 5, 5
        X_low = sigma * X_randn - mu
        X_high = sigma * X_randn + mu

        preds_high = cd.predict(X_high)
        if alternative != "less":
            assert preds_high.drifted

        preds_low = cd.predict(X_low)
        if alternative != "greater":
            assert preds_low.drifted

        assert preds_low.distances.min() >= 0.0  # type: ignore

        if correction == "bonferroni":
            assert preds_low.threshold == cd.p_val / cd.n_features
