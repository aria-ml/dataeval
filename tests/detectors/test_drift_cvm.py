"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

import numpy as np
import pytest

from dataeval.detectors.drift._cvm import DriftCVM

np.random.seed(0)


@pytest.mark.required
class TestCVMDrift:
    n, n_test = 500, 200
    n_features = [2]  # TODO - test 1D case once BaseUnivariateDrift updated
    tests_cvmdrift = list(n_features)
    n_tests = len(tests_cvmdrift)

    @pytest.fixture(scope="class")
    def cvmdrift_params(self, request):
        return self.tests_cvmdrift[request.param]

    @pytest.mark.parametrize("cvmdrift_params", list(range(n_tests)), indirect=True)
    def test_cvmdrift(self, cvmdrift_params):
        n_feat = cvmdrift_params

        # Reference data
        x_ref = np.random.normal(0, 1, size=(self.n, n_feat)).squeeze()  # squeeze to test vec input in 1D case

        # Instantiate detector
        cd = DriftCVM(x_ref=x_ref, p_val=0.05)

        # Test predict on reference data
        x_h0 = x_ref.copy()
        preds = cd.predict(x_h0)
        assert not preds.is_drift and (preds.p_vals >= cd.p_val).any()

        # Test predict on heavily drifted data
        x_h1 = np.random.normal(2, 2, size=(self.n, n_feat)).squeeze()
        preds = cd.predict(x_h1)
        assert preds.is_drift
        assert preds.distances.min() >= 0.0
