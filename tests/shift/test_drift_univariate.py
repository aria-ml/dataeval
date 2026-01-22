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

from dataeval._embeddings import Embeddings
from dataeval.shift._drift._univariate import DriftUnivariate
from dataeval.shift.update_strategies import LastSeenUpdateStrategy, ReservoirSamplingUpdateStrategy


@pytest.mark.required
class TestKSDrift:
    """Tests for Kolmogorov-Smirnov drift detection."""

    n, n_hidden, n_classes = 200, 10, 5
    n_features = [1, 10]
    alternative = ["two-sided", "less", "greater"]
    correction = ["bonferroni", "fdr"]
    update_strategy = [LastSeenUpdateStrategy(1000), ReservoirSamplingUpdateStrategy(1000)]
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
        mock = MagicMock(spec=Embeddings)
        mock._data = np.random.randn(n * n_features).reshape(n, n_features).astype(np.float32)
        mock.__getitem__.side_effect = lambda idx: mock._data[idx]
        mock.__len__.return_value = n
        mock.__array__.return_value = mock._data
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

        cd = DriftUnivariate(
            data=data,
            method="ks",
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
        X_low.__array__.return_value = sigma * X_randn - mu
        X_high = MagicMock(spec=Embeddings)
        X_high.__array__.return_value = sigma * X_randn + mu

        preds_high = cd.predict(X_high)
        if alternative != "less":
            assert preds_high.drifted

        preds_low = cd.predict(X_low)
        if alternative != "greater":
            assert preds_low.drifted

        assert preds_low.distances.min() >= 0.0  # type: ignore

        if correction == "bonferroni":
            assert preds_low.threshold == cd.p_val / cd.n_features


@pytest.mark.required
class TestCVMDrift:
    """Tests for Cramér-von Mises drift detection."""

    n, n_test = 500, 200

    def get_embeddings(self, n: int = 100, n_features: int = 10, value: float | None = None) -> Embeddings:
        arr = (
            np.random.random((n, n_features)).astype(np.float32)
            if value is None
            else np.full(shape=(n, n_features), fill_value=value, dtype=np.float32)
        )
        mock = MagicMock(spec=Embeddings)
        mock.__getitem__.side_effect = lambda idx: arr[idx]
        mock.__len__.return_value = n
        mock.__array__.return_value = arr
        return mock

    def test_cvmdrift(self):
        # Reference data
        ref_emb = self.get_embeddings(self.n, value=0.0)

        # Instantiate detector
        cd = DriftUnivariate(data=ref_emb, method="cvm", p_val=0.05, correction="fdr")

        # Test predict on reference data
        preds = cd.predict(ref_emb)
        assert not preds.drifted and (preds.p_vals >= cd.p_val).any()

        # Test predict on heavily drifted data
        x = self.get_embeddings(self.n_test, value=0.5)
        preds = cd.predict(x)
        assert preds.drifted
        assert preds.distances.min() >= 0.0


@pytest.mark.required
class TestMWUDrift:
    """Tests for Mann-Whitney U drift detection."""

    n, n_test = 500, 200

    def get_embeddings(self, n: int = 100, n_features: int = 10, value: float | None = None) -> Embeddings:
        arr = (
            np.random.random((n, n_features)).astype(np.float32)
            if value is None
            else np.full(shape=(n, n_features), fill_value=value, dtype=np.float32)
        )
        mock = MagicMock(spec=Embeddings)
        mock.__getitem__.side_effect = lambda idx: arr[idx]
        mock.__len__.return_value = n
        mock.__array__.return_value = arr
        return mock

    def test_mwu_drift(self):
        # Reference data
        ref_emb = self.get_embeddings(self.n, value=0.0)

        # Instantiate detector
        cd = DriftUnivariate(data=ref_emb, method="mwu", p_val=0.05, correction="fdr")

        # Test predict on reference data
        preds = cd.predict(ref_emb)
        assert not preds.drifted and (preds.p_vals >= cd.p_val).any()

        # Test predict on heavily drifted data
        x = self.get_embeddings(self.n_test, value=0.5)
        preds = cd.predict(x)
        assert preds.drifted
        assert preds.distances.min() >= 0.0


@pytest.mark.required
class TestAndersonDrift:
    """Tests for Anderson-Darling drift detection."""

    n, n_test = 500, 200

    def get_embeddings(self, n: int = 100, n_features: int = 10, value: float | None = None) -> Embeddings:
        arr = (
            np.random.random((n, n_features)).astype(np.float32)
            if value is None
            else np.full(shape=(n, n_features), fill_value=value, dtype=np.float32)
        )
        mock = MagicMock(spec=Embeddings)
        mock.__getitem__.side_effect = lambda idx: arr[idx]
        mock.__len__.return_value = n
        mock.__array__.return_value = arr
        return mock

    def test_anderson_drift(self):
        # Anderson-Darling requires variation in data, so use random values
        np.random.seed(42)
        ref_emb = self.get_embeddings(self.n, value=None)

        # Instantiate detector
        cd = DriftUnivariate(data=ref_emb, method="anderson", p_val=0.05, correction="fdr")

        # Test predict on reference data - should not detect drift
        np.random.seed(42)
        preds = cd.predict(ref_emb)
        assert not preds.drifted and (preds.p_vals >= cd.p_val).any()

        # Test predict on heavily drifted data (shifted distribution)
        np.random.seed(43)
        x_shifted = self.get_embeddings(self.n_test, value=None)[:]
        # Add shift to make drift detectable
        x_shifted += 2.0
        preds = cd.predict(x_shifted)
        assert preds.drifted
        assert preds.distances.min() >= 0.0


@pytest.mark.required
class TestBWSDrift:
    """Tests for Baumgartner-Weiss-Schindler drift detection."""

    n, n_test = 500, 200

    def get_embeddings(self, n: int = 100, n_features: int = 10, value: float | None = None) -> Embeddings:
        arr = (
            np.random.random((n, n_features)).astype(np.float32)
            if value is None
            else np.full(shape=(n, n_features), fill_value=value, dtype=np.float32)
        )
        mock = MagicMock(spec=Embeddings)
        mock.__getitem__.side_effect = lambda idx: arr[idx]
        mock.__len__.return_value = n
        mock.__array__.return_value = arr
        return mock

    def test_bws_drift(self):
        # Check if bws_test is available
        import scipy.stats

        if not hasattr(scipy.stats, "bws_test"):
            pytest.skip("bws_test not available in this scipy version (requires >=1.12.0)")

        # Reference data
        ref_emb = self.get_embeddings(self.n, value=0.0)

        # Instantiate detector
        cd = DriftUnivariate(data=ref_emb, method="bws", p_val=0.05, correction="fdr")

        # Test predict on reference data
        preds = cd.predict(ref_emb)
        assert not preds.drifted and (preds.p_vals >= cd.p_val).any()

        # Test predict on heavily drifted data
        x = self.get_embeddings(self.n_test, value=0.5)
        preds = cd.predict(x)
        assert preds.drifted
        assert preds.distances.min() >= 0.0

    def test_bws_import_error(self):
        """Test that using bws without scipy 1.12+ raises ImportError."""
        import scipy.stats

        if hasattr(scipy.stats, "bws_test"):
            pytest.skip("bws_test is available, cannot test ImportError")

        data = np.random.randn(100, 10).astype(np.float32)
        with pytest.raises(ImportError, match="scipy>=1.12.0"):
            DriftUnivariate(data, method="bws")


@pytest.mark.required
class TestDriftUnivariateValidation:
    """Tests for parameter validation in DriftUnivariate."""

    def test_invalid_method(self):
        """Test that invalid method raises ValueError."""
        data = np.random.randn(100, 10).astype(np.float32)
        with pytest.raises(ValueError, match="method"):
            DriftUnivariate(data, method="invalid")  # type: ignore

    def test_invalid_alternative(self):
        """Test that invalid alternative raises ValueError."""
        data = np.random.randn(100, 10).astype(np.float32)
        with pytest.raises(ValueError, match="alternative"):
            DriftUnivariate(data, method="ks", alternative="invalid")  # type: ignore

    def test_cvm_ignores_alternative(self):
        """Test that CVM method works with alternative parameter (ignores it)."""
        data = np.random.randn(100, 10).astype(np.float32)
        # Should not raise, even though CVM doesn't use alternative
        cd = DriftUnivariate(data, method="cvm", alternative="greater")
        assert cd.method == "cvm"
        assert cd.alternative == "greater"  # stored but not used by CVM
