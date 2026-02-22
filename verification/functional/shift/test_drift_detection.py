"""Verify that distribution shift detectors produce correct output types.

Maps to meta repo test cases:
  - TC-4.1: Drift detection (DriftUnivariate, DriftMMD)
"""

import numpy as np
import pytest

from dataeval.shift._drift._base import DriftOutput


@pytest.mark.test_case("4-1")
class TestDriftDetection:
    """Verify DriftUnivariate and DriftMMD detectors."""

    def test_drift_univariate_returns_output(self):
        from dataeval.shift import DriftUnivariate

        ref = np.ones((50, 8), dtype=np.float32)
        detector = DriftUnivariate(method="ks").fit(ref)
        result = detector.predict(np.zeros((20, 8), dtype=np.float32))
        assert hasattr(result, "drifted")
        assert hasattr(result, "p_val")
        assert hasattr(result, "distance")

    def test_drift_univariate_detects_clear_shift(self):
        from dataeval.shift import DriftUnivariate

        ref = np.zeros((100, 8), dtype=np.float32)
        test = np.ones((50, 8), dtype=np.float32)
        detector = DriftUnivariate(method="ks").fit(ref)
        result = detector.predict(test)
        assert result.drifted is True

    def test_drift_univariate_no_shift_on_same_distribution(self):
        from dataeval.shift import DriftUnivariate

        rng = np.random.default_rng(42)
        ref = rng.standard_normal((100, 8)).astype(np.float32)
        test = rng.standard_normal((50, 8)).astype(np.float32)
        detector = DriftUnivariate(method="ks").fit(ref)
        result = detector.predict(test)
        assert result.drifted is False

    def test_drift_univariate_has_per_feature_results(self):
        from dataeval.shift import DriftUnivariate

        ref = np.ones((50, 8), dtype=np.float32)
        detector = DriftUnivariate(method="ks").fit(ref)
        result = detector.predict(np.zeros((20, 8), dtype=np.float32))
        assert isinstance(result, DriftOutput)
        assert len(result.stats["feature_drift"]) == 8
        assert result.stats["p_vals"] is not None
        assert result.stats["distances"] is not None

    def test_drift_univariate_cvm_method(self):
        from dataeval.shift import DriftUnivariate

        ref = np.ones((50, 8), dtype=np.float32)
        detector = DriftUnivariate(method="cvm").fit(ref)
        result = detector.predict(np.zeros((20, 8), dtype=np.float32))
        assert hasattr(result, "drifted")

    def test_drift_mmd_returns_output(self):
        from dataeval.shift import DriftMMD

        rng = np.random.default_rng(42)
        ref = rng.standard_normal((50, 8)).astype(np.float32)
        detector = DriftMMD(n_permutations=10).fit(ref)
        result = detector.predict(rng.standard_normal((20, 8)).astype(np.float32))
        assert hasattr(result, "drifted")
        assert hasattr(result, "p_val")
        assert hasattr(result, "distance")
        assert hasattr(result, "stats")

    def test_drift_mmd_detects_clear_shift(self):
        from dataeval.shift import DriftMMD

        ref = np.zeros((100, 8), dtype=np.float32)
        test = np.full((50, 8), 10.0, dtype=np.float32)
        detector = DriftMMD(n_permutations=20).fit(ref)
        result = detector.predict(test)
        assert result.drifted is True
