"""Verify that distribution shift detectors produce correct output types.

Maps to meta repo test cases:
  - TC-4.1: Drift detection (DriftUnivariate, DriftMMD)
"""

import numpy as np
import pytest


@pytest.mark.test_case("4-1")
class TestDriftDetection:
    """Verify DriftUnivariate and DriftMMD detectors."""

    def test_drift_univariate_returns_output(self):
        from dataeval.shift import DriftUnivariate

        ref = np.ones((50, 8), dtype=np.float32)
        detector = DriftUnivariate(data=ref, method="ks")
        result = detector.predict(np.zeros((20, 8), dtype=np.float32))
        assert hasattr(result, "drifted")
        assert hasattr(result, "p_val")
        assert hasattr(result, "distance")

    def test_drift_univariate_detects_clear_shift(self):
        from dataeval.shift import DriftUnivariate

        ref = np.zeros((100, 8), dtype=np.float32)
        test = np.ones((50, 8), dtype=np.float32)
        detector = DriftUnivariate(data=ref, method="ks")
        result = detector.predict(test)
        assert result.drifted is True

    def test_drift_univariate_no_shift_on_same_distribution(self):
        from dataeval.shift import DriftUnivariate

        rng = np.random.default_rng(42)
        ref = rng.standard_normal((100, 8)).astype(np.float32)
        test = rng.standard_normal((50, 8)).astype(np.float32)
        detector = DriftUnivariate(data=ref, method="ks")
        result = detector.predict(test)
        assert result.drifted is False

    def test_drift_univariate_has_per_feature_results(self):
        from dataeval.shift import DriftUnivariate

        ref = np.ones((50, 8), dtype=np.float32)
        detector = DriftUnivariate(data=ref, method="ks")
        result = detector.predict(np.zeros((20, 8), dtype=np.float32))
        assert hasattr(result, "feature_drift")
        assert hasattr(result, "p_vals")
        assert hasattr(result, "distances")
        assert len(result.feature_drift) == 8

    def test_drift_univariate_cvm_method(self):
        from dataeval.shift import DriftUnivariate

        ref = np.ones((50, 8), dtype=np.float32)
        detector = DriftUnivariate(data=ref, method="cvm")
        result = detector.predict(np.zeros((20, 8), dtype=np.float32))
        assert hasattr(result, "drifted")

    def test_drift_mmd_returns_output(self):
        from dataeval.shift import DriftMMD

        rng = np.random.default_rng(42)
        ref = rng.standard_normal((50, 8)).astype(np.float32)
        detector = DriftMMD(data=ref, n_permutations=10)
        result = detector.predict(rng.standard_normal((20, 8)).astype(np.float32))
        assert hasattr(result, "drifted")
        assert hasattr(result, "p_val")
        assert hasattr(result, "distance")
        assert hasattr(result, "distance_threshold")

    def test_drift_mmd_detects_clear_shift(self):
        from dataeval.shift import DriftMMD

        ref = np.zeros((100, 8), dtype=np.float32)
        test = np.full((50, 8), 10.0, dtype=np.float32)
        detector = DriftMMD(data=ref, n_permutations=20)
        result = detector.predict(test)
        assert result.drifted is True
