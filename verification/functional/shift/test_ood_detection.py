"""Verify that out-of-distribution detectors produce correct output types.

Maps to meta repo test cases:
  - TC-5.1: OOD detection (OODKNeighbors)
"""

import numpy as np
import pytest


@pytest.mark.test_case("5-1")
class TestOODDetection:
    """Verify OODKNeighbors detector."""

    def test_ood_kneighbors_score_returns_output(self):
        from dataeval.shift import OODKNeighbors

        rng = np.random.default_rng(42)
        ref = rng.standard_normal((50, 16)).astype(np.float32)
        detector = OODKNeighbors(k=5)
        detector.fit(ref, threshold_perc=95.0)

        test = rng.standard_normal((10, 16)).astype(np.float32)
        scores = detector.score(test)
        assert hasattr(scores, "instance_score")
        assert len(scores.instance_score) == 10

    def test_ood_kneighbors_predict_returns_output(self):
        from dataeval.shift import OODKNeighbors

        rng = np.random.default_rng(42)
        ref = rng.standard_normal((50, 16)).astype(np.float32)
        detector = OODKNeighbors(k=5)
        detector.fit(ref, threshold_perc=95.0)

        test = rng.standard_normal((10, 16)).astype(np.float32)
        result = detector.predict(test)
        assert hasattr(result, "is_ood")
        assert hasattr(result, "instance_score")
        assert len(result.is_ood) == 10

    def test_ood_kneighbors_flags_outliers(self):
        from dataeval.shift import OODKNeighbors

        rng = np.random.default_rng(42)
        # Tight cluster for reference
        ref = rng.standard_normal((100, 16)).astype(np.float32)
        detector = OODKNeighbors(k=5, distance_metric="euclidean")
        detector.fit(ref, threshold_perc=95.0)

        # Far-away OOD samples (euclidean distance will be very large)
        ood = np.full((10, 16), 100.0, dtype=np.float32)
        result = detector.predict(ood)
        assert result.is_ood.sum() > 0

    def test_ood_kneighbors_supports_euclidean(self):
        from dataeval.shift import OODKNeighbors

        rng = np.random.default_rng(42)
        ref = rng.standard_normal((50, 16)).astype(np.float32)
        detector = OODKNeighbors(k=5, distance_metric="euclidean")
        detector.fit(ref, threshold_perc=95.0)

        test = rng.standard_normal((10, 16)).astype(np.float32)
        result = detector.predict(test)
        assert hasattr(result, "is_ood")
