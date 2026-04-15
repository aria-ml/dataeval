"""Verify that out-of-distribution detectors produce correct output types.

Maps to meta repo test cases:
  - TC-5.1: OOD detection (KNeighbors, Reconstruction, DomainClassifier)
"""

import numpy as np
import pytest

import dataeval.config as config


@pytest.fixture(autouse=True)
def set_batch_size():
    config.set_batch_size(16)
    yield
    config.set_batch_size(None)


@pytest.mark.test_case("5-1")
class TestOODDetection:
    """Verify OOD detectors."""

    def test_ood_kneighbors_predict_returns_output(self):
        from dataeval.shift import OODKNeighbors

        rng = np.random.default_rng(42)
        ref = rng.standard_normal((50, 16)).astype(np.float32)
        detector = OODKNeighbors(k=5, threshold_perc=95.0)
        detector.fit(ref)

        test = rng.standard_normal((10, 16)).astype(np.float32)
        result = detector.predict(test)
        assert hasattr(result, "is_ood")
        assert len(result.is_ood) == 10

    def test_ood_reconstruction_detects_ood(self):
        pytest.importorskip("torch")
        from dataeval.shift import OODReconstruction
        from dataeval.utils.models import AE

        rng = np.random.default_rng(42)
        # Data must be on unit interval [0-1]
        ref = rng.random((20, 1, 28, 28)).astype(np.float32)
        model = AE(input_shape=(1, 28, 28))
        detector = OODReconstruction(model=model).fit(ref)

        ood = rng.random((10, 1, 28, 28)).astype(np.float32) + 0.5
        np.clip(ood, 0, 1, out=ood)
        result = detector.predict(ood)
        assert hasattr(result, "is_ood")

    def test_ood_domain_classifier_detects_ood(self):
        pytest.importorskip("torch")
        from dataeval.shift import OODDomainClassifier

        rng = np.random.default_rng(42)
        ref = rng.standard_normal((100, 8)).astype(np.float32)
        detector = OODDomainClassifier().fit(ref)

        ood = np.full((10, 8), 10.0, dtype=np.float32)
        result = detector.predict(ood)
        assert hasattr(result, "is_ood")

    def test_ood_detectors_support_threshold_perc(self):
        from dataeval.shift import OODKNeighbors

        rng = np.random.default_rng(42)
        ref = rng.standard_normal((50, 16)).astype(np.float32)
        # OODKNeighbors stores threshold_perc in its config
        detector = OODKNeighbors(threshold_perc=99.0)
        detector.fit(ref)
        assert detector.config.threshold_perc == 99.0
