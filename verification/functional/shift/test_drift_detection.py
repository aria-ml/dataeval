"""Verify that distribution shift detectors produce correct output types.

Maps to meta repo test cases:
  - TC-4.1: Drift detection (Univariate, MMD, KNeighbors, Reconstruction, DomainClassifier)
"""

import numpy as np
import pytest

import dataeval.config as config


@pytest.fixture(autouse=True)
def set_batch_size():
    config.set_batch_size(16)
    yield
    config.set_batch_size(None)


@pytest.mark.test_case("4-1")
class TestDriftDetection:
    """Verify Drift detectors."""

    def test_drift_univariate_detects_clear_shift(self):
        from dataeval.shift import DriftUnivariate

        ref = np.zeros((100, 8), dtype=np.float32)
        test = np.ones((50, 8), dtype=np.float32)
        detector = DriftUnivariate(method="ks").fit(ref)
        result = detector.predict(test)
        assert result.drifted is True

    def test_drift_mmd_detects_clear_shift(self):
        from dataeval.shift import DriftMMD

        rng = np.random.default_rng(42)
        ref = rng.standard_normal((50, 8)).astype(np.float32)
        detector = DriftMMD(n_permutations=20).fit(ref)
        result = detector.predict(rng.standard_normal((20, 8)).astype(np.float32) + 10.0)
        assert result.drifted is True

    def test_drift_kneighbors_detects_shift(self):
        from dataeval.shift import DriftKNeighbors

        rng = np.random.default_rng(42)
        ref = rng.standard_normal((50, 8)).astype(np.float32)
        detector = DriftKNeighbors().fit(ref)
        result = detector.predict(rng.standard_normal((20, 8)).astype(np.float32) + 10.0)
        assert result.drifted is True

    def test_drift_reconstruction_detects_shift(self):
        # Only test if torch is available as reconstruction usually needs a model
        pytest.importorskip("torch")
        from dataeval.shift import DriftReconstruction
        from dataeval.utils.models import AE

        rng = np.random.default_rng(42)
        # Use 28x28 to avoid kernel size issues in AE
        ref = rng.random((20, 1, 28, 28)).astype(np.float32)
        model = AE(input_shape=(1, 28, 28))
        detector = DriftReconstruction(model=model).fit(ref)
        test = rng.random((10, 1, 28, 28)).astype(np.float32) + 0.5
        np.clip(test, 0, 1, out=test)
        result = detector.predict(test)
        assert hasattr(result, "drifted")

    def test_drift_domain_classifier_detects_shift(self):
        pytest.importorskip("torch")
        from dataeval.shift import DriftDomainClassifier

        rng = np.random.default_rng(42)
        ref = rng.standard_normal((100, 8)).astype(np.float32)
        detector = DriftDomainClassifier().fit(ref)
        result = detector.predict(rng.standard_normal((50, 8)).astype(np.float32) + 5.0)
        assert hasattr(result, "drifted")

    def test_chunked_drift_wrapper(self):
        from dataeval.shift import ChunkedDrift, DriftUnivariate

        ref = np.zeros((100, 8), dtype=np.float32)
        detector = DriftUnivariate(method="ks")
        chunked = ChunkedDrift(detector, chunk_size=10)
        # Must fit the WRAPPER which fits the detector
        chunked.fit(ref)

        test = np.ones((20, 8), dtype=np.float32)
        result = chunked.predict(test)
        # ChunkedDrift result.details is a list of results per chunk
        assert len(result.details) == 2
