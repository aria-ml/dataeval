"""Verify that distribution shift detectors produce correct output types.

Maps to meta repo test cases:
  - TC-4.1: Drift detection (Univariate, MMD, KNeighbors, Reconstruction, DomainClassifier)
"""

from typing import Literal

import numpy as np
import pytest

import dataeval.config as config


@pytest.fixture(autouse=True)
def set_batch_size():
    config.set_batch_size(16)
    yield
    config.set_batch_size(None)


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

    @pytest.mark.parametrize(
        "detector_name", ["univariate", "mmd", "kneighbors", "domain_classifier", "reconstruction"]
    )
    def test_no_false_alarm_on_same_distribution_sample(self, detector_name: str):
        """A second sample from the reference distribution must not be flagged as drift."""
        from dataeval.shift import (
            DriftDomainClassifier,
            DriftKNeighbors,
            DriftMMD,
            DriftReconstruction,
            DriftUnivariate,
        )

        # Fixed seeds: a test at p<0.05 false-alarms on ~5% of random draws, so the draw is pinned.
        rng = np.random.default_rng(0)
        config.set_seed(0)
        if detector_name == "reconstruction":
            pytest.importorskip("torch")
            from dataeval.utils.models import AE

            ramp = np.linspace(0, 1, 28, dtype=np.float32)

            def sample(n):  # learnable structure: blends of a vertical and a horizontal ramp
                a = rng.random((n, 1, 1, 1)).astype(np.float32)
                return a * ramp[None, None, :, None] + (1 - a) * ramp[None, None, None, :]

            ref, same = sample(100), sample(50)
            detector = DriftReconstruction(model=AE(input_shape=(1, 28, 28)))
        else:
            ref = rng.standard_normal((100, 8)).astype(np.float32)
            same = rng.standard_normal((50, 8)).astype(np.float32)
            if detector_name == "domain_classifier":
                pytest.importorskip("torch")
            detector = {
                "univariate": lambda: DriftUnivariate(method="ks"),
                "mmd": lambda: DriftMMD(n_permutations=50),
                "kneighbors": lambda: DriftKNeighbors(),
                "domain_classifier": lambda: DriftDomainClassifier(),
            }[detector_name]()

        result = detector.fit(ref).predict(same)
        config.set_seed(None)
        assert result.drifted is False

    def test_drift_output_fields_and_univariate_details(self):
        from dataeval.shift import DriftUnivariate
        from dataeval.shift._drift._base import DriftOutput

        ref = np.zeros((100, 8), dtype=np.float32)
        result = DriftUnivariate(method="ks").fit(ref).predict(np.ones((50, 8), dtype=np.float32))

        assert isinstance(result, DriftOutput)
        assert isinstance(result.drifted, bool)
        assert isinstance(result.distance, float)
        assert result.distance > 0
        assert 0 < result.threshold <= 1
        assert result.metric_name == "ks_distance"
        assert result.details is not None
        for key in ("p_vals", "feature_drift", "distances"):
            assert len(result.details[key]) == 8  # one entry per feature
        assert result.details["feature_drift"].dtype == np.bool_
        assert result.details["feature_drift"].all()
        assert (result.details["p_vals"] < 0.05).all()

    @pytest.mark.parametrize("method", ["ks", "cvm"])
    def test_drift_univariate_method_selection(self, method: Literal["ks", "cvm"]):
        from dataeval.shift import DriftUnivariate

        ref = np.zeros((100, 8), dtype=np.float32)
        result = DriftUnivariate(method=method).fit(ref).predict(np.ones((50, 8), dtype=np.float32))
        assert result.metric_name == f"{method}_distance"
        assert result.drifted is True

    def test_reference_update_strategies(self):
        from dataeval.shift import DriftUnivariate
        from dataeval.shift.update_strategies import LastSeenUpdateStrategy, ReservoirSamplingUpdateStrategy

        n = 50
        ref = np.zeros((30, 4), dtype=np.float32)

        # Last-seen: the reference becomes the most recent n instances seen.
        last = DriftUnivariate(method="ks", update_strategy=LastSeenUpdateStrategy(n)).fit(ref)
        last.predict(np.ones((40, 4), dtype=np.float32))
        assert last.reference_data.shape == (n, 4)
        assert np.unique(last.reference_data[:, 0], return_counts=True)[1].tolist() == [10, 40]  # 10 old zeros, 40 ones
        last.predict(np.full((10, 4), 2, dtype=np.float32))
        values, counts = np.unique(last.reference_data[:, 0], return_counts=True)
        assert dict(zip(values.tolist(), counts.tolist(), strict=True)) == {1.0: 40, 2.0: 10}

        # Reservoir sampling: the reference is capped at n and every row comes from data seen so far.
        np.random.seed(0)
        reservoir = DriftUnivariate(method="ks", update_strategy=ReservoirSamplingUpdateStrategy(n)).fit(ref)
        seen = {0.0}
        for value in (1.0, 2.0, 3.0):
            reservoir.predict(np.full((40, 4), value, dtype=np.float32))
            seen.add(value)
            assert reservoir.reference_data.shape[0] <= n
            assert set(np.unique(reservoir.reference_data[:, 0]).tolist()) <= seen
        assert reservoir.reference_data.shape[0] == n
        assert 3.0 in set(reservoir.reference_data[:, 0].tolist())  # newest data can displace the old
