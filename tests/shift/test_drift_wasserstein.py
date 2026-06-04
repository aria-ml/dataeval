"""Tests for the Wasserstein-distance drift detector."""

import numpy as np
import pytest

from dataeval.exceptions import NotFittedError
from dataeval.shift._drift._base import DriftOutput
from dataeval.shift._drift._wasserstein import DriftWasserstein


def _ref(n: int = 200, n_features: int = 8, loc: float = 0.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, n_features)) + loc).astype(np.float32)


@pytest.mark.required
class TestDriftWassersteinFit:
    """Construction, fitting, and lazy state."""

    def test_fit_returns_self_and_sets_baseline(self):
        det = DriftWasserstein()
        out = det.fit(_ref(seed=1), _ref(n=100, seed=2))
        assert out is det
        assert det.baseline_distances.shape == (8,)
        assert det.n_features == 8

    def test_baseline_cached_across_predicts(self):
        """Baseline distances are computed once in fit and reused."""
        det = DriftWasserstein().fit(_ref(seed=1), _ref(n=100, seed=2))
        baseline = det.baseline_distances
        det.predict(_ref(n=50, seed=3))
        assert det.baseline_distances is baseline

    def test_n_features_explicit_override(self):
        det = DriftWasserstein(n_features=8).fit(_ref(seed=1), _ref(n=100, seed=2))
        assert det.n_features == 8


@pytest.mark.required
class TestDriftWassersteinNotFitted:
    """Accessing fitted state before fit raises."""

    def test_predict_before_fit(self):
        with pytest.raises(NotFittedError):
            DriftWasserstein().predict(_ref())

    def test_baseline_before_fit(self):
        with pytest.raises(NotFittedError):
            _ = DriftWasserstein().baseline_distances

    def test_n_features_before_fit(self):
        with pytest.raises(NotFittedError):
            _ = DriftWasserstein().n_features


@pytest.mark.required
class TestDriftWassersteinValidation:
    """ratio_threshold validation."""

    def test_rejects_non_positive(self):
        with pytest.raises(ValueError, match="must be positive"):
            DriftWasserstein(ratio_threshold=0.0)

    def test_rejects_bool(self):
        with pytest.raises(ValueError, match="must be a positive float"):
            DriftWasserstein(ratio_threshold=True)  # type: ignore[arg-type]

    def test_config_supplies_default(self):
        det = DriftWasserstein(config=DriftWasserstein.Config(ratio_threshold=1.2))
        assert det.ratio_threshold == 1.2

    def test_explicit_arg_overrides_config(self):
        det = DriftWasserstein(ratio_threshold=2.0, config=DriftWasserstein.Config(ratio_threshold=1.2))
        assert det.ratio_threshold == 2.0


@pytest.mark.required
class TestDriftWassersteinPredict:
    """Drift decisions."""

    def test_no_drift_on_in_distribution(self):
        # The any-feature rule at a tight threshold is sensitive to sampling noise
        # across many features, so use a generous threshold to test the decision
        # logic (in-distribution ratios cluster near 1; a 5x baseline is not drift).
        det = DriftWasserstein(ratio_threshold=5.0).fit(_ref(n=500, seed=1), _ref(n=500, seed=2))
        result = det.predict(_ref(n=500, seed=3))
        assert isinstance(result, DriftOutput)
        assert result.drifted is False

    def test_drift_on_shifted_data(self):
        det = DriftWasserstein().fit(_ref(seed=1), _ref(n=100, seed=2))
        result = det.predict(_ref(n=100, loc=5.0, seed=3))
        assert result.drifted is True
        assert result.details["feature_drift"].any()

    def test_score_shapes(self):
        det = DriftWasserstein().fit(_ref(seed=1), _ref(n=100, seed=2))
        ratios, distances = det.score(_ref(n=50, seed=3))
        assert ratios.shape == (8,)
        assert distances.shape == (8,)

    def test_zero_baseline_handling(self):
        """Identical train/val gives zero baseline; ratios stay finite when distances are zero."""
        data = _ref(seed=1)
        det = DriftWasserstein().fit(data, data.copy())
        np.testing.assert_allclose(det.baseline_distances, 0.0)
        # Same data again -> zero distance over zero baseline -> ratio defined as 1.0 -> no drift
        result = det.predict(data.copy())
        assert result.drifted is False


@pytest.mark.required
class TestDriftWassersteinChunked:
    """Two-reference chunked monitoring."""

    def test_chunked_fit_predict(self):
        chunked = DriftWasserstein().chunked(chunk_size=50)
        chunked.fit(_ref(seed=1), _ref(n=100, seed=2))
        result = chunked.predict(_ref(n=100, loc=5.0, seed=3))
        assert result.drifted is True
