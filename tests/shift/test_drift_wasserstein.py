"""Tests for the Wasserstein-distance drift detector."""

import numpy as np
import pytest

from dataeval.exceptions import NotFittedError
from dataeval.shift._drift._base import DriftOutput
from dataeval.shift._drift._wasserstein import DriftWasserstein
from dataeval.shift.update_strategies import LastSeenUpdateStrategy


def _ref(n: int = 200, n_features: int = 8, loc: float = 0.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, n_features)) + loc).astype(np.float32)


def _correction(n_ref: int, n_op: int, n_fit: int, n_val: int) -> float:
    """The sample-size correction the detector divides each ratio by, written out independently."""
    return float(np.sqrt((1 / n_ref + 1 / n_op) / (1 / n_fit + 1 / n_val)))


def _trials(
    sizes: tuple[int, int, int], n_features: int, n_trials: int, shift: float | np.ndarray = 0.0, seed: int = 0
) -> list[DriftOutput]:
    """Fit and predict on standard normal train/validation/test sets of the given sizes, test shifted by `shift`."""
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n_trials):
        train, val, test = (rng.standard_normal((n, n_features)) for n in sizes)
        det = DriftWasserstein().fit(train.astype(np.float32), val.astype(np.float32))
        results.append(det.predict((test + shift).astype(np.float32)))
    return results


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

    def test_fit_without_validation_data_raises(self):
        """fit requires validation_data; omitting it raises ValueError."""
        with pytest.raises(ValueError, match="requires both `reference_data` and `validation_data`"):
            DriftWasserstein().fit(_ref(seed=1))

    def test_fit_with_none_validation_data_raises(self):
        with pytest.raises(ValueError, match="requires both `reference_data` and `validation_data`"):
            DriftWasserstein().fit(_ref(seed=1), validation_data=None)


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
        with pytest.raises(ValueError, match="ratio_threshold"):
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
        # The default threshold, now that drift is decided on the mean ratio (1.12 here). The old any-feature rule
        # flagged this clean batch at 1.4, because one feature's ratio is 1.94, so the test used a threshold of 5.0.
        det = DriftWasserstein().fit(_ref(n=500, seed=1), _ref(n=500, seed=2))
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
class TestDriftWassersteinMeanRatio:
    """Drift is decided on the mean ratio over features, corrected for sample sizes."""

    @pytest.mark.parametrize("sizes", [(500, 200, 100), (2000, 500, 50), (500, 100, 400)])
    def test_few_false_alarms_on_clean_data(self, sizes):
        # Measured with these seeds: 0 false alarms in 100 trials and a median mean ratio of 1.10 to 1.12 in all
        # three size mixes. Uncorrected, the median is 1.44, 3.16 and 0.69, so the first two would alarm on most
        # clean batches and the last would hide drift.
        results = _trials(sizes, n_features=64, n_trials=100)
        assert np.mean([r.drifted for r in results]) <= 0.05
        assert 1.0 <= np.median([r.distance for r in results]) <= 1.25

    def test_detects_a_shift_of_every_feature(self):
        # A 0.5 sd shift of all 64 features gives a mean ratio of 3.7 to 4.3 in these seeded trials
        results = _trials((500, 100, 100), n_features=64, n_trials=10, shift=0.5)
        assert all(r.drifted for r in results)

    def test_detects_a_large_shift_in_two_of_64_features(self):
        # Measured with these seeds: detected in 91 of 100 trials (92% in an independent 200-trial run)
        shift = np.r_[np.full(2, 2.0), np.zeros(62)]
        results = _trials((500, 100, 100), n_features=64, n_trials=100, shift=shift)
        assert np.mean([r.drifted for r in results]) >= 0.8

    def test_misses_a_shift_in_one_of_512_features(self):
        # The documented blind spot: the mean over 512 features dilutes a shift in one of them, even of 5 sd, so it
        # is not detected (mean ratio 1.11 to 1.16 here, 0 of 100 trials in the evidence run). The feature's own flag
        # still points at it.
        shift = np.r_[5.0, np.zeros(511)]
        results = _trials((500, 100, 100), n_features=512, n_trials=5, shift=shift)
        assert not any(r.drifted for r in results)
        assert all(r.details["feature_drift"][0] for r in results)

    def test_drifted_is_the_mean_ratio_against_the_threshold(self):
        # With 64 clean features 13 feature ratios pass the threshold by chance here, but the mean (1.06) does not
        result = _trials((500, 100, 100), n_features=64, n_trials=1)[0]
        ratios = result.details["feature_ratios"]
        assert result.distance == result.details["ratio"] == pytest.approx(float(np.mean(ratios)))
        assert result.drifted == (result.distance > result.threshold)
        np.testing.assert_array_equal(result.details["feature_drift"], ratios > result.threshold)
        assert result.details["feature_drift"].shape == (64,)
        assert result.details["feature_drift"].any()
        assert result.drifted is False

    def test_ratios_are_corrected_for_sample_sizes(self):
        det = DriftWasserstein().fit(_ref(n=300, seed=1), _ref(n=150, seed=2))
        ratios, distances = det.score(_ref(n=60, seed=3))
        c = _correction(n_ref=300, n_op=60, n_fit=300, n_val=150)
        np.testing.assert_allclose(ratios, distances / det.baseline_distances / c, rtol=1e-5)

    @pytest.mark.parametrize("sizes", [(2000, 500, 50), (2000, 100, 1000)])
    def test_constant_features_keep_a_ratio_of_one(self, sizes):
        # 19 of 64 features constant 0 everywhere. Their zero baseline must not be corrected: in these mixes the
        # correction is 2.9 and 0.38, which would turn each 1.0 into 0.35 (a blind detector) or 2.6 (alarms on
        # every batch). Measured with these seeds: median mean ratio 1.07 in both mixes and no false alarms, where
        # correcting the zero baselines gives a median of 0.88, and of 1.56 with an alarm on all 50 batches.
        rng = np.random.default_rng(0)
        results = []
        for _ in range(50):
            train, val, test = (rng.standard_normal((n, 64)).astype(np.float32) for n in sizes)
            for x in (train, val, test):
                x[:, :19] = 0.0
            results.append(DriftWasserstein().fit(train, val).predict(test))
        assert all((r.details["feature_ratios"][:19] == 1.0).all() for r in results)
        assert 1.0 <= np.median([r.distance for r in results]) <= 1.25
        assert not any(r.drifted for r in results)

    def test_constant_feature_that_changes_gives_inf(self):
        train, val, test = _ref(n=300, seed=1), _ref(n=100, seed=2), _ref(n=50, seed=3)
        train[:, 0] = val[:, 0] = 0.0
        test[:, 0] = 1.0
        result = DriftWasserstein().fit(train, val).predict(test)
        assert np.isinf(result.details["feature_ratios"][0])
        assert np.isfinite(result.details["feature_ratios"][1:]).all()
        assert result.drifted is True

    def test_correction_uses_the_reference_size_at_predict_time(self):
        # LastSeenUpdateStrategy caps the reference at 200 rows after the first predict, while the mixin's
        # `n` keeps counting every row seen. The update runs after predict, so record the size before it.
        det = DriftWasserstein(update_strategy=LastSeenUpdateStrategy(n=200))
        det.fit(_ref(n=300, seed=1), _ref(n=100, seed=2))
        det.predict(_ref(n=50, seed=3))
        n_ref = len(det.reference_data)
        assert (n_ref, det.n) == (200, 350)
        result = det.predict(_ref(n=50, seed=4))
        c = _correction(n_ref=n_ref, n_op=50, n_fit=300, n_val=100)
        expected = result.details["distances"] / det.baseline_distances / c
        np.testing.assert_allclose(result.details["feature_ratios"], expected, rtol=1e-5)


@pytest.mark.required
class TestDriftWassersteinChunked:
    """Two-reference chunked monitoring."""

    def test_chunked_fit_predict(self):
        chunked = DriftWasserstein().chunked(chunk_size=50)
        chunked.fit(_ref(seed=1), _ref(n=100, seed=2))
        result = chunked.predict(_ref(n=100, loc=5.0, seed=3))
        assert result.drifted is True

    def test_operational_chunks_use_the_corrected_ratio(self):
        det = DriftWasserstein()
        chunked = det.chunked(chunk_size=50)
        chunked.fit(_ref(n=200, seed=1), _ref(n=100, seed=2))
        data = _ref(n=100, seed=3)
        values = chunked.predict(data).details["value"].to_list()
        c = _correction(n_ref=200, n_op=50, n_fit=200, n_val=100)
        expected = [np.mean(det.score(chunk)[1] / det.baseline_distances / c) for chunk in (data[:50], data[50:])]
        np.testing.assert_allclose(values, expected, rtol=1e-5)
