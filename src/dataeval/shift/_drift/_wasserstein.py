"""Wasserstein distance drift detector."""

__all__ = []

from typing import Any, Self, TypedDict

import numpy as np
import scipy.stats
from numpy.typing import NDArray
from pydantic import PositiveFloat, PositiveInt

from dataeval.exceptions import NotFittedError
from dataeval.protocols import Array, FeatureExtractor, Threshold, UpdateStrategy
from dataeval.shift._drift._base import BaseDrift, ChunkableMixin, DriftAdaptiveMixin, DriftOutput
from dataeval.types import EvaluatorConfig, set_metadata
from dataeval.utils.thresholds import ConstantThreshold


class _DriftWassersteinStats(TypedDict):
    ratio: float
    feature_drift: NDArray[np.bool_]
    feature_ratios: NDArray[np.float32]
    distances: NDArray[np.float32]
    baseline_distances: NDArray[np.float32]


class DriftWasserstein(DriftAdaptiveMixin, ChunkableMixin, BaseDrift[_DriftWassersteinStats]):
    """:term:`Drift` detector using Wasserstein distance with a validation baseline.

    Detects distributional changes by comparing the Wasserstein distance between
    training and operational data against a baseline distance computed from training
    and validation data. Drift is flagged when the mean ratio of operational distance
    to baseline distance over the features, corrected for sample sizes, exceeds a
    threshold.

    Unlike hypothesis-test-based detectors, this detector requires two reference
    datasets to be provided at fit time: a training set and an in-distribution
    validation set. The per-feature train/validation Wasserstein distance is the
    baseline for what "no drift" looks like. It comes from one train/validation
    comparison, so it is itself a noisy estimate. At predict time, the per-feature
    train/operational distance is divided by this baseline and corrected for the
    sample sizes (see Notes), and the ratios are averaged over the features. A mean
    ratio near 1.0 means the operational data differs from training about as much as
    the validation data did; drift is declared when it exceeds ``ratio_threshold``.

    Each feature's ratio is also reported, with a per-feature flag in
    ``feature_drift`` of :class:`~dataeval.shift.DriftWasserstein.Stats`. A single
    feature's ratio is noisy, so the flags say where to look and not whether drift
    happened.

    Uses a fit/predict lifecycle: construct with hyperparameters, call :meth:`fit`
    with training and validation data, then call :meth:`predict` with test data.
    Use :meth:`chunked` to create a chunked wrapper for time-series monitoring. In
    chunked mode only the default :class:`~dataeval.utils.thresholds.ConstantThreshold`
    is calibrated. The baseline chunks are subsets of the training reference, so
    their distance to it, and their mean ratio, is smaller than that of operational
    chunks from the same distribution (in a simulation with chunks of a quarter of
    the reference, median 0.87 against 1.12). A threshold derived from the spread of
    the baseline chunks, such as ``"zscore"``, then alarms on clean chunks: on 6% to
    100% of them in that simulation, more as the chunks grow relative to the
    reference.

    .. versionchanged:: 1.2
        Drift is decided on the mean ratio over the features, corrected for the
        sizes of the training, validation and operational data. It was decided when
        any single feature's ratio exceeded the threshold, which raised a false
        alarm on nearly every clean batch with many features.

    Parameters
    ----------
    ratio_threshold : float, default 1.4
        Mean distance ratio over the features, corrected for sample sizes, above
        which drift is declared. A value of 1.4 means operational distances on
        average more than 40% larger than the train/validation baseline are flagged
        as drift. Lower it for more sensitivity, after you measure the false alarm
        rate on held-out batches from the reference distribution.
    n_features : int or None, default None
        Number of features to analyze. When None, automatically inferred from
        the flattened shape of the first data sample.
    extractor : FeatureExtractor or None, default None
        Optional feature extraction function to convert input data to arrays.
        When provided, enables drift detection on non-array inputs such as
        datasets, metadata, or raw model outputs. The extractor is applied to
        training, validation, and test data before drift detection.
        When None, data must already be Array-like.
    update_strategy : UpdateStrategy or None, default None
        Strategy for updating the training reference data when new data arrives.
        When None, reference data remains fixed throughout detection.
    config : DriftWasserstein.Config or None, default None
        Optional configuration object with default parameters. Parameters
        specified directly in __init__ will override config defaults.

    See Also
    --------
    :class:`~dataeval.shift.DriftWasserstein.Stats` : Per-prediction statistics returned in
        :attr:`~dataeval.shift.DriftOutput.details`.
    :class:`~dataeval.shift.DriftUnivariate` : The calibrated KS test, for 1 to 4 features.

    Notes
    -----
    The expected Wasserstein distance between two samples of one distribution
    scales with ``sqrt(1/n_a + 1/n_b)``. Each ratio with a nonzero baseline is
    therefore divided by ``sqrt((1/n_ref + 1/n_op) / (1/n_fit + 1/n_val))``, where
    ``n_ref`` is the size of the training reference at predict time (an
    ``update_strategy`` can change it), ``n_op`` the size of the operational data,
    and ``n_fit`` and ``n_val`` the sizes of the training and validation data at
    :meth:`fit`. Without the correction, a validation set larger than the
    operational batch raises the ratio of clean data above the threshold, and a
    smaller one hides drift.

    Measured false alarm rates at the default threshold, on data from the reference
    distribution:

    - about 0% to 3% from 16 independent continuous features of finite variance;
    - 5% to 19% when the features are correlated;
    - roughly 15% to 35% with 1 to 4 features, where the single baseline is noisy.
      Use :class:`~dataeval.shift.DriftUnivariate`, the calibrated KS test, there.

    The detector is unusable on constant or discrete features and on features with
    infinite-variance tails. Drop constant and discrete features before you use it.
    A feature that is constant in both reference samples has a baseline of 0. Its
    ratio is 1.0 when the operational distance is also 0 and infinite otherwise, so
    any change in it declares drift.

    The mean ratio does not see shifts confined to a few of many features. A shift
    in 1 of 512 features, even of 5 standard deviations, was not detected, and a 2
    standard deviation shift in 2 of 64 features was detected in 66% to 92% of
    trials, depending on the sample sizes. Lower ``ratio_threshold`` for
    more sensitivity, and measure the false alarm rate on held-out batches first.

    Example
    -------
    Basic drift detection with Wasserstein distance

    >>> rng = np.random.default_rng(42)
    >>> train_emb = rng.standard_normal((200, 64)).astype(np.float32)
    >>> val_emb = rng.standard_normal((100, 64)).astype(np.float32)
    >>> drift_detector = DriftWasserstein().fit(train_emb, val_emb)
    >>> test_emb = np.zeros((50, 64), dtype=np.float32)
    >>> result = drift_detector.predict(test_emb)
    >>> print(f"Drift detected: {result.drifted}")
    Drift detected: True

    Chunked drift detection

    >>> chunked = DriftWasserstein().chunked(chunk_size=50)
    >>> chunked.fit(train_emb, val_emb)  # doctest: +ELLIPSIS
    ChunkedDrift(DriftWasserstein(...), chunker=SizeChunker(...), fitted=True)
    >>> result = chunked.predict(test_emb)
    >>> print(f"Drift detected: {result.drifted}, chunks: {len(result.details)}")
    Drift detected: True, chunks: 1

    Using configuration:

    >>> config = DriftWasserstein.Config(ratio_threshold=1.2)
    >>> drift = DriftWasserstein(config=config).fit(train_emb, val_emb)
    """

    _metric_name = "wasserstein_ratio"

    class Stats(_DriftWassersteinStats):
        """Per-feature statistics from Wasserstein drift detection.

        Attributes
        ----------
        ratio : float
            Mean of ``feature_ratios`` over the features, the statistic drift is
            decided on: drift is declared when it exceeds ``ratio_threshold``. The
            same value as :attr:`~dataeval.shift.DriftOutput.distance`. Values near
            1.0 mean the operational data differs from training about as much as
            the validation data did.
        feature_drift : NDArray[bool]
            Whether each feature's ratio exceeds ``ratio_threshold``, a diagnostic
            of where to look. A single feature's ratio is noisy: on data from the
            reference distribution it exceeds the default threshold in roughly 15%
            to 35% of trials, so with many features some flags are expected by
            chance. Drift is decided on ``ratio``, so
            :attr:`~dataeval.shift.DriftOutput.drifted` does not equal
            ``feature_drift.any()``.
            Shape matches the number of features in the input data.

            .. versionchanged:: 1.2
                No longer decides drift.
        feature_ratios : NDArray[np.float32]
            Distance ratio for each feature: the operational distance divided by
            the baseline distance and by the sample-size correction described in
            the class Notes. It equals ``distances / baseline_distances`` only when
            the sample sizes match. A feature with a baseline of 0 has a ratio of
            1.0 when its operational distance is also 0, and ``inf`` otherwise.
            Shape matches the number of features in the input data.

            .. versionchanged:: 1.2
                Corrected for sample sizes.
        distances : NDArray[np.float32]
            Wasserstein distances between training and operational data,
            one per feature. Shape matches the number of features.
        baseline_distances : NDArray[np.float32]
            Wasserstein distances between training and validation data,
            one per feature. Computed once during :meth:`fit` and reused
            across all subsequent :meth:`predict` calls.
            Shape matches the number of features.
        """

    class Config(EvaluatorConfig):
        """Configuration for DriftWasserstein detector.

        Attributes
        ----------
        ratio_threshold : float, default 1.4
            Mean distance ratio over the features, corrected for sample sizes,
            above which drift is declared. The ratio's baseline comes from one
            train/validation comparison. At the default, measured false alarm
            rates are about 0% to 3% from 16 independent continuous features,
            5% to 19% with correlated features and roughly 15% to 35% with 1 to 4
            features. Shifts confined to a few of many features are not detected
            (see the Notes of :class:`~dataeval.shift.DriftWasserstein`).
        n_features : int or None, default None
            Number of features to analyze.
        update_strategy : UpdateStrategy or None, default None
            Strategy for updating reference data over time.
        extractor : FeatureExtractor or None, default None
            Feature extractor for transforming input data before drift detection.
        """

        ratio_threshold: PositiveFloat = 1.4
        n_features: PositiveInt | None = None
        update_strategy: UpdateStrategy | None = None
        extractor: FeatureExtractor | None = None

    def __init__(
        self,
        ratio_threshold: float | None = None,
        n_features: int | None = None,
        extractor: FeatureExtractor | None = None,
        update_strategy: UpdateStrategy | None = None,
        config: Config | None = None,
    ) -> None:
        base_config = config or DriftWasserstein.Config()

        ratio_threshold = ratio_threshold if ratio_threshold is not None else base_config.ratio_threshold
        n_features = n_features if n_features is not None else base_config.n_features

        self.config: DriftWasserstein.Config = DriftWasserstein.Config(
            ratio_threshold=ratio_threshold,
            n_features=n_features,
            update_strategy=update_strategy,
            extractor=extractor,
        )

        # Initialize base + mixins
        BaseDrift.__init__(self)
        self._init_adaptive(
            extractor=extractor,
            update_strategy=update_strategy,
        )

        if not isinstance(ratio_threshold, float | int) or isinstance(ratio_threshold, bool):
            raise ValueError("`ratio_threshold` must be a positive float.")

        self.ratio_threshold = ratio_threshold
        self._n_features = n_features

        # Populated during fit(); stored as encoded float32 arrays
        self._validation_data: NDArray[np.float32] | None = None
        self._baseline_distances: NDArray[np.float32] | None = None
        # 1/n_fit + 1/n_val, the sample-size term of the baseline distance; set in fit()
        self._baseline_size_term = 0.0

    @property
    def n_features(self) -> int:
        """Number of features in the reference data.

        Lazily computes the number of features from the encoded reference array
        if not provided during initialization.

        Returns
        -------
        int
            Number of features (flattened dimensions) in the reference data.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        """
        if self._n_features is None:
            if self._data is None:
                raise NotFittedError("Must call fit() before accessing n_features.")
            self._n_features = self.reference_data.shape[1]
        return self._n_features

    @property
    def baseline_distances(self) -> NDArray[np.float32]:
        """Per-feature Wasserstein distances between training and validation data.

        Computed once during :meth:`fit` and reused across all subsequent
        :meth:`predict` calls.

        Returns
        -------
        NDArray[np.float32]
            Baseline distances, shape ``(n_features,)``.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        """
        if self._baseline_distances is None:
            raise NotFittedError("Must call fit() before accessing baseline_distances.")
        return self._baseline_distances

    @set_metadata
    def from_embeddings(
        self, reference: Array, validation: Array, data: Array
    ) -> DriftOutput["DriftWasserstein.Stats"]:
        """
        Fit on reference embeddings and test data embeddings in one call.

        The same as ``fit(reference, validation).predict(data)``. Call :meth:`fit` and :meth:`predict`
        separately to fit once and test several datasets against the same reference.

        .. versionadded:: 1.2

        Parameters
        ----------
        reference : Array
            Embeddings of the reference data, shape ``(n_reference, n_features)``.
        validation : Array
            Embeddings of in-distribution data held out from the reference, which set the
            baseline distance, as :meth:`fit` takes them.
        data : Array
            Embeddings of the data to test for drift, shape ``(n_data, n_features)``.

        Returns
        -------
        DriftOutput[DriftWasserstein.Stats]
            The output :meth:`predict` returns after :meth:`fit`.

        See Also
        --------
        fit : Fit on the reference alone.
        predict : Test data against the fitted reference.

        Examples
        --------
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> result = DriftWasserstein().from_embeddings(rng.random((100, 8)), rng.random((100, 8)), rng.random((50, 8)))
        """
        return self.fit(reference, validation).predict(data)

    def fit(self, reference_data: Any, validation_data: Any = None) -> Self:
        """Fit detector with training and validation reference data.

        Encodes both datasets, stores the training set as the reference for
        subsequent drift comparisons, and computes the per-feature Wasserstein
        distance between training and validation data as the drift baseline. The
        baseline is one train/validation comparison, so it is itself noisy. The
        sizes of both datasets are recorded for the sample-size correction of the
        ratios (see the Notes of :class:`~dataeval.shift.DriftWasserstein`).

        Drop constant and discrete features first. A feature that is constant in
        both datasets has a baseline of 0 and gives an infinite ratio, and so
        declares drift, whenever the operational data differs in it.

        Parameters
        ----------
        reference_data : Any
            Training dataset used as the primary reference for drift detection.
            When an ``extractor`` is configured, this may be any input the extractor
            accepts (e.g. a full MAITE dataset or raw images); otherwise it must be
            array-like or an :class:`~dataeval.Embeddings`.
        validation_data : Any, default None
            Validation dataset drawn from the same distribution as
            ``reference_data``. Sets the baseline distance. Must be
            compatible with ``reference_data`` (same feature dimensionality after
            encoding) and accepts the same range of inputs. Required despite the
            ``None`` default.

        Returns
        -------
        Self

        Raises
        ------
        ValueError
            If ``validation_data`` is not provided.
        """
        if validation_data is None:
            raise ValueError("`DriftWasserstein.fit` requires both `reference_data` and `validation_data`.")
        self._set_adaptive_data(reference_data)
        self._validation_data = self._encode(validation_data)

        # Compute and cache per-feature baseline distances (train vs val)
        self._baseline_distances = self._wasserstein_distances(self.reference_data, self._validation_data)
        self._baseline_size_term = 1 / len(self.reference_data) + 1 / len(self._validation_data)
        self._fitted = True
        return self

    def _wasserstein_distances(
        self,
        x: NDArray[np.float32],
        y: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Compute per-feature Wasserstein distances between two datasets.

        Parameters
        ----------
        x : NDArray[np.float32]
            First dataset, shape ``(n_samples_x, n_features)``.
        y : NDArray[np.float32]
            Second dataset, shape ``(n_samples_y, n_features)``.

        Returns
        -------
        NDArray[np.float32]
            Wasserstein distance for each feature, shape ``(n_features,)``.
        """
        n_features = x.shape[1]
        distances = np.zeros(n_features, dtype=np.float32)
        for f in range(n_features):
            distances[f] = scipy.stats.wasserstein_distance(x[:, f], y[:, f])
        return distances

    def _compute_ratios(
        self,
        distances: NDArray[np.float32],
        n_operational: int,
    ) -> NDArray[np.float32]:
        """Compute per-feature distance ratios, corrected for sample sizes.

        A ratio with a nonzero baseline is divided by
        ``sqrt((1/n_ref + 1/n_operational) / (1/n_fit + 1/n_val))``, with ``n_ref`` the
        current reference size (the update strategy can cap it; the mixin's ``n`` counts
        every batch seen). A zero baseline (train and val identical for that feature) is
        left uncorrected: the ratio is 1.0 if the operational distance is also zero, and
        ``inf`` otherwise. Correcting it too would turn the 1.0 of a constant feature into
        ``1/c`` and shift the mean ratio.
        """
        baseline = self.baseline_distances
        c = np.sqrt((1 / len(self.reference_data) + 1 / n_operational) / self._baseline_size_term)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(baseline == 0.0, np.where(distances == 0.0, 1.0, np.inf), distances / (baseline * c))
        return ratios.astype(np.float32)

    def score(self, data: Any) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Compute per-feature distance ratios and raw Wasserstein distances.

        Encodes ``data``, computes the per-feature Wasserstein distance between
        the training reference and ``data``, then divides by the baseline
        train/validation distances and by the sample-size correction described in
        the Notes of :class:`~dataeval.shift.DriftWasserstein`. A feature with a
        baseline of 0 gets a ratio of 1.0 when its distance is also 0, and ``inf``
        otherwise. The ratios therefore equal ``distances / baseline_distances``
        only when the sample sizes match.

        .. versionchanged:: 1.2
            The ratios are corrected for sample sizes.

        Parameters
        ----------
        data : Any
            Test dataset to compare against reference data. When an ``extractor`` is
            configured, this may be any input the extractor accepts (e.g. a full
            MAITE dataset or raw images); otherwise it must be array-like or an
            :class:`~dataeval.Embeddings`.

        Returns
        -------
        tuple[NDArray[np.float32], NDArray[np.float32]]
            First array contains distance ratios for each feature.
            Second array contains raw Wasserstein distances for each feature.
            Both arrays have shape ``(n_features,)``.
        """
        x_np = self._encode(data)
        distances = self._wasserstein_distances(self.reference_data, x_np)
        ratios = self._compute_ratios(distances, len(x_np))
        return ratios, distances

    def _compute_chunk_metric(self, chunk_data: NDArray[np.float32]) -> float:
        """Compute mean distance ratio, corrected for the chunk size: chunk vs training reference."""
        distances = self._wasserstein_distances(self.reference_data, chunk_data)
        ratios = self._compute_ratios(distances, len(chunk_data))
        return float(np.mean(ratios))

    def _compute_chunk_baselines(
        self,
        chunks: list[NDArray[np.float32]],
    ) -> NDArray[np.float32]:
        """Compute baseline chunk metrics using the train/val ratio.

        Rather than comparing chunks against each other (as in the univariate
        case), each chunk is compared against the training reference, mirroring
        the predict-time behavior. The chunks are subsets of that reference, so
        their ratios run below those of clean operational chunks, and only the
        default constant threshold is calibrated (see the class docstring).
        """
        return np.array([self._compute_chunk_metric(c) for c in chunks], dtype=np.float32)

    def _default_chunk_threshold(self) -> Threshold:
        return ConstantThreshold(upper=self.ratio_threshold)

    @set_metadata
    def predict(self, data: Any) -> DriftOutput["DriftWasserstein.Stats"]:
        """Predict drift and optionally update reference data.

        Computes per-feature Wasserstein distances between the training
        reference and ``data``, divides by the train/validation baseline
        distances and corrects for the sample sizes, then flags drift when the
        mean ratio over the features exceeds :attr:`ratio_threshold`. The
        baseline is one train/validation comparison.

        At the default threshold, measured false alarm rates on data from the
        reference distribution are about 0% to 3% from 16 independent continuous
        features of finite variance, 5% to 19% with correlated features and
        roughly 15% to 35% with 1 to 4 features (use
        :class:`~dataeval.shift.DriftUnivariate`, the calibrated KS test, there).
        The detector is unusable on constant or discrete features and on
        infinite-variance tails. Shifts confined to a few of many features are
        not detected: a shift in 1 of 512 features was missed, and a 2 standard
        deviation shift in 2 of 64 features was detected in 66% to 92% of trials.
        Lower :attr:`ratio_threshold` for more sensitivity, and measure the false
        alarm rate on held-out batches first.

        The per-feature flags in the returned ``feature_drift`` say where to
        look. They do not decide drift, so ``drifted`` can be False while some
        flags are True.

        .. versionchanged:: 1.2
            Drift is decided on the mean ratio, corrected for sample sizes, and
            not when any single feature's ratio exceeds the threshold.

        Parameters
        ----------
        data : Any
            Test dataset to analyze for drift. When an ``extractor`` is configured,
            this may be any input the extractor accepts (e.g. a full MAITE dataset or
            raw images); otherwise it must be array-like or an :class:`~dataeval.Embeddings`.

        Returns
        -------
        DriftOutput[DriftWasserstein.Stats]
            Drift prediction with per-feature statistics.
        """
        if not self._fitted:
            raise NotFittedError("Must call fit() before predict().")

        ratios, distances = self.score(data)

        # The flags only show where to look: one feature's ratio is too noisy to decide drift on
        feature_drift = (ratios > self.ratio_threshold).astype(np.bool_)
        ratio = float(np.mean(ratios))

        result = DriftOutput(
            drifted=ratio > self.ratio_threshold,
            threshold=self.ratio_threshold,
            distance=ratio,
            metric_name=self._metric_name,
            feature_names=self._feature_names,
            details=_DriftWassersteinStats(
                ratio=ratio,
                feature_drift=feature_drift,
                feature_ratios=ratios,
                distances=distances,
                baseline_distances=self.baseline_distances,
            ),
        )

        self._apply_update_strategy(data)

        return result
