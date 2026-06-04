"""Wasserstein distance drift detector."""

__all__ = []

from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np
import scipy.stats
from numpy.typing import NDArray
from typing_extensions import Self

from dataeval.exceptions import NotFittedError
from dataeval.protocols import Array, FeatureExtractor, Threshold, UpdateStrategy
from dataeval.shift._drift._base import BaseDrift, ChunkableMixin, DriftAdaptiveMixin, DriftOutput
from dataeval.types import set_metadata
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
    and validation data. Drift is flagged when the ratio of operational distance to
    baseline distance exceeds a threshold.

    Unlike hypothesis-test-based detectors, this detector requires two reference
    datasets to be provided at fit time: a training set and an in-distribution
    validation set. The train/validation Wasserstein distance serves as a calibrated
    baseline for what "no drift" looks like. At predict time, the train/operational
    Wasserstein distance is divided by this baseline; ratios substantially above 1.0
    indicate drift.

    For multivariate data (e.g. model embeddings), the test is applied independently
    to each feature. A feature is considered drifted when its individual distance
    ratio exceeds the threshold. Overall drift is declared when any feature drifts.

    Uses a fit/predict lifecycle: construct with hyperparameters, call :meth:`fit`
    with training and validation data, then call :meth:`predict` with test data.
    Use :meth:`chunked` to create a chunked wrapper for time-series monitoring.

    Parameters
    ----------
    ratio_threshold : float, default 1.4
        Distance ratio above which drift is declared, per feature.
        A value of 1.4 means operational distances more than 40% larger than
        the train/validation baseline are flagged as drift.
    n_features : int or None, default None
        Number of features to analyse. When None, automatically inferred from
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
    :class:`DriftWasserstein.Stats` : Per-prediction statistics returned in
        :attr:`DriftOutput.details`.

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
            Mean distance ratio across all features. Values substantially above
            1.0 indicate drift; values near 1.0 indicate no drift.
        feature_drift : NDArray[bool]
            Boolean array indicating which features show drift.
            Shape matches the number of features in the input data.
        feature_ratios : NDArray[np.float32]
            Distance ratio for each feature (operational / baseline).
            Shape matches the number of features in the input data.
        distances : NDArray[np.float32]
            Wasserstein distances between training and operational data,
            one per feature. Shape matches the number of features.
        baseline_distances : NDArray[np.float32]
            Wasserstein distances between training and validation data,
            one per feature. Computed once during :meth:`fit` and reused
            across all subsequent :meth:`predict` calls.
            Shape matches the number of features.
        """

    @dataclass
    class Config:
        """Configuration for DriftWasserstein detector.

        Attributes
        ----------
        ratio_threshold : float, default 1.4
            Distance ratio above which drift is declared, per feature.
        n_features : int or None, default None
            Number of features to analyse.
        update_strategy : UpdateStrategy or None, default None
            Strategy for updating reference data over time.
        extractor : FeatureExtractor or None, default None
            Feature extractor for transforming input data before drift detection.
        """

        ratio_threshold: float = 1.4
        n_features: int | None = None
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

        # Initialise base + mixins
        BaseDrift.__init__(self)
        self._init_adaptive(
            extractor=extractor,
            update_strategy=update_strategy,
        )

        if not isinstance(ratio_threshold, float | int) or isinstance(ratio_threshold, bool):
            raise ValueError("`ratio_threshold` must be a positive float.")
        if ratio_threshold <= 0:
            raise ValueError(f"`ratio_threshold` must be positive, got {ratio_threshold}.")

        self.ratio_threshold = ratio_threshold
        self._n_features = n_features

        # Populated during fit(); stored as encoded float32 arrays
        self._validation_data: NDArray[np.float32] | None = None
        self._baseline_distances: NDArray[np.float32] | None = None

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

    def fit(self, reference_data: Any, validation_data: Any = None) -> Self:
        """Fit detector with training and validation reference data.

        Encodes both datasets, stores the training set as the reference for
        subsequent drift comparisons, and computes the per-feature Wasserstein
        distance between training and validation data as the drift baseline.

        Parameters
        ----------
        reference_data : Any
            Training dataset used as the primary reference for drift detection.
            Can be Array-like or any type supported by the configured extractor.
        validation_data : Any, default None
            Validation dataset drawn from the same distribution as
            ``reference_data``. Used to calibrate the baseline distance. Must be
            compatible with ``reference_data`` (same feature dimensionality after
            encoding). Required despite the ``None`` default.

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
    ) -> NDArray[np.float32]:
        """Compute per-feature distance ratios, guarding against zero baselines.

        When a baseline distance is zero (train and val are identical for that
        feature), the ratio is set to 1.0 if the operational distance is also
        zero, and ``inf`` otherwise.

        Parameters
        ----------
        distances : NDArray[np.float32]
            Operational distances, shape ``(n_features,)``.

        Returns
        -------
        NDArray[np.float32]
            Distance ratios, shape ``(n_features,)``.
        """
        baseline = self.baseline_distances
        zero_mask = baseline == 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(zero_mask, np.where(distances == 0.0, 1.0, np.inf), distances / baseline)
        return ratios.astype(np.float32)

    def score(self, data: Array) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Compute per-feature distance ratios and raw Wasserstein distances.

        Encodes ``data``, computes the per-feature Wasserstein distance between
        the training reference and ``data``, then divides by the baseline
        train/validation distances.

        Parameters
        ----------
        data : Array
            Test dataset to compare against reference data.

        Returns
        -------
        tuple[NDArray[np.float32], NDArray[np.float32]]
            First array contains distance ratios for each feature.
            Second array contains raw Wasserstein distances for each feature.
            Both arrays have shape ``(n_features,)``.
        """
        x_np = self._encode(data)
        distances = self._wasserstein_distances(self.reference_data, x_np)
        ratios = self._compute_ratios(distances)
        return ratios, distances

    def _compute_chunk_metric(self, chunk_data: NDArray[np.float32]) -> float:
        """Compute mean distance ratio: chunk vs training reference."""
        distances = self._wasserstein_distances(self.reference_data, chunk_data)
        ratios = self._compute_ratios(distances)
        return float(np.mean(ratios))

    def _compute_chunk_baselines(
        self,
        chunks: list[NDArray[np.float32]],
    ) -> NDArray[np.float32]:
        """Compute baseline chunk metrics using the train/val ratio.

        Rather than comparing chunks against each other (as in the univariate
        case), each chunk is compared against the training reference, mirroring
        the predict-time behaviour. This means the chunked baseline reflects the
        expected ratio distribution for in-distribution data.

        Parameters
        ----------
        chunks : list[NDArray[np.float32]]
            Reference data split into chunks (derived from training data).

        Returns
        -------
        NDArray[np.float32]
            Mean distance ratio per chunk.
        """
        return np.array([self._compute_chunk_metric(c) for c in chunks], dtype=np.float32)

    def _default_chunk_threshold(self) -> Threshold:
        return ConstantThreshold(upper=self.ratio_threshold)

    @set_metadata
    def predict(self, data: Any) -> DriftOutput["DriftWasserstein.Stats"]:
        """Predict drift and optionally update reference data.

        Computes per-feature Wasserstein distances between the training
        reference and ``data``, divides by the train/validation baseline
        distances, and flags drift when any feature ratio exceeds
        :attr:`ratio_threshold`.

        Parameters
        ----------
        data : Any
            Test dataset to analyse for drift.

        Returns
        -------
        DriftOutput[DriftWasserstein.Stats]
            Drift prediction with per-feature statistics.
        """
        if not self._fitted:
            raise NotFittedError("Must call fit() before predict().")

        ratios, distances = self.score(data)

        feature_drift = (ratios > self.ratio_threshold).astype(np.bool_)
        drift_pred = bool(feature_drift.any())

        result = DriftOutput(
            drifted=drift_pred,
            threshold=self.ratio_threshold,
            distance=float(np.mean(ratios)),
            metric_name=self._metric_name,
            details=_DriftWassersteinStats(
                ratio=float(np.mean(ratios)),
                feature_drift=feature_drift,
                feature_ratios=ratios,
                distances=distances,
                baseline_distances=self.baseline_distances,
            ),
        )

        self._apply_update_strategy(data)

        return result
