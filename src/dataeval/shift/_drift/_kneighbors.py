"""K-Nearest Neighbors based drift detector."""

__all__ = []

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.stats import mannwhitneyu
from typing_extensions import Self

from dataeval.exceptions import NotFittedError, ShapeMismatchError
from dataeval.protocols import FeatureExtractor, Threshold, UpdateStrategy
from dataeval.shift._drift._base import BaseDrift, ChunkableMixin, DriftAdaptiveMixin, DriftOutput
from dataeval.shift._shared._kneighbors import KNeighborsScorer
from dataeval.types import set_metadata
from dataeval.utils.thresholds import ZScoreThreshold


class _DriftKNeighborsStats(TypedDict):
    p_val: float
    mean_ref_distance: float
    mean_test_distance: float


class DriftKNeighbors(DriftAdaptiveMixin, ChunkableMixin, BaseDrift[_DriftKNeighborsStats]):
    """K-Nearest Neighbors based drift detector.

    Detects drift by comparing k-NN distances of test samples against the
    reference set. If test samples are farther from their k nearest neighbors
    in the reference set than expected, drift is detected.

    Uses a fit/predict lifecycle: construct with hyperparameters, call
    :meth:`fit` with reference data, then call :meth:`predict` with test data.
    Use :meth:`chunked` to create a chunked wrapper for time-series monitoring.

    Supports two modes:

    - **Non-chunked** (default): Computes per-sample k-NN distances for the
      test set and uses a Mann-Whitney U test against the reference baseline
      to produce a p-value. Drift is flagged when ``p_val < p_val_threshold``.
    - **Chunked** (via :meth:`chunked`): Splits data into chunks, computes
      mean k-NN distance per chunk, and uses threshold bounds to flag drift
      per chunk.

    Parameters
    ----------
    k : int, default 10
        Number of nearest neighbors.
    distance_metric : {"cosine", "euclidean"}, default "euclidean"
        Distance metric for neighbor search.
    p_val : float, default 0.05
        Significance threshold for non-chunked mode.
    extractor : FeatureExtractor or None, default None
        Feature extractor for transforming input data before drift detection.
        When provided, raw data is passed through the extractor before
        flattening and comparison. When None, data is used as-is.
    update_strategy : UpdateStrategy or None, default None
        Strategy for updating reference data when new data arrives.
        When None, reference data remains fixed throughout detection.
    config : DriftKNeighbors.Config or None, default None
        Optional configuration object.

    See Also
    --------
    :class:`DriftKNeighbors.Stats` : Per-prediction statistics returned in :attr:`DriftOutput.details`.

    Examples
    --------
    Non-chunked mode:

    >>> ref = np.random.randn(200, 32).astype(np.float32)
    >>> test = np.random.randn(100, 32).astype(np.float32) + 5  # shifted
    >>> detector = DriftKNeighbors(k=5).fit(ref)
    >>> result = detector.predict(test)
    >>> print(f"Drift: {result.drifted}")
    Drift: ...

    Chunked mode:

    >>> chunked = DriftKNeighbors(k=5).chunked(chunk_size=50)
    >>> chunked.fit(ref)
    ChunkedDrift(DriftKNeighbors(k=5, distance_metric='euclidean', p_val=0.05, extractor=None, update_strategy=None), chunker=SizeChunker(chunk_size=50, incomplete='keep'), fitted=True)
    >>> result = chunked.predict(test)
    """  # noqa: E501

    class Stats(_DriftKNeighborsStats):
        """Statistics from K-Nearest Neighbors drift detection.

        Attributes
        ----------
        p_val : float
            P-value from Mann-Whitney U test on k-NN distances, between 0 and 1.
        mean_ref_distance : float
            Mean k-NN distance in the reference set (baseline).
        mean_test_distance : float
            Mean k-NN distance for the test samples.
        """

    @dataclass
    class Config:
        """
        Configuration for DriftKNeighbors detector.

        Attributes
        ----------
        k : int, default 10
            Number of nearest neighbors.
        distance_metric : {"cosine", "euclidean"}, default "euclidean"
            Distance metric to use.
        p_val : float, default 0.05
            Significance threshold for non-chunked mode.
        extractor : FeatureExtractor or None, default None
            Feature extractor for transforming input data before drift detection.
        update_strategy : UpdateStrategy or None, default None
            Strategy for updating reference data over time.
        """

        k: int = 10
        distance_metric: Literal["cosine", "euclidean"] = "euclidean"
        p_val: float = 0.05
        extractor: FeatureExtractor | None = None
        update_strategy: UpdateStrategy | None = None

    def __init__(
        self,
        k: int | None = None,
        distance_metric: Literal["cosine", "euclidean"] | None = None,
        p_val: float | None = None,
        extractor: FeatureExtractor | None = None,
        update_strategy: UpdateStrategy | None = None,
        config: Config | None = None,
    ) -> None:
        base_config = config or DriftKNeighbors.Config()

        k = k if k is not None else base_config.k
        distance_metric = distance_metric if distance_metric is not None else base_config.distance_metric
        self._p_val = p_val if p_val is not None else base_config.p_val
        extractor = extractor if extractor is not None else base_config.extractor
        update_strategy = update_strategy if update_strategy is not None else base_config.update_strategy

        self.config: DriftKNeighbors.Config = DriftKNeighbors.Config(
            k=k,
            distance_metric=distance_metric,
            p_val=self._p_val,
            extractor=extractor,
            update_strategy=update_strategy,
        )

        # Initialise base + mixins
        BaseDrift.__init__(self)
        self._init_adaptive(extractor=extractor, update_strategy=update_strategy)

        self._scorer = KNeighborsScorer(k, distance_metric)
        self._metric_name = "knn_distance"
        self._ref_mean: float = 0.0
        self._ref_std: float = 1.0

    def fit(self, reference_data: Any) -> Self:
        """Fit the k-NN drift detector on reference data.

        Parameters
        ----------
        reference_data : Any
            Reference data. When an extractor is configured, this can be
            any data type accepted by the extractor (e.g., a dataset or
            raw images). Otherwise, must be array-like with shape
            ``(n_samples, n_features)``.

        Returns
        -------
        Self
        """
        self._set_adaptive_data(reference_data)
        ref = self.reference_data  # lazily encoded + flattened
        self.n_features: int = ref.shape[1]

        # Fit the scorer on the full reference set
        self._scorer.fit(ref)

        # Store reference distribution stats for z-test
        ref_scores = self._scorer.reference_scores
        self._ref_mean = float(np.mean(ref_scores))
        self._ref_std = float(np.std(ref_scores))

        self._fitted = True
        return self

    def _compute_chunk_metric(self, chunk_data: NDArray[np.float32]) -> float:
        """Compute mean k-NN distance for a chunk of data."""
        scores = self._scorer.score(chunk_data)
        return float(np.mean(scores))

    def _compute_chunk_baselines(
        self,
        chunks: list[NDArray[np.float32]],
    ) -> NDArray[np.float32]:
        """Compute per-chunk k-NN distances from reference self-scores.

        Uses the per-sample reference scores (which properly exclude self
        via leave-one-out) aggregated per chunk. This avoids the
        self-inclusion bias that would occur if chunks were re-scored
        against the full reference (where each sample is its own
        nearest neighbor with distance 0).

        Chunks are ordered contiguous slices of reference data, so we
        split ``reference_scores`` by chunk sizes to get the correct
        per-sample scores for each chunk.
        """
        ref_scores = self._scorer.reference_scores
        baselines: list[float] = []
        offset = 0
        for chunk in chunks:
            n = len(chunk)
            baselines.append(float(np.mean(ref_scores[offset : offset + n])))
            offset += n
        return np.array(baselines, dtype=np.float32)

    def _default_chunk_threshold(self) -> Threshold:
        return ZScoreThreshold()

    @set_metadata
    def predict(self, data: Any) -> DriftOutput["DriftKNeighbors.Stats"]:
        """
        Predict whether test data has drifted from reference data.

        Parameters
        ----------
        data : Any
            Test data. When an extractor is configured, this can be
            any data type accepted by the extractor. Otherwise, must be
            array-like.

        Returns
        -------
        DriftOutput[DriftKNeighbors.Stats]
            Drift prediction with k-NN statistics.
        """
        if not self._fitted:
            raise NotFittedError("Must call fit() before predict().")

        x_test = self._prepare_data(data)
        if x_test.shape[1] != self.n_features:
            raise ShapeMismatchError("Reference and test embeddings have different number of features")

        test_scores = self._scorer.score(x_test)
        mean_test = float(np.mean(test_scores))

        # Mann-Whitney U test: are test distances stochastically greater
        # than reference self-distances?  This is a proper two-sample rank
        # test that compares the full distributions of per-sample k-NN
        # distances without an arbitrary effective-sample-size cap.
        _, p_val = mannwhitneyu(test_scores, self._scorer.reference_scores, alternative="greater")
        p_val = float(p_val)

        drifted = p_val < self._p_val

        self._apply_update_strategy(data)

        return DriftOutput(
            drifted=drifted,
            threshold=self._p_val,
            distance=mean_test,
            metric_name="knn_distance",
            details=_DriftKNeighborsStats(
                p_val=p_val,
                mean_ref_distance=self._ref_mean,
                mean_test_distance=mean_test,
            ),
        )
