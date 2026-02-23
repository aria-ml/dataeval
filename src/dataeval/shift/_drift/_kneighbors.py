"""K-Nearest Neighbors based drift detector."""

__all__ = []

from dataclasses import dataclass
from typing import Literal, TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import mannwhitneyu
from typing_extensions import Self

from dataeval.protocols import Threshold
from dataeval.shift._drift._base import BaseDrift, DriftChunkerMixin, DriftOutput
from dataeval.shift._drift._chunk import BaseChunker
from dataeval.shift._shared._kneighbors import KNeighborsScorer
from dataeval.types import set_metadata
from dataeval.utils.arrays import flatten_samples
from dataeval.utils.thresholds import ZScoreThreshold


class DriftKNeighborsStats(TypedDict):
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

    p_val: float
    mean_ref_distance: float
    mean_test_distance: float


class DriftKNeighbors(DriftChunkerMixin, BaseDrift):
    """K-Nearest Neighbors based drift detector.

    Detects drift by comparing k-NN distances of test samples against the
    reference set. If test samples are farther from their k nearest neighbors
    in the reference set than expected, drift is detected.

    Uses a fit/predict lifecycle: construct with hyperparameters, call
    :meth:`fit` with reference data, then call :meth:`predict` with test data.

    Supports two modes:

    - **Non-chunked** (default): Computes per-sample k-NN distances for the
      test set and uses a Mann-Whitney U test against the reference baseline
      to produce a p-value. Drift is flagged when ``p_val < p_val_threshold``.
    - **Chunked**: Splits data into chunks, computes mean k-NN distance per
      chunk, and uses threshold bounds to flag drift per chunk.

    Parameters
    ----------
    k : int, default 10
        Number of nearest neighbors.
    distance_metric : {"cosine", "euclidean"}, default "euclidean"
        Distance metric for neighbor search.
    p_val : float, default 0.05
        Significance threshold for non-chunked mode.
    config : DriftKNeighbors.Config or None, default None
        Optional configuration object.

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

    >>> detector = DriftKNeighbors(k=5).fit(ref, chunk_size=50)
    >>> result = detector.predict(test)
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
        """

        k: int = 10
        distance_metric: Literal["cosine", "euclidean"] = "euclidean"
        p_val: float = 0.05

    def __init__(
        self,
        k: int | None = None,
        distance_metric: Literal["cosine", "euclidean"] | None = None,
        p_val: float | None = None,
        config: Config | None = None,
    ) -> None:
        super().__init__()
        self._init_chunking()

        self.config: DriftKNeighbors.Config = config or DriftKNeighbors.Config()

        k = k if k is not None else self.config.k
        distance_metric = distance_metric if distance_metric is not None else self.config.distance_metric
        self._p_val = p_val if p_val is not None else self.config.p_val

        self._scorer = KNeighborsScorer(k, distance_metric)
        self._metric_name = "knn_distance"
        self._ref_mean: float = 0.0
        self._ref_std: float = 1.0

    def fit(
        self,
        x_ref: ArrayLike,
        chunker: BaseChunker | None = None,
        chunk_size: int | None = None,
        chunk_count: int | None = None,
        chunks: list[ArrayLike] | None = None,
        chunk_indices: list[list[int]] | None = None,
        threshold: Threshold | None = None,
    ) -> Self:
        """Fit the k-NN drift detector on reference data.

        Parameters
        ----------
        x_ref : ArrayLike
            Reference data with dim[n_samples, n_features].
        chunker : BaseChunker or None, default None
            Explicit chunker instance for chunked mode.
        chunk_size : int or None, default None
            Create fixed-size chunks.
        chunk_count : int or None, default None
            Split into this many equal chunks.
        chunks : list[ArrayLike] or None, default None
            Pre-split reference data for chunked mode.
        chunk_indices : list[list[int]] or None, default None
            Index groupings for chunking reference data.
        threshold : Threshold or None, default None
            Threshold strategy for chunked mode. Defaults to ZScoreThreshold.

        Returns
        -------
        Self
        """
        self._x_ref = flatten_samples(np.atleast_2d(np.asarray(x_ref, dtype=np.float32)))
        self.n_features: int = self._x_ref.shape[1]

        # Fit the scorer on the full reference set
        self._scorer.fit(self._x_ref)

        # Store reference distribution stats for z-test
        ref_scores = self._scorer.reference_scores
        self._ref_mean = float(np.mean(ref_scores))
        self._ref_std = float(np.std(ref_scores))

        # Handle chunking (prebuilt chunks are converted here)
        if chunks is not None:
            chunks = [flatten_samples(np.atleast_2d(np.asarray(c, dtype=np.float32))) for c in chunks]

        self._resolve_fit_chunks(
            len(self._x_ref),
            chunker=chunker,
            chunk_size=chunk_size,
            chunk_count=chunk_count,
            chunks=chunks,
            chunk_indices=chunk_indices,
            threshold=threshold,
            default_threshold=ZScoreThreshold(),
        )

        self._fitted = True
        return self

    def _fit_chunked_baseline(
        self,
        chunker: BaseChunker,
        threshold: Threshold | None,
        default_threshold: Threshold | None,
    ) -> None:
        """Compute per-chunk k-NN distances from reference self-scores.

        Uses the per-sample reference scores (which properly exclude self
        via leave-one-out) aggregated per chunk.  This avoids the
        self-inclusion bias of the default implementation (where
        ``score()`` would count self as a neighbor with distance 0) and
        keeps the same full-reference index used at prediction time.
        """
        ref_scores = self._scorer.reference_scores
        index_groups = chunker.split(len(ref_scores))
        baseline = np.array(
            [float(np.mean(ref_scores[idx])) for idx in index_groups],
            dtype=np.float32,
        )
        self._resolve_baseline_threshold(baseline, threshold, default_threshold)

    def _compute_chunk_metric(self, chunk_data: NDArray[np.float32]) -> float:
        """Compute mean k-NN distance for a chunk of data."""
        scores = self._scorer.score(chunk_data)
        return float(np.mean(scores))

    @set_metadata
    def predict(
        self,
        x: ArrayLike | None = None,
        chunks: list[ArrayLike] | None = None,
        chunk_indices: list[list[int]] | None = None,
    ) -> DriftOutput:
        """
        Predict whether test data has drifted from reference data.

        Parameters
        ----------
        x : ArrayLike or None
            Test data.
        chunks : list[ArrayLike] or None, default None
            Pre-built test data chunks.
        chunk_indices : list[list[int]] or None, default None
            Index groupings for chunking test data.

        Returns
        -------
        DriftOutput
            Non-chunked mode: ``details`` is a :class:`DriftKNeighborsStats` TypedDict.
            Chunked mode: ``details`` is a :class:`polars.DataFrame` with per-chunk results.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict().")

        if self.is_chunked or chunks is not None or chunk_indices is not None:
            # Prepare data for the mixin's _predict_chunked
            if chunks is not None:
                prepared = [flatten_samples(np.atleast_2d(np.asarray(c, dtype=np.float32))) for c in chunks]
                for c in prepared:
                    if c.shape[1] != self.n_features:
                        raise ValueError("Reference and test embeddings have different number of features")
                return self._predict_chunked(chunks_override=prepared)

            x_test = None
            if x is not None:
                x_test = flatten_samples(np.atleast_2d(np.asarray(x, dtype=np.float32)))
                if x_test.shape[1] != self.n_features:
                    raise ValueError("Reference and test embeddings have different number of features")

            return self._predict_chunked(
                x_test=x_test,
                chunk_indices_override=chunk_indices,
            )

        if x is None:
            raise ValueError("x is required for non-chunked prediction.")
        return self._predict_single(x)

    def _predict_single(self, x: ArrayLike) -> DriftOutput:
        """Non-chunked prediction: Mann-Whitney U test on k-NN distances."""
        x_test = flatten_samples(np.atleast_2d(np.asarray(x, dtype=np.float32)))
        if x_test.shape[1] != self.n_features:
            raise ValueError("Reference and test embeddings have different number of features")

        test_scores = self._scorer.score(x_test)
        mean_test = float(np.mean(test_scores))

        # Mann-Whitney U test: are test distances stochastically greater
        # than reference self-distances?  This is a proper two-sample rank
        # test that compares the full distributions of per-sample k-NN
        # distances without an arbitrary effective-sample-size cap.
        _, p_val = mannwhitneyu(test_scores, self._scorer.reference_scores, alternative="greater")
        p_val = float(p_val)

        drifted = p_val < self._p_val

        return DriftOutput(
            drifted=drifted,
            threshold=self._p_val,
            distance=mean_test,
            metric_name="knn_distance",
            details=DriftKNeighborsStats(
                p_val=p_val,
                mean_ref_distance=self._ref_mean,
                mean_test_distance=mean_test,
            ),
        )
