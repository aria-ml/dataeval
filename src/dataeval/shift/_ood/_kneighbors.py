from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from dataeval.protocols import FeatureExtractor
from dataeval.shift._ood._base import BaseOOD, ExtractorMixin, OODScoreOutput
from dataeval.shift._shared._kneighbors import KNeighborsScorer


class OODKNeighbors(ExtractorMixin, BaseOOD):
    """
    K-Nearest Neighbors Out-of-Distribution detector.

    Uses average distance to k nearest neighbors in embedding space to detect OOD samples.
    Samples with larger average distances to their k nearest neighbors in the
    reference (in-distribution) set are considered more likely to be OOD.

    Based on the methodology from:
    "Back to the Basics: Revisiting Out-of-Distribution Detection Baselines"
    (Kuan & Mueller, 2022)

    As referenced in:
    "Safe AI for coral reefs: Benchmarking out-of-distribution detection
    algorithms for coral reef image surveys"

    Parameters
    ----------
    k : int, default 10
        Number of nearest neighbors to consider
    distance_metric : "cosine" | "euclidean", default "cosine"
        Distance metric to use
    threshold_perc : float or None, default None
        Percentage of reference data considered normal (0-100).
        Higher values result in more permissive thresholds.
        If None, uses config.threshold_perc (default 95.0).
    extractor : FeatureExtractor or None, default None
        Feature extractor for transforming input data before scoring.
        When provided, raw data is passed through the extractor in both
        :meth:`fit` and :meth:`score`/:meth:`predict`. When None, data
        is used as-is (must be array-like embeddings).
    config : OODKNeighbors.Config or None, default None
        Optional configuration object with default parameters. Parameters
        specified directly in __init__ will override config defaults.

    Examples
    --------
    >>> from dataeval.shift import OODKNeighbors
    >>> import numpy as np
    >>>
    >>> # Create reference embeddings (in-distribution)
    >>> ref_embeddings = np.random.randn(100, 128).astype(np.float32)
    >>>
    >>> # Fit the detector
    >>> detector = OODKNeighbors(k=10, distance_metric="cosine", threshold_perc=95.0)
    >>> detector.fit(ref_embeddings)
    OODKNeighbors(k=10, distance_metric='cosine', threshold_perc=95.0, extractor=None, fitted=True)
    >>>
    >>> # Score new samples
    >>> test_embeddings = np.random.randn(20, 128).astype(np.float32)
    >>> scores = detector.score(test_embeddings)
    >>> predictions = detector.predict(test_embeddings)

    Using configuration:

    >>> config = OODKNeighbors.Config(k=15, distance_metric="euclidean", threshold_perc=99.0)
    >>> detector = OODKNeighbors(config=config)
    >>> detector.fit(ref_embeddings)
    OODKNeighbors(k=15, distance_metric='euclidean', threshold_perc=99.0, extractor=None, fitted=True)
    """

    @dataclass
    class Config:
        """
        Configuration for OODKNeighbors detector.

        Attributes
        ----------
        k : int, default 10
            Number of nearest neighbors to consider.
        distance_metric : {"cosine", "euclidean"}, default "cosine"
            Distance metric to use.
        threshold_perc : float, default 95.0
            Percentage of reference data considered normal.
        extractor : FeatureExtractor or None, default None
            Feature extractor for transforming input data before scoring.
        """

        k: int = 10
        distance_metric: Literal["cosine", "euclidean"] = "cosine"
        threshold_perc: float = 95.0
        extractor: FeatureExtractor | None = None

    def __init__(
        self,
        k: int | None = None,
        distance_metric: Literal["cosine", "euclidean"] | None = None,
        threshold_perc: float | None = None,
        extractor: FeatureExtractor | None = None,
        config: Config | None = None,
    ) -> None:
        # Store config or create default
        base_config = config or OODKNeighbors.Config()

        # Use config defaults if parameters not specified
        threshold_perc = threshold_perc if threshold_perc is not None else base_config.threshold_perc
        super().__init__(threshold_perc)

        self.k: int = k if k is not None else base_config.k
        self.distance_metric: Literal["cosine", "euclidean"] = (
            distance_metric if distance_metric is not None else base_config.distance_metric
        )
        self._extractor = extractor if extractor is not None else base_config.extractor
        self.config: OODKNeighbors.Config = OODKNeighbors.Config(
            k=self.k, distance_metric=self.distance_metric, threshold_perc=threshold_perc, extractor=self._extractor
        )
        self._scorer = KNeighborsScorer(self.k, self.distance_metric)

    @property
    def reference_embeddings(self) -> NDArray[np.float32]:
        """Reference embeddings stored by the scorer."""
        return self._scorer.reference_embeddings

    def fit(self, reference_data: Any) -> Self:
        """
        Fit the detector using reference (in-distribution) data.

        Builds a k-NN index for efficient nearest neighbor search and
        computes reference scores for automatic thresholding.

        Parameters
        ----------
        reference_data : Any
            Reference (in-distribution) data. When an extractor is
            configured, this can be any data type accepted by the
            extractor. Otherwise, must be array-like embeddings.

        Returns
        -------
        Self
            The fitted detector (for method chaining).
        """
        reference_data_np = self._preprocess(reference_data)

        # Store data info for validation
        self._data_info = (reference_data_np.shape[1:], reference_data_np.dtype.type)

        # Delegate math to scorer
        self._scorer.fit(reference_data_np)

        # Compute reference scores for automatic thresholding
        self._ref_score = OODScoreOutput(instance_score=self._scorer.reference_scores)
        return self

    def _score(self, x: NDArray[np.float32], batch_size: int | None = None) -> OODScoreOutput:  # noqa: ARG002
        """
        Compute OOD scores for input embeddings.

        Parameters
        ----------
        x : NDArray
            Input embeddings to score
        batch_size : int or None
            Not used by this detector.

        Returns
        -------
        OODScoreOutput
            OOD scores (higher = more likely to be OOD)
        """
        return OODScoreOutput(instance_score=self._scorer.score(x))
