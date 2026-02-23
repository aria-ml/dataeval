from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from dataeval.protocols import Array
from dataeval.shift._ood._base import BaseOOD, OODScoreOutput
from dataeval.shift._shared._kneighbors import KNeighborsScorer


class OODKNeighbors(BaseOOD):
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
    >>> detector = OODKNeighbors(k=10, distance_metric="cosine")
    >>> detector.fit(ref_embeddings, threshold_perc=95.0)
    >>>
    >>> # Score new samples
    >>> test_embeddings = np.random.randn(20, 128).astype(np.float32)
    >>> scores = detector.score(test_embeddings)
    >>> predictions = detector.predict(test_embeddings)

    Using configuration:

    >>> config = OODKNeighbors.Config(k=15, distance_metric="euclidean", threshold_perc=99.0)
    >>> detector = OODKNeighbors(config=config)
    >>> detector.fit(ref_embeddings)  # Uses config.threshold_perc
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
        """

        k: int = 10
        distance_metric: Literal["cosine", "euclidean"] = "cosine"
        threshold_perc: float = 95.0

    def __init__(
        self,
        k: int | None = None,
        distance_metric: Literal["cosine", "euclidean"] | None = None,
        config: Config | None = None,
    ) -> None:
        super().__init__()

        # Store config or create default
        self.config: OODKNeighbors.Config = config or OODKNeighbors.Config()

        # Use config defaults if parameters not specified
        self.k: int = k if k is not None else self.config.k
        self.distance_metric: Literal["cosine", "euclidean"] = (
            distance_metric if distance_metric is not None else self.config.distance_metric
        )
        self._scorer = KNeighborsScorer(self.k, self.distance_metric)

    @property
    def reference_embeddings(self) -> NDArray[np.float32]:
        """Reference embeddings stored by the scorer."""
        return self._scorer.reference_embeddings

    def fit(self, embeddings: Array, threshold_perc: float | None = None) -> None:
        """
        Fit the detector using reference (in-distribution) embeddings.

        Builds a k-NN index for efficient nearest neighbor search and
        computes reference scores for automatic thresholding.

        Parameters
        ----------
        embeddings : Array
            Reference (in-distribution) embeddings
        threshold_perc : float or None, default None
            Percentage of reference data considered normal (0-100).
            Higher values result in more permissive thresholds.
            If None, uses config.threshold_perc (default 95.0).
        """
        # Use config default if not specified
        threshold_perc = threshold_perc if threshold_perc is not None else self.config.threshold_perc

        embeddings_np = np.asarray(embeddings, dtype=np.float32)

        # Store data info for validation
        self._data_info = (embeddings_np.shape[1:], embeddings_np.dtype.type)

        # Delegate math to scorer
        self._scorer.fit(embeddings_np)

        # Compute reference scores for automatic thresholding
        self._ref_score = OODScoreOutput(instance_score=self._scorer.reference_scores)
        self._threshold_perc = threshold_perc

    def _score(self, x: NDArray[np.float32], batch_size: int = int(1e10)) -> OODScoreOutput:  # noqa: ARG002
        """
        Compute OOD scores for input embeddings.

        Parameters
        ----------
        x : NDArray
            Input embeddings to score
        batch_size : int
            Not used by this detector.

        Returns
        -------
        OODScoreOutput
            OOD scores (higher = more likely to be OOD)
        """
        return OODScoreOutput(instance_score=self._scorer.score(x))
