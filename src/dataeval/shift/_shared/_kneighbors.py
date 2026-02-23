"""
Core k-nearest-neighbors math: index building and per-sample scoring.

Shared by OODKNeighbors (per-instance OOD scoring) and DriftKNeighbors (drift detection).
"""

__all__ = []

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors

from dataeval.config import get_max_processes


class KNeighborsScorer:
    """Pure k-NN math: index building + per-sample scoring.

    Parameters
    ----------
    k : int
        Number of nearest neighbors.
    distance_metric : {"cosine", "euclidean"}
        Distance metric for neighbor search.
    """

    def __init__(self, k: int, distance_metric: Literal["cosine", "euclidean"]) -> None:
        self.k = k
        self.distance_metric = distance_metric
        self.reference_embeddings: NDArray[np.float32]
        self.reference_scores: NDArray[np.float32]
        self._nn_model: NearestNeighbors

    def fit(self, embeddings: NDArray[np.float32]) -> None:
        """Validate, build sklearn NearestNeighbors, compute reference self-scores.

        Parameters
        ----------
        embeddings : NDArray[np.float32]
            Reference (in-distribution) embeddings, shape (n_samples, n_features).

        Raises
        ------
        ValueError
            If k >= number of reference embeddings.
        """
        if self.k >= len(embeddings):
            raise ValueError(
                f"k ({self.k}) must be less than number of reference embeddings ({len(embeddings)})",
            )

        self.reference_embeddings = embeddings

        # Build k-NN index
        self._nn_model = NearestNeighbors(
            n_neighbors=self.k,
            metric=self.distance_metric,
            algorithm="auto",
            n_jobs=get_max_processes(),
        )
        self._nn_model.fit(self.reference_embeddings)

        # Compute reference scores: k+1 neighbors (including self), skip self
        distances, _ = self._nn_model.kneighbors(self.reference_embeddings, n_neighbors=self.k + 1)
        self.reference_scores = np.mean(distances[:, 1:], axis=1).astype(np.float32)

    def score(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Per-sample mean k-NN distances.

        Parameters
        ----------
        x : NDArray[np.float32]
            Input embeddings, shape (n_samples, n_features).

        Returns
        -------
        NDArray[np.float32]
            Mean distance to k nearest neighbors, shape (n_samples,).
        """
        distances, _ = self._nn_model.kneighbors(x)
        return np.mean(distances, axis=1).astype(np.float32)
