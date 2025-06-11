from typing import Literal

import numpy as np
from sklearn.neighbors import NearestNeighbors

from dataeval.data import Embeddings
from dataeval.detectors.ood.base import EmbeddingBasedOODBase
from dataeval.outputs._ood import OODScoreOutput
from dataeval.typing import ArrayLike


class OOD_KNN(EmbeddingBasedOODBase):
    """
    K-Nearest Neighbors Out-of-Distribution detector.

    Uses average cosine distance to k nearest neighbors in embedding space to detect OOD samples.
    Samples with larger average distances to their k nearest neighbors in the
    reference (in-distribution) set are considered more likely to be OOD.

    Based on the methodology from:
    "Back to the Basics: Revisiting Out-of-Distribution Detection Baselines"
    (Kuan & Mueller, 2022)

    As referenced in:
    "Safe AI for coral reefs: Benchmarking out-of-distribution detection
    algorithms for coral reef image surveys"
    """

    def __init__(self, k: int = 10, distance_metric: Literal["cosine", "euclidean"] = "cosine") -> None:
        """
        Initialize KNN OOD detector.

        Args:
            k: Number of nearest neighbors to consider (default: 10)
            distance_metric: Distance metric to use ('cosine' or 'euclidean')
        """
        super().__init__()
        self.k = k
        self.distance_metric = distance_metric
        self._nn_model: NearestNeighbors
        self.reference_embeddings: ArrayLike

    def fit_embeddings(self, embeddings: Embeddings, threshold_perc: float = 95.0) -> None:
        """
        Fit the detector using reference (in-distribution) embeddings.

        Builds a k-NN index for efficient nearest neighbor search and
        computes reference scores for automatic thresholding.

        Args:
            embeddings: Reference embeddings from in-distribution data
            threshold_perc: Percentage of reference data considered normal
        """
        self.reference_embeddings = embeddings.to_numpy()

        if self.k >= len(self.reference_embeddings):
            raise ValueError(
                f"k ({self.k}) must be less than number of reference embeddings ({len(self.reference_embeddings)})"
            )

        # Build k-NN index using sklearn
        self._nn_model = NearestNeighbors(
            n_neighbors=self.k,
            metric=self.distance_metric,
            algorithm="auto",  # Let sklearn choose the best algorithm
        )
        self._nn_model.fit(self.reference_embeddings)

        # efficiently compute reference scores for automatic thresholding
        ref_scores = self._compute_reference_scores()
        self._ref_score = OODScoreOutput(instance_score=ref_scores)
        self._threshold_perc = threshold_perc
        self._data_info = self._get_data_info(self.reference_embeddings)

    def _compute_reference_scores(self) -> np.ndarray:
        """Efficiently compute reference scores by excluding self-matches."""
        # Find k+1 neighbors (including self) for reference points
        distances, _ = self._nn_model.kneighbors(self.reference_embeddings, n_neighbors=self.k + 1)
        # Skip first neighbor (self with distance 0) and average the rest
        return np.mean(distances[:, 1:], axis=1)

    def _score(self, X: np.ndarray, batch_size: int = int(1e10)) -> OODScoreOutput:
        """
        Compute OOD scores for input embeddings.

        Args:
            X: Input embeddings to score
            batch_size: Batch size (not used, kept for interface compatibility)

        Returns:
            OODScoreOutput containing instance-level scores
        """
        # Compute OOD scores using sklearn's efficient k-NN search
        distances, _ = self._nn_model.kneighbors(X)
        return OODScoreOutput(instance_score=np.mean(distances, axis=1))
