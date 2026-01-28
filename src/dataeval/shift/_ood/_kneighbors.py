from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors

from dataeval.config import get_max_processes
from dataeval.protocols import Array, ArrayLike
from dataeval.shift._ood._base import OODOutput, OODScoreOutput
from dataeval.types import set_metadata
from dataeval.utils.arrays import as_numpy, to_numpy


class OODKNeighbors:
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
        # Store config or create default
        self.config: OODKNeighbors.Config = config or OODKNeighbors.Config()

        # Use config defaults if parameters not specified
        self.k: int = k if k is not None else self.config.k
        self.distance_metric: Literal["cosine", "euclidean"] = (
            distance_metric if distance_metric is not None else self.config.distance_metric
        )
        self._nn_model: NearestNeighbors
        self._ref_score: OODScoreOutput
        self._threshold_perc: float
        self._data_info: tuple[tuple, type] | None = None
        self.reference_embeddings: NDArray[np.float32]

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

        self.reference_embeddings = np.asarray(embeddings, dtype=np.float32)

        # Validate inputs
        if not isinstance(self.reference_embeddings, np.ndarray):
            raise TypeError("Embeddings should be of type: `NDArray`.")
        if self.k >= len(self.reference_embeddings):
            raise ValueError(
                f"k ({self.k}) must be less than number of reference embeddings ({len(self.reference_embeddings)})"
            )

        # Store data info for validation
        self._data_info = (self.reference_embeddings.shape[1:], self.reference_embeddings.dtype.type)

        # Build k-NN index using sklearn
        self._nn_model = NearestNeighbors(
            n_neighbors=self.k,
            metric=self.distance_metric,
            algorithm="auto",  # Let sklearn choose the best algorithm
            n_jobs=get_max_processes(),
        )
        self._nn_model.fit(self.reference_embeddings)

        # Compute reference scores for automatic thresholding
        ref_scores = self._compute_reference_scores()
        self._ref_score = OODScoreOutput(instance_score=ref_scores)
        self._threshold_perc = threshold_perc

    def _compute_reference_scores(self) -> NDArray[np.float32]:
        """Efficiently compute reference scores by excluding self-matches."""
        # Find k+1 neighbors (including self) for reference points
        distances, _ = self._nn_model.kneighbors(self.reference_embeddings, n_neighbors=self.k + 1)
        # Skip first neighbor (self with distance 0) and average the rest
        return np.mean(distances[:, 1:], axis=1).astype(np.float32)

    def _validate(self, X: NDArray) -> None:
        """Validate that input data matches expected shape and dtype."""
        if not isinstance(X, np.ndarray):
            raise TypeError("Dataset should be of type: `NDArray`.")

        check_data_info = (X.shape[1:], X.dtype.type)
        if self._data_info is not None and check_data_info != self._data_info:
            raise RuntimeError(
                f"Expect data of type: {self._data_info[1]} and shape: {self._data_info[0]}. "
                f"Provided data is type: {check_data_info[1]} and shape: {check_data_info[0]}."
            )

    def _validate_state(self, X: NDArray) -> None:
        """Validate that detector has been fitted and data is valid."""
        if not hasattr(self, "_ref_score") or not hasattr(self, "_threshold_perc"):
            raise RuntimeError("Detector needs to be `fit` before calling predict or score.")
        self._validate(X)

    def _score(self, X: NDArray[np.float32]) -> OODScoreOutput:
        """
        Compute OOD scores for input embeddings.

        Parameters
        ----------
        X : NDArray
            Input embeddings to score

        Returns
        -------
        OODScoreOutput
            OOD scores (higher = more likely to be OOD)
        """
        # Compute OOD scores using sklearn's efficient k-NN search
        distances, _ = self._nn_model.kneighbors(X)
        return OODScoreOutput(instance_score=np.mean(distances, axis=1).astype(np.float32))

    @set_metadata
    def score(self, X: ArrayLike, batch_size: int = int(1e10)) -> OODScoreOutput:
        """
        Compute the :term:`out of distribution<Out-of-distribution (OOD)>` scores for a given dataset.

        Parameters
        ----------
        X : ArrayLike
            Input embeddings to score.
        batch_size : int, default 1e10
            Batch size parameter for API consistency. Not used by this detector.

        Returns
        -------
        OODScoreOutput
            An object containing the instance-level OOD scores.
            Higher scores indicate samples are more likely to be OOD.
        """
        X_np = as_numpy(X).astype(np.float32)
        self._validate(X_np)
        return self._score(X_np)

    def _threshold_score(self, ood_type: Literal["feature", "instance"] = "instance") -> np.floating:
        """Get the threshold score for a given OOD type."""
        return np.percentile(self._ref_score.get(ood_type), self._threshold_perc)

    @set_metadata
    def predict(
        self,
        X: ArrayLike,
        batch_size: int = int(1e10),
        ood_type: Literal["feature", "instance"] = "instance",
    ) -> OODOutput:
        """
        Predict whether instances are :term:`out of distribution<Out-of-distribution (OOD)>` or not.

        Parameters
        ----------
        X : ArrayLike
            Input embeddings for out-of-distribution prediction.
        batch_size : int, default 1e10
            Batch size parameter for API consistency. Not used by this detector.
        ood_type : "feature" | "instance", default "instance"
            OOD type parameter for API consistency. This detector only supports "instance" level.

        Returns
        -------
        OODOutput
            Dictionary containing:
            - is_ood: Boolean array indicating which samples are OOD
            - instance_score: OOD scores for all samples
            - feature_score: None (not supported by this detector)
        """
        X_np = to_numpy(X).astype(np.float32)
        self._validate_state(X_np)

        # Compute OOD scores
        score = self.score(X_np, batch_size=batch_size)
        ood_pred = score.get(ood_type) > self._threshold_score(ood_type)
        return OODOutput(is_ood=ood_pred, **score.data())
