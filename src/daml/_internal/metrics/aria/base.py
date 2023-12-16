from abc import ABC
from typing import Literal, Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras import losses

from daml._internal.datasets.datasets import DamlDataset
from daml._internal.models.autoencoder import ARiAAutoencoder, create_default_model


class _AriaMetric(ABC):
    """Abstract base class for ARiA metrics"""

    def __init__(self, encode: bool):
        """Constructor method"""

        self.encode = encode
        self.autoencoder: Optional[ARiAAutoencoder] = None
        self._is_trained: bool = False

    def fit_dataset(
        self,
        dataset: DamlDataset,
        epochs: int = 3,
    ) -> None:
        """
        Trains a model on a dataset to be used during calculation of metrics.

        Parameters
        ----------
        dataset : DamlDataset
            An array of images for the model to train on
        epochs : int, default 3
            Number of epochs to train the detector for.

        """
        images = dataset.images
        shape = (images.shape[1], images.shape[2], images.shape[3])
        latent_dim = 64
        self.autoencoder = create_default_model(ARiAAutoencoder, shape, latent_dim)

        self.autoencoder.compile(optimizer="adam", loss=losses.MeanSquaredError())

        self.autoencoder.fit(
            images,
            images,
            epochs=epochs,
            shuffle=True,
        )
        self._is_trained = True

    def _compute_neighbors(
        self,
        A: np.ndarray,
        B: np.ndarray,
        k: int = 1,
        algorithm: Literal["auto", "ball_tree", "kd_tree"] = "auto",
    ) -> np.ndarray:
        """
        For each sample in A, compute the nearest neighbor in B

        Parameters
        ----------
        A, B : np.ndarray
            The n_samples and n_features respectively
        k : int
            The number of neighbors to find
        algorithm : Literal
            Tree method for nearest neighbor (auto, ball_tree or kd_tree)

        Note
        ----
            Do not use kd_tree if n_features > 20

        Returns
        -------
        List:
            Closest points to each point in A and B

        See Also
        --------
        :func:`sklearn.neighbors.NearestNeighbors`
        """

        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm=algorithm).fit(B)
        nns = nbrs.kneighbors(A)[1]
        nns = nns[:, 1]

        return nns
