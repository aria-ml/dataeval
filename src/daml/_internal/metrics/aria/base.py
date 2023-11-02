from typing import Literal, Optional

import numpy as np
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

from daml._internal.datasets.datasets import DamlDataset
from daml._internal.metrics.base import Metric


class _AriaMetric(Metric):
    """Abstract base class for ARiA metrics"""

    def __init__(self, encode: bool):
        """Constructor method"""

        self.is_trained: bool = False
        self.encode = encode
        self.autoencoder: Optional[ARiAAutoencoder] = None

    def fit_dataset(
        self,
        dataset: DamlDataset,
        epochs: int = 3,
    ) -> None:
        """
        Trains a model on a dataset containing that can be used
        for the detection of outliers in :method:`evaluate`

        Parameters
        ----------
        dataset : DamlDataset
            An array of images for the model to train on
        epochs : int, default 3
            Number of epochs to train the detector for.

        """
        images = dataset.images
        shape = images.shape[1:]
        latent_dim = 64
        self.autoencoder = ARiAAutoencoder(latent_dim, shape)

        self.autoencoder.compile(optimizer="adam", loss=losses.MeanSquaredError())

        self.autoencoder.fit(
            images,
            images,
            epochs=epochs,
            shuffle=True,
        )
        self.is_trained = True

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


class ARiAAutoencoder(Model):
    """
    Basic encoder for ARiA data metrics

    Attributes
    ----------
    encoder : tf.keras.Model
        The internal encoder
    decoder : tf.keras.Model
        The internal decoder

    See also
    ----------
    https://www.tensorflow.org/tutorials/generative/autoencoder
    """

    def __init__(self, latent_dim, shape):
        super(ARiAAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        self.encoder = tf.keras.Sequential(
            [
                layers.Flatten(),
                layers.Dense(latent_dim, activation="relu"),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                layers.Dense(tf.math.reduce_prod(shape), activation="sigmoid"),
                layers.Reshape(shape),
            ]
        )

    def call(self, x):
        """
        Override for TensorFlow model call method

        Parameters
        ----------
        x : Iterable[float]
            Data to run through the model

        Returns
        -------
        decoded : Iterable[float]
            The resulting data after x is run through the encoder and decoder.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
