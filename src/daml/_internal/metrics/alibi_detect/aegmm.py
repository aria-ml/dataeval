from typing import Iterable, Tuple, Type

import alibi_detect
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, InputLayer

from daml._internal.metrics.alibi_detect.base import BaseAlibiDetectOD
from daml._internal.metrics.outputs import OutlierDetectorOutput


class AlibiAEGMM(BaseAlibiDetectOD):
    """
    Gaussian Mixture Model Autoencoder-based outlier detector, from alibi-detect

    The implementation is based on https://docs.seldon.io/projects/alibi-detect/en/latest/examples/od_aegmm_kddcup.html
    """  # noqa E501

    def __init__(self):
        """Constructor method"""

        super().__init__()
        self._FLATTEN_DATASET_FLAG: bool = True
        self._DATASET_TYPE: Type = np.float32

    def initialize_detector(self, input_shape: Tuple[int, int, int]) -> None:
        """
        Initialize the architecture and model weights of the autoencoder.

        Parameters
        ----------
        input_shape : tuple(int)
            The shape (W,H,C) of the dataset
            that the detector will be constructed around. This influences
            the internal neural network architecture.
            This should be the same shape as the dataset that
            the detector will be trained on.

        """
        self._reference_input_shape = input_shape
        tf.keras.backend.clear_session()
        n_features = tf.math.reduce_prod(input_shape)
        latent_dim = 1
        n_gmm = 2  # nb of components in GMM

        # The outlier detector is an encoder/decoder architecture
        # Here we define the encoder
        encoder_net = Sequential(
            [
                InputLayer(input_shape=(n_features,)),
                Dense(60, activation=tf.nn.tanh),
                Dense(30, activation=tf.nn.tanh),
                Dense(10, activation=tf.nn.tanh),
                Dense(latent_dim, activation=None),
            ]
        )
        # Here we define the decoder
        decoder_net = Sequential(
            [
                InputLayer(input_shape=(latent_dim,)),
                Dense(10, activation=tf.nn.tanh),
                Dense(30, activation=tf.nn.tanh),
                Dense(60, activation=tf.nn.tanh),
                Dense(n_features, activation=None),
            ]
        )
        # GMM autoencoders have a density network too
        gmm_density_net = Sequential(
            [
                InputLayer(input_shape=(latent_dim + 2,)),
                Dense(10, activation=tf.nn.tanh),
                Dense(n_gmm, activation=tf.nn.softmax),
            ]
        )

        # common inputs across encoders...
        self._kwargs.update(
            {
                "threshold": None,  # threshold for outlier score
                "encoder_net": encoder_net,  # can also pass AE model instead
                "decoder_net": decoder_net,  # of separate encoder and decoder
                "gmm_density_net": gmm_density_net,
                "n_gmm": n_gmm,
            }
        )
        # initialize outlier detector using autoencoder network
        self.detector = alibi_detect.od.OutlierAEGMM(**self._kwargs)

    def fit_dataset(
        self,
        dataset: Iterable[float],
        epochs: int = 3,
        verbose: bool = False,
    ) -> None:
        """
        Trains a model on a dataset containing that can be used
        for the detection of outliers in :meth:`evaluate`

        Parameters
        ----------
        dataset : Iterable[float]
            An array of images for the model to train on
        epochs : int, default 3
            Number of epochs to train the detector for.
        verbose : bool, default False
            Flag to output logs from Alibi-Detect verbose mode.

        Raises
        ------
        TypeError
            If the detector has not been initialized or loaded from path

        Note
        ----
            The supplied dataset should contain no outliers for maximum benefit
        """

        # Cast and flatten dataset
        dataset = self.format_dataset(
            dataset,
            flatten_dataset=self._FLATTEN_DATASET_FLAG,
            dataset_type=self._DATASET_TYPE,
            reference_input_shape=self._reference_input_shape,
        )

        super().fit_dataset(dataset, epochs, verbose)
        self.detector.infer_threshold(dataset, threshold_perc=95)

    def evaluate(self, dataset: Iterable[float]) -> OutlierDetectorOutput:
        """
        Evaluate the outlier detector metric on a dataset.

        Parameters
        ----------
        dataset : Iterable[float]
            An array of images

        Returns
        -------
        :class:`OutlierDetectorOutput`
            A dataclass containing outlier mask, feature scores
            and instance scores if applicable
        """

        # Cast and flatten dataset
        dataset = self.format_dataset(
            dataset,
            flatten_dataset=self._FLATTEN_DATASET_FLAG,
            dataset_type=self._DATASET_TYPE,
            reference_input_shape=self._reference_input_shape,
        )

        if self.detector is None:
            raise TypeError(
                "Tried to evaluate without initializing a detector. \
                    Try calling metric.initialize_detector()"
            )

        if not self.is_trained:
            raise TypeError(
                "Error: tried to evaluate a metric that is not trained. \
                    Try calling metric.fit_dataset(data)"
            )
        predictions = self.detector.predict(
            dataset,
            return_instance_score=True,
        )
        return self._format_results(predictions)
