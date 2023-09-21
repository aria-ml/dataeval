from typing import Iterable, Optional, Tuple, Type

import alibi_detect
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    Dense,
    Flatten,
    InputLayer,
    Reshape,
)
from tensorflow.nn import relu

from daml._internal.metrics.alibi_detect.base import BaseAlibiDetectOD
from daml._internal.metrics.outputs import OutlierDetectorOutput


class AlibiAE(BaseAlibiDetectOD):
    """
    Autoencoder-based outlier detector, from alibi-detect

    Based on https://docs.seldon.io/projects/alibi-detect/en/latest/od/methods/ae.html
    """

    def __init__(self):
        """Constructor method"""

        super().__init__()
        self._DATASET_TYPE: Optional[Type] = None
        self._FLATTEN_DATASET_FLAG: bool = False

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
        encoding_dim = 1024

        # The outlier detector is an encoder/decoder architecture
        # Here we define the encoder
        encoder_net = Sequential(
            [
                InputLayer(input_shape=input_shape),
                Conv2D(
                    64,
                    4,
                    strides=2,
                    padding="same",
                    activation=relu,
                ),
                Conv2D(
                    128,
                    4,
                    strides=2,
                    padding="same",
                    activation=relu,
                ),
                Conv2D(
                    512,
                    4,
                    strides=2,
                    padding="same",
                    activation=relu,
                ),
                Flatten(),
                Dense(encoding_dim),
            ]
        )
        # Here we define the decoder
        decoder_net = Sequential(
            [
                InputLayer(input_shape=(encoding_dim,)),
                Dense(4 * 4 * 128),
                Reshape(target_shape=(4, 4, 128)),
                Conv2DTranspose(
                    256,
                    4,
                    strides=2,
                    padding="same",
                    activation=relu,
                ),
                Conv2DTranspose(
                    64,
                    4,
                    strides=2,
                    padding="same",
                    activation=relu,
                ),
                Flatten(),
                Dense(np.prod(input_shape)),
                Reshape(target_shape=input_shape),
            ]
        )

        # common inputs across encoders...
        self._kwargs.update(
            {
                "threshold": 0.015,  # threshold for outlier score
                "encoder_net": encoder_net,  # can also pass AE model instead
                "decoder_net": decoder_net,  # of separate encoder and decoder
            }
        )
        # initialize outlier detector using autoencoder network
        self.detector = alibi_detect.od.OutlierAE(**self._kwargs)

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

        super().fit_dataset(dataset, epochs, verbose)
        self.detector.infer_threshold(dataset, threshold_perc=95)

    def evaluate(self, dataset: Iterable[float]) -> OutlierDetectorOutput:
        """
        Evaluate the outlier detector metric on a dataset.

        Parameters
        ----------
        dataset : Iterable[float]
            The dataset to detect outliers on.

        Returns
        -------
        :class: OutlierDetectorOutput
            Outlier mask, and associated feature andinstance scores

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
            outlier_type="instance",
            outlier_perc=75,
            return_feature_score=True,
            return_instance_score=True,
        )
        return self._format_results(predictions)
