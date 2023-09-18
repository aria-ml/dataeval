"""
This module contains implementations of Image Outlier Detection methods
created by Alibi Detect
"""

from typing import Iterable, Tuple

import alibi_detect
import numpy as np
import tensorflow as tf
from alibi_detect.models.tensorflow import PixelCNN

from daml._internal.metrics.alibi_detect.base import BaseAlibiDetectOD
from daml.metrics.outputs import OutlierDetectorOutput


class AlibiLLR(BaseAlibiDetectOD):
    """
    Log likelihood Ratio (LLR) outlier detector, from alibi-detect
    Based on https://docs.seldon.io/projects/alibi-detect
             /en/latest/examples/od_llr_mnist.html
    """

    def __init__(self):
        "Constructor method"

        super().__init__()
        self._FLATTEN_DATASET_FLAG = False
        self._DATASET_TYPE: type = np.float32

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
        tf.keras.backend.clear_session()
        self._reference_input_shape = input_shape

        # LLR internally uses a Pixel CNN architecture,
        # which we initialize here
        llr_model = PixelCNN(
            image_shape=input_shape,
            num_resnet=5,
            num_hierarchies=2,
            num_filters=32,
            num_logistic_mix=1,
            receptive_field_dims=(3, 3),
            dropout_p=0.3,
            l2_weight=0.0,
        )

        # common inputs across encoders...
        self._kwargs.update(
            {
                "threshold": None,  # threshold for outlier score
                "model": llr_model,
            }
        )
        # initialize outlier detector using autoencoder network
        self.detector = alibi_detect.od.LLR(**self._kwargs)

    def fit_dataset(
        self,
        dataset: Iterable[float],
        epochs: int = 3,
        verbose: bool = False,
    ) -> None:
        """
        Trains a model on a dataset containing that can be used
        for the detection of outliers in :method:`evaluate`

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

        .. note::
            The supplied dataset should contain no outliers for maximum benefit
        """

        # Cast and flatten dataset
        dataset = self.format_dataset(
            dataset,
            flatten_dataset=self._FLATTEN_DATASET_FLAG,
            dataset_type=self._DATASET_TYPE,
        )

        super().fit_dataset(dataset, epochs, verbose)
        self.detector.infer_threshold(
            dataset,
            threshold_perc=95,
            batch_size=32,
        )

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

        dataset = self.format_dataset(
            dataset,
            flatten_dataset=self._FLATTEN_DATASET_FLAG,
            dataset_type=self._DATASET_TYPE,
            reference_input_shape=self._reference_input_shape,
        )

        predictions = self.detector.predict(
            dataset,
            outlier_type="instance",
            return_instance_score=True,
        )
        return self._format_results(predictions)
