"""
This module contains implementations of Image Outlier Detection methods
created by Alibi Detect
"""

import math
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, Tuple

import numpy as np
import tensorflow as tf

from daml._internal.datasets import DamlDataset
from daml._internal.metrics.base import Metric, Threshold, ThresholdType
from daml._internal.metrics.outputs import OutlierDetectorOutput


class AlibiDetectOutlierType(str, Enum):
    INSTANCE = "instance"
    FEATURE = "feature"

    def __str__(self) -> str:
        return str.__str__(self)  # pragma: no cover


class _AlibiDetectMetric(Metric, ABC):
    """
    Base class for all outlier detection metrics in alibi-detect

    Attributes
    ----------
    detector : Any, default None
        A model used for outlier detection after being trained on clean data
    is_trained : bool, default False
        Flag to determine if the detector has been trained on the dataset
    """

    # TODO: Add model loading & saving
    @staticmethod
    def _update_kwargs_with_locals(kwargs_to_update, **kwargs):
        kwargs_to_update.update(
            {k: v for k, v in kwargs.items() if k != "self" and v is not None}
        )

    @abstractmethod
    def set_model(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_prediction_args(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _get_default_model_kwargs(self) -> dict:
        raise NotImplementedError

    @property
    @abstractmethod
    def _default_predict_kwargs(self) -> dict:
        raise NotImplementedError

    def __init__(
        self,
        alibi_detect_class: type,
        flatten_dataset: bool,
        dataset_type: Optional[type] = None,
    ):
        """
        Constructor method

        Parameters
        ----------
        alibi_detect_class : type
            This is the alibi_detect outlier detection class to instantiate
        flatten_dataset : bool, default False
            Flag to flatten a dataset to shape (B, H * W * C)
        dataset_type : Optional[type], default None
            Type used for formatting the dataset
        """

        self.detector: Any = None
        self.is_trained: bool = False

        self._alibi_detect_class = alibi_detect_class
        self._flatten_dataset = flatten_dataset
        self._dataset_type = dataset_type

        self._model_kwargs = dict()
        self._predict_kwargs = self._default_predict_kwargs
        self._default_batch_size = 64

        self._input_shape: Tuple[int, int, int]

    def initialize_detector(self, input_shape: Tuple[int, int, int]) -> None:
        """
        Initialize the architecture and model weights of the autoencoder.

        Parameters
        ----------
        input_shape : tuple(int)
            The shape (H, W, C) of the dataset that the detector will be constructed
            around. This influences the internal neural network architecture and should
            be the same shape as the dataset that the detector will be trained on.
        """
        self._input_shape = input_shape
        tf.keras.backend.clear_session()

        # code is covered by concrete child classes
        if not any(self._model_kwargs):  # pragma: no cover
            self._model_kwargs.update(self._get_default_model_kwargs())

        # initialize outlier detector using autoencoder network
        self.detector = self._alibi_detect_class(**self._model_kwargs)

    # Train the alibi-detect metric on dataset
    def fit_dataset(
        self,
        dataset: DamlDataset,
        epochs: int = 3,
        threshold: Threshold = Threshold(95.0, ThresholdType.PERCENTAGE),
        batch_size: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        """
        Trains a model on a dataset containing that can be used
        for the detection of outliers in :meth:`evaluate`

        Parameters
        ----------
        dataset : DamlDataset
            An array of images for the model to train on
        epochs : int, default 3
            Number of epochs to train the detector for
        threshold : Threshold, default Threshold(95.0, ThresholdType.PERCENTAGE)
            Sets the expected threshold of an outlier in the dataset
        batch_size : Optional[int], default None
            Batch size override to use during training
        verbose : bool, default False
            Flag to output logs from Alibi-Detect verbose mode

        Raises
        ------
        TypeError
            If the detector has not been initialized or loaded from path

        Note
        ----
        The supplied dataset should contain no outliers for maximum benefit
        """
        if self.detector is None:
            raise TypeError(
                "Tried to evaluate without initializing a detector. \
                    Try calling metric.initialize_detector()"
            )

        # Autoencoders only need images, so extract from dataset and format
        images = self._format_images(dataset.images)
        # Train the autoencoder using only the formatted images
        self.detector.fit(
            images,
            epochs=epochs,
            batch_size=batch_size or self._default_batch_size,
            verbose=verbose,
        )

        # if save_path: save_detector(self.detector, save_path)
        self.is_trained: bool = True

        if threshold.type is ThresholdType.PERCENTAGE:
            self.detector.infer_threshold(
                images,
                threshold_perc=threshold.value,
                batch_size=batch_size or self._default_batch_size,
            )
        else:
            self.detector.threshold = threshold.value

    def _check_dtype(self, images: np.ndarray):
        """
        Check if the dataset dtype fits with the required dtype of the model.
        None is used for any accepted type. Can extend to check for multiple types.

        Parameters
        ----------
        images : np.ndarray
            A numpy ndarray of images in the shape (B, H, W, C)
        dtype : Type
            The preferred dtype of the array

        Raises
        ------
        TypeError
            If the dataset is not of type np.ndarray

            If the dataset is not of the preferred dtype
        """

        if self._dataset_type is None:
            return

        # Use ndarray type conversion function
        if not isinstance(images, np.ndarray):
            raise TypeError("Dataset should be of type: np.ndarray")

        if not images.dtype.type == self._dataset_type:
            raise TypeError(
                f"Dataset values should be of type {self._dataset_type}, \
                not {images.dtype.type}"
            )

    def _check_image_shape(self, images: np.ndarray):
        """
        Verifies that a dataset has the same shape as a reference shape.
        Use this when fitting or evaluating a model on a dataset, to
        ensure that the dataset's dims are compatible with that which the model
        was built for.

        Parameters
        ----------
        image : np.ndarray
            An array of images in shape (B, H, W, C)

        Raises
        -------
        TypeError
            Raised if the dims of the last 3 channels of dataset
            don't line up with reference_input_shape
        """
        if self.detector is None:
            return

        if images.shape[-3:] != self._input_shape:
            raise TypeError(
                f"Model was initialized on dataset shape \
                (W,H,C)={self._input_shape}, \
                but provided dataset has shape (W,H,C)={images.shape[-3:]}"
            )

    def _format_images(self, images: np.ndarray) -> np.ndarray:
        """
        Formats images such that it fits the required datatype and shape

        Parameters
        ----------
        images : np.ndarray
            A numpy array of images in shape (B, H, W, C)

        Returns
        -------
        np.ndarray
            A numpy array of formatted images

        Note
        ----
        - Some metric libraries want a dataset in a particular format
            (e.g. float32, or flattened)
        - Override this to set the standard dataset formatting.
        """
        self._check_image_shape(images)
        self._check_dtype(images)

        if self._flatten_dataset:
            images = np.reshape(images, (len(images), math.prod(np.shape(images[0]))))
        return images

    def evaluate(
        self,
        dataset: DamlDataset,
    ) -> OutlierDetectorOutput:
        """
        Evaluate the outlier detector metric on a dataset.

        Parameters
        ----------
        dataset : DamlDataset
            The dataset to detect outliers on.

        Returns
        -------
        OutlierDetectorOutput
            Outlier mask, and associated feature and instance scores

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

        # Cast and flatten images
        images = self._format_images(dataset.images)

        self._predict_kwargs.update({"X": images})

        predictions = self.detector.predict(**self._predict_kwargs)

        output = OutlierDetectorOutput(
            is_outlier=predictions["data"]["is_outlier"].tolist(),
            feature_score=predictions["data"]["feature_score"],
            instance_score=predictions["data"]["instance_score"],
        )

        return output
