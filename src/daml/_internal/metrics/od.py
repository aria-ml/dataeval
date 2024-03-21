import math
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, NamedTuple, Optional, Tuple

import keras
import numpy as np

from daml import _alibi_detect
from daml._internal.models.tensorflow.alibi import create_model


class OutlierType(str, Enum):
    """
    Enum to determine outliers by instance or by feature
    """

    INSTANCE = "instance"
    FEATURE = "feature"

    def __str__(self) -> str:
        return str.__str__(self)  # pragma: no cover


class ThresholdType(Enum):
    """
    Enum of threshold types for outlier detection
    """

    VALUE = "value"
    PERCENTAGE = "percentage"


class Threshold(NamedTuple):
    """
    NamedTuple to specify the threshold value and type for outlier detection

    Parameters
    ----------
    value : float
        The threshold to determine an outlier
    type : ThresholdType
        Whether the value provided is a value or percentage type
    """

    value: float
    type: ThresholdType


class OD_Base(ABC):
    """
    Base class for all outlier detection metrics in alibi-detect
    """

    @staticmethod
    def _update_kwargs_with_locals(kwargs_to_update, **kwargs):
        kwargs_to_update.update(
            {k: v for k, v in kwargs.items() if k != "self" and v is not None}
        )

    @abstractmethod
    def set_prediction_args(self) -> None:
        """Abstract method to set specific prediction arguments for the detector"""

    @property
    @abstractmethod
    def _default_predict_kwargs(self) -> dict:
        """Abstract method for the default prediction arguments for the detector"""

    def __init__(
        self,
        alibi_detect_class: type,
        model_param_name: str,
        flatten_dataset: bool,
        dataset_type: Optional[type] = None,
    ):
        """
        Constructor method

        Parameters
        ----------
        detector : Any, default None
            A model used for outlier detection after being trained on clean data
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
        self._model_param_name = model_param_name
        self._flatten_dataset = flatten_dataset
        self._dataset_type = dataset_type

        self._model_kwargs = {}
        self._predict_kwargs = self._default_predict_kwargs
        self._default_batch_size = 64

        self._input_shape: Tuple[int, int, int]

    # Train the alibi-detect metric on dataset
    def fit_dataset(
        self,
        images: np.ndarray,
        epochs: int = 3,
        threshold: Optional[Threshold] = None,
        batch_size: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        """
        Trains a model on a dataset containing that can be used
        for the detection of outliers in :meth:`evaluate`

        Parameters
        ----------
        images : np.ndarray
            A numpy array of images for the model to train on
        epochs : int, default 3
            Number of epochs to train the detector for
        threshold : Threshold, default None
            Sets the expected threshold of an outlier in the dataset
            If None, uses Threshold(95.0, ThresholdType.PERCENTAGE)
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
        keras.backend.clear_session()
        if self.detector is None:
            self._input_shape = images[0].shape
            model = create_model(type(self).__name__, self._input_shape)
            self.detector = self._alibi_detect_class(
                threshold=0,
                **{self._model_param_name: model},
            )

        # Autoencoders only need images, so extract from dataset and format
        images = self._format_images(images)
        # Train the autoencoder using only the formatted images
        self.detector.fit(
            images,
            epochs=epochs,
            batch_size=batch_size or self._default_batch_size,
            verbose=verbose,
        )

        # if save_path: save_detector(self.detector, save_path)
        self.is_trained: bool = True

        if threshold is None:
            threshold = Threshold(95.0, ThresholdType.PERCENTAGE)

        if threshold.type is ThresholdType.PERCENTAGE:
            self.detector.infer_threshold(
                images,
                threshold_perc=threshold.value,
                batch_size=batch_size or self._default_batch_size,
            )
        else:
            self.detector.threshold = threshold.value

    def export_model(self, path: str):
        if self.detector is None:
            raise RuntimeError("The model must be initialized first")
        if self._model_param_name not in self.detector.__dict__:
            raise ValueError("Member not found in detector")
        getattr(self.detector, self._model_param_name).save(path)

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
            raise ValueError(
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
        images: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Evaluate the outlier detector metric on a dataset.

        Parameters
        ----------
        images : np.ndarray
            The images to detect outliers on.

        Returns
        -------
        Dict[str, np.ndarray]
            is_outlier
            feature_score
            instance_score
        """
        if self.detector is None or not self.is_trained:
            raise TypeError(
                "Error: tried to evaluate a metric that is not trained. \
                    Try calling metric.fit_dataset(data)"
            )

        # Cast and flatten images
        images = self._format_images(images)

        self._predict_kwargs.update({"X": images})

        predictions = self.detector.predict(**self._predict_kwargs)

        output = {
            "is_outlier": predictions["data"]["is_outlier"].tolist(),
            "feature_score": predictions["data"]["feature_score"],
            "instance_score": predictions["data"]["instance_score"],
        }

        return output


class OD_AE(OD_Base):
    """
    Autoencoder-based outlier detector, using `alibi-detect ae. <https://docs.seldon.io/projects/alibi-detect/en/latest/od/methods/ae.html>`_
    """

    def __init__(self):
        super().__init__(
            alibi_detect_class=_alibi_detect.od.OutlierAE,
            model_param_name="ae",
            flatten_dataset=False,
            dataset_type=None,
        )

    def set_prediction_args(
        self,
        outlier_type: Optional[OutlierType] = None,
        outlier_perc: Optional[float] = None,
        return_feature_score: Optional[bool] = None,
        return_instance_score: Optional[bool] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        """
        Sets additional arguments to be used during prediction.

        Note
        ----
        Visit `alibi-detect ae <https://docs.seldon.io/projects/alibi-detect/en/latest/od/methods/ae.html#Detect>`_ for additional information on prediction parameters.
        """  # noqa E501
        self._update_kwargs_with_locals(self._predict_kwargs, **locals())

    @property
    def _default_predict_kwargs(self) -> dict:
        return {
            "outlier_type": OutlierType.INSTANCE,
            "outlier_perc": 75,
            "return_feature_score": True,
            "return_instance_score": True,
            "batch_size": 64,
        }


class OD_AEGMM(OD_Base):
    """
    Gaussian Mixture Model Autoencoder-based outlier detector,
    using alibi-detect aegmm. `<https://docs.seldon.io/projects/alibi-detect/en/latest/od/methods/aegmm.html>`_


    The model used by this class is :py:class:`daml.models.AEGMM`
    """  # E501

    def __init__(self):
        super().__init__(
            alibi_detect_class=_alibi_detect.od.OutlierAEGMM,
            model_param_name="aegmm",
            flatten_dataset=True,
            dataset_type=np.float32,
        )

    def set_prediction_args(
        self,
        return_instance_score: Optional[bool] = None,
    ) -> None:
        """
        Sets additional arguments to be used during prediction.

        Note
        ----
        Visit `alibi-detect aegmm <https://docs.seldon.io/projects/alibi-detect/en/latest/od/methods/aegmm.html#Detect>`_ for additional information on prediction parameters.
        """  # noqa E501
        self._update_kwargs_with_locals(self._predict_kwargs, **locals())

    @property
    def _default_predict_kwargs(self) -> dict:
        return {
            "return_instance_score": True,
            "batch_size": 64,
        }


class OD_LLR(OD_Base):
    """
    Log likelihood Ratio (LLR) outlier detector,
    using `alibi-detect llr. <https://docs.seldon.io/projects/alibi-detect/en/latest/examples/od_llr_mnist.html>`_


    The model used by this class is :py:class:`daml.models.LLR`
    """  # E501

    def __init__(self):
        super().__init__(
            alibi_detect_class=_alibi_detect.od.LLR,
            model_param_name="model",
            flatten_dataset=False,
            dataset_type=np.float32,
        )

    def set_prediction_args(
        self,
        outlier_type: Optional[OutlierType] = None,
        return_instance_score: Optional[bool] = None,
    ) -> None:
        """
        Sets additional arguments to be used during prediction.

        Note
        ----
        Visit `alibi-detect llr <https://docs.seldon.io/projects/alibi-detect/en/latest/od/methods/llr.html#Detect>`_ for additional information on prediction parameters.
        """  # noqa E501
        self._update_kwargs_with_locals(self._predict_kwargs, **locals())

    @property
    def _default_predict_kwargs(self) -> dict:
        return {
            "outlier_type": OutlierType.INSTANCE,
            "return_instance_score": True,
            "batch_size": 64,
        }


class OD_VAE(OD_Base):
    """
    Variational Autoencoder-based outlier detector,
    using `alibi-detect vae. <https://docs.seldon.io/projects/alibi-detect/en/latest/od/methods/vae.html>`_

    The model used by this class is :py:class:`daml.models.VAE`
    """  # E501

    def __init__(self):
        super().__init__(
            alibi_detect_class=_alibi_detect.od.OutlierVAE,
            model_param_name="vae",
            flatten_dataset=False,
            dataset_type=None,
        )

    def set_prediction_args(
        self,
        outlier_type: Optional[OutlierType] = None,
        outlier_perc: Optional[float] = None,
        return_feature_score: Optional[bool] = None,
        return_instance_score: Optional[bool] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        """
        Sets additional arguments to be used during prediction.

        Note
        ----
        Visit `alibi-detect vae <https://docs.seldon.io/projects/alibi-detect/en/latest/od/methods/vae.html#Detect>`_ for additional information on prediction parameters.
        """  # noqa E501
        self._update_kwargs_with_locals(self._predict_kwargs, **locals())

    @property
    def _default_predict_kwargs(self) -> dict:
        return {
            "outlier_type": OutlierType.INSTANCE,
            "outlier_perc": 75,
            "return_feature_score": True,
            "return_instance_score": True,
            "batch_size": 64,
        }


class OD_VAEGMM(OD_Base):
    """
    Variational Gaussian Mixture Model Autoencoder-based outlier detector,
    using `alibi-detect vaegmm. <https://docs.seldon.io/projects/alibi-detect/en/latest/od/methods/vaegmm.html>`_


    The model used by this class is :py:class:`daml.models.VAEGMM`
    """  # E501

    def __init__(self):
        super().__init__(
            alibi_detect_class=_alibi_detect.od.OutlierVAEGMM,
            model_param_name="vaegmm",
            flatten_dataset=True,
            dataset_type=np.float32,
        )

    def set_prediction_args(
        self,
        return_instance_score: Optional[bool] = None,
    ) -> None:
        """
        Sets additional arguments to be used during prediction.

        Note
        ----
        Visit `alibi-detect vaegmm <https://docs.seldon.io/projects/alibi-detect/en/latest/od/methods/vaegmm.html#Detect>`_ for additional information on prediction parameters.
        """  # noqa E501
        self._update_kwargs_with_locals(self._predict_kwargs, **locals())

    @property
    def _default_predict_kwargs(self) -> dict:
        return {"return_instance_score": True, "batch_size": 64}
