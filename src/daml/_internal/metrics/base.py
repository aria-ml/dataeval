from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional, Tuple, Type

import numpy as np


class Metric(ABC):
    """Abstract class for all DAML metrics"""

    @abstractmethod
    def evaluate(self, dataset: Iterable[float]) -> Any:
        """Implement to evaluated the dataset against the stored metric"""
        raise NotImplementedError(
            "evaluate must be called by a child class that implements it"
        )


class OutlierDetector(Metric, ABC):
    """
    Abstract class for all DAML outlier detection metrics.

    Attributes
    ----------
    is_trained : bool, default False
        Flag if a model has been trained to the data
    dataset_type : Type, optional
        The required type of a dataset for a model
    FLATTEN_DATASET_FLAG :  bool, default False
        Flag if a dataset should be flattened to shape (B, H * W * C)
    """

    def __init__(self):
        """Constructor method"""

        self.is_trained: bool = False
        self._DATASET_TYPE: Type = None
        self.FLATTEN_DATASET_FLAG: bool = False

    @abstractmethod
    def fit_dataset(
        self,
        dataset: Iterable[float],
        epochs: int,
        verbose: bool,
    ) -> None:
        """Implement to train an instance specific model on the given dataset"""

        raise NotImplementedError(
            "fit_dataset must be called by a child class that implements it"
        )

    @abstractmethod
    def initialize_detector(self) -> Any:
        """Implement to initialize model weights and parameters"""

        raise NotImplementedError(
            "initialize_detector must be called \
            by a child class that implements it"
        )

    def flatten_dataset(self, dataset: np.ndarray) -> np.ndarray:
        """
        Flattens an array of (H, W, C) images into an array of (H*W*C)

        Parameters
        ----------
        dataset : np.ndarray
            An array of images in the shape (B, H, W, C)

        Returns
        -------
        np.ndarray
            An array of iamges in the shape (B, H * W * C)
        """

        return np.reshape(dataset, (len(dataset), np.prod(np.shape(dataset[0]))))

    def check_dtype(self, dataset: np.ndarray, dtype: Type):
        """
        Check if the dataset dtype fits with the required dtype of the model.
        None is used for any accepted type. Can extend to check for multiple types.

        Parameters
        ----------
        dataset : np.ndarray
            A numpy ndarray of images in the shape (B, H, W, C)
        dtype : Type
            The preferred dtype of the array

        Raises
        ------
        TypeError
            If the dataset is not of type np.ndarray

            If the dataset is not of the preferred dtype
        """

        if dtype is None:
            return
        # Use ndarray type conversion function
        if not isinstance(dataset, np.ndarray):
            raise TypeError("Dataset should be of type: np.ndarray")

        if not dataset.dtype.type == dtype:
            raise TypeError(
                f"Dataset values should be of type {dtype}, not {dataset.dtype.type}"
            )

    def check_dataset_shape(self, dataset, reference_input_shape):
        """
        Verifies that a dataset has the same shape as a reference shape.
        Use this when fitting or evaluating a model on a dataset, to
        ensure that the dataset's dims are compatible with that which the model
        was built for.

        Parameters
        ----------
        dataset : Iterable[float]
            An array of images in shape (B, H, W, C)
        reference_input_shape : Tuple[int, int, int]
            Dimensions of one image (H,W,C) to compare the dataset shape to

        Raises
        -------
        TypeError
            Raised if the dims of the last 3 channels of dataset
            don't line up with reference_input_shape
        """
        if dataset.shape[-3:] != reference_input_shape:
            raise TypeError(
                f"Model was initialzied on dataset shape \
                (W,H,C)={reference_input_shape}, \
                but provided dataset has shape (W,H,C)={dataset.shape[-3:]}"
            )

    def format_dataset(
        self,
        dataset: Any,
        flatten_dataset: bool = False,
        dataset_type: Type = None,
        reference_input_shape: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """
        Formats a dataset such that it fits the required datatype and shape

        Parameters
        ----------
        dataset : np.ndarray
            A numpy array of images in shape (B, H, W, C)
        flatten_dataset : bool, default False
            Flag to flatten a dataset to shape (B, H * W * C)
        dataset_type : Type, Optional

        Returns
        -------
        Iterable[float]
            Returns a dataset style array

        Note
        ----
            - Some metric libraries want a dataset in a particular format
                (e.g. float32, or flattened)
            - Override this to set the standard dataset formatting.
        """
        if reference_input_shape:
            self.check_dataset_shape(dataset, reference_input_shape)
        if dataset_type:
            self.check_dtype(dataset, dataset_type)
        if flatten_dataset:
            dataset = self.flatten_dataset(dataset)
        return dataset
