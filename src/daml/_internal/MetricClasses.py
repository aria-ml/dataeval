"""
This module contains Base and Abstract classes for all metric implementations in DAML

Based on example code from
https://docs.seldon.io/projects/alibi-detect/en/latest/od/methods/ae.html#Examples
"""

from abc import ABC, abstractmethod
from typing import Any, Iterable, Type

import numpy as np


class Metrics:
    """A global dictionary to parse metrics, providers, and methods"""

    OutlierDetection = "OutlierDetection"
    Divergence = "Divergence"
    BER = "BER"

    class Provider:
        """Set of libraries that implement methods"""

        AlibiDetect = "Alibi-Detect"
        ARiA = "ARiA"

    class Method:
        """A set of solutions for a given metric"""

        AutoEncoder = "Autoencoder"
        VariationalAutoEncoder = "VAE"
        AutoEncoderGMM = "AEGMM"
        VariationalAutoEncoderGMM = "VAEGMM"
        LLR = "LLR"
        DpDivergence = "Dp_Divergence"
        MultiClassBER = "MultiClassBER"

    class Algorithm:
        """A set of differing ways to calculate a solution"""

        FirstNearestNeighbor = "fnn"
        MinimumSpanningTree = "mst"

    # This might be better referred to as a list of supported operations.
    # Potentially a subset of the above enum classes. Not all permutations
    # may be supported.
    metrics_providers_methods = {
        OutlierDetection: {
            Provider.AlibiDetect: [
                Method.AutoEncoder,
                Method.VariationalAutoEncoder,
                Method.AutoEncoderGMM,
                Method.VariationalAutoEncoderGMM,
                Method.LLR,
            ]
        },
        Divergence: {
            Provider.ARiA: [
                Method.DpDivergence,
            ]
        },
        BER: {
            Provider.ARiA: [
                Method.MultiClassBER,
            ]
        },
    }


class DataMetric(ABC):
    """Abstract class for all DAML metrics"""

    @abstractmethod
    def evaluate(self, dataset: Iterable[float]) -> Any:
        """Implement to evaluated the dataset against the stored metric"""
        raise NotImplementedError(
            "evaluate must be called by a child class that implements it"
        )


class OutlierDetector(DataMetric, ABC):
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
        self._dataset_type: Type = None
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
        None is used for any accepted type. Can extend to check for multiple types

        Parameters
        ----------
        dataset : Iterable[float]
            An array of images in the shape (B, H, W, C)
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
        if not isinstance(dataset, np.ndarray):  # Use ndarray type conversion function
            raise TypeError("Dataset should be of type: np.ndarray")

        if not dataset.dtype.type == dtype:
            raise TypeError(
                f"Dataset values should be of type {dtype}, not {dataset.dtype.type}"
            )

    def format_dataset(
        self,
        dataset: Any,
        flatten_dataset: bool = False,
        dataset_type: Type = None,
    ) -> Iterable[float]:
        """
        Formats a dataset such that it fits the required datatype and shape

        Parameters
        ----------
        dataset : Iterable[float]
            An array of images in shape (B, H, W, C)
        flatten_dataset : bool, default False
            Flag to flatten a dataset to shape (B, H * W * C)
        dataset_type : Type, Optional

        Returns
        -------
        Iterable[float]
            Returns a dataset style array

        .. note::
            - Some metric libraries want a dataset in a particular format
                (e.g. float32, or flattened)
            - Override this to set the standard dataset formatting.
        """

        if dataset_type:
            self.check_dtype(dataset, dataset_type)
        if flatten_dataset:
            dataset = self.flatten_dataset(dataset)
        return dataset


class Divergence(DataMetric, ABC):
    """Abstract class for calculating Dp Divergence between datasets."""

    def __init__(self) -> None:
        """Constructor method"""

        super().__init__()


class BER(DataMetric, ABC):
    """Abstract class for calculating the Bayesian Error Rate."""

    def __init__(self) -> None:
        """Constructor method"""

        super().__init__()
