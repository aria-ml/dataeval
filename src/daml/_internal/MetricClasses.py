# Max Bright
# Based on example code from
# https://docs.seldon.io/projects/alibi-detect/en/latest/od/methods/ae.html#Examples

from abc import ABC, abstractmethod
from typing import Any, Iterable, Type

import numpy as np


class Metrics:
    """A global dictionary to parse metrics, providers, and methods"""

    OutlierDetection = "OutlierDetection"
    Divergence = "Divergence"
    BER = "BER"

    class Provider:
        AlibiDetect = "Alibi-Detect"
        ARiA = "ARiA"

    class Method:
        AutoEncoder = "Autoencoder"
        VariationalAutoEncoder = "VAE"
        AutoEncoderGMM = "AEGMM"
        VariationalAutoEncoderGMM = "VAEGMM"
        LLR = "LLR"
        DpDivergence = "Dp_Divergence"
        MultiClassBER = "MultiClassBER"

    class Algorithm:
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
    """Abstract class for all DAML outlier detection metrics."""

    def __init__(self):
        self.is_trained: bool = False
        self.DATASET_TYPE: Type = None
        self.FLATTEN_DATASET: bool = False

    @abstractmethod
    def fit_dataset(
        self,
        dataset: Iterable[float],
        epochs: int,
        verbose: bool,
    ) -> None:
        """Implement to fit a metric to the given dataset"""
        raise NotImplementedError(
            "fit_dataset must be called by a child class that implements it"
        )

    @abstractmethod
    def initialize_detector(self) -> Any:
        """Implement to initialize an model weights and parameters"""

        raise NotImplementedError(
            "initialize_detector must be called \
            by a child class that implements it"
        )

    def flatten_dataset(self, dataset) -> np.ndarray:
        """Flattens an array of (H, W, C) images into an array of (H*W*C)"""
        return np.reshape(dataset, (len(dataset), np.prod(np.shape(dataset[0]))))

    def check_dtype(self, dataset, dtype):
        """
        Check is the dataset dtype fits with the required dtype of the model.
        None is used for any accepted type. Can extend to check for multiple types
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
    ) -> Any:
        """
        Some metric libraries want a dataset in a particular format
        (e.g. float32, or flattened)
        This returns an iterable dataset object.
        Override this to set the standard dataset formatting.
        """
        if dataset_type:
            self.check_dtype(dataset, dataset_type)
        if flatten_dataset:
            dataset = self.flatten_dataset(dataset)
        return dataset


class Divergence(DataMetric, ABC):
    """Abstract class for calculating Dp Divergence between datasets."""

    def __init__(self) -> None:
        super().__init__()


class BER(DataMetric, ABC):
    """Abstract class for calculating the Bayesian Error Rate."""

    def __init__(self) -> None:
        super().__init__()
