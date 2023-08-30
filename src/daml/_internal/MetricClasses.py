# Max Bright
# Based on example code from
# https://docs.seldon.io/projects/alibi-detect/en/latest/od/methods/ae.html#Examples

from abc import ABC, abstractmethod
from typing import Any, Iterable


class Metrics:
    """A global dictionary to parse metrics, providers, and methods"""

    OutlierDetection = "OutlierDetection"
    Divergence = "Divergence"

    class Provider:
        AlibiDetect = "Alibi-Detect"
        ARiA = "ARiA"

    class Method:
        AutoEncoder = "Autoencoder"
        VariationalAutoEncoder = "VAE"
        DpDivergence = "Dp_Divergence"

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
            ]
        },
        Divergence: {
            Provider.ARiA: [
                Method.DpDivergence,
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


class Divergence(DataMetric, ABC):
    """Abstract class for calculating Dp Divergence between datasets."""

    def __init__(self) -> None:
        super().__init__()
