# Max Bright
# Based on example code from
# https://docs.seldon.io/projects/alibi-detect/en/latest/od/methods/ae.html#Examples

from abc import ABC, abstractmethod
from typing import Any, Iterable


class DataMetric(ABC):
    """Abstract class for all DAML metrics"""

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
    def initialize_detector(self) -> Any:
        """Implement to initialize an model weights and parameters"""

        raise NotImplementedError(
            "initialize_detector must be called \
            by a child class that implements it"
        )
