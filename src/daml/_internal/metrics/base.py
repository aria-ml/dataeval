from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable


class ThresholdType(Enum):
    VALUE = "value"
    PERCENTAGE = "percentage"


@dataclass
class Threshold:
    value: float
    type: ThresholdType


class Metric(ABC):
    """Abstract class for all DAML metrics"""

    @abstractmethod
    def evaluate(self, dataset: Iterable[float]) -> Any:
        """Implement to evaluated the dataset against the stored metric"""
        raise NotImplementedError(
            "evaluate must be called by a child class that implements it"
        )
