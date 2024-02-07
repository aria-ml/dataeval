from abc import ABC, abstractmethod
from typing import Generic, TypeVar

TOutput = TypeVar("TOutput")


class _BaseMetric(ABC, Generic[TOutput]):
    @abstractmethod
    def evaluate(self) -> TOutput:
        """Abstract method to calculate metric based off of constructor parameters"""
