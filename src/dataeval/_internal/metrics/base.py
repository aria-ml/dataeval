from abc import ABC, abstractmethod
from typing import Generic, TypeVar

TOutput = TypeVar("TOutput", bound=dict)


class EvaluateMixin(ABC, Generic[TOutput]):
    @abstractmethod
    def evaluate(self, *args, **kwargs) -> TOutput:
        """Abstract method to calculate metric based off of constructor parameters"""
