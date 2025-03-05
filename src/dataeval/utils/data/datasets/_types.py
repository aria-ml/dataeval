from __future__ import annotations

__all__ = []

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from torch.utils.data import Dataset

TDatum = TypeVar("TDatum")
TArray = TypeVar("TArray")


class InfoMixin:
    _image_set: str

    def info(self) -> str:
        """Pretty prints dataset name and information."""

        return f"{self._image_set.capitalize()}\n{'-' * len(self._image_set)}\n{str(self)}\n"


class SizedDataset(Dataset[TDatum], ABC):
    @abstractmethod
    def __getitem__(self, index: int) -> TDatum: ...

    @abstractmethod
    def __len__(self) -> int: ...


class ImageClassificationDataset(SizedDataset[tuple[TArray, TArray, dict[str, Any]]]): ...


@dataclass
class ObjectDetectionTarget(Generic[TArray]):
    boxes: TArray
    labels: TArray
    scores: TArray


class ObjectDetectionDataset(SizedDataset[tuple[TArray, ObjectDetectionTarget[TArray], dict[str, Any]]]): ...
