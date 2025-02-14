from __future__ import annotations

__all__ = []

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from torch.utils.data import Dataset

TDatum = TypeVar("TDatum")
TArrayLike = TypeVar("TArrayLike")


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


class ImageClassificationDataset(SizedDataset[tuple[TArrayLike, TArrayLike, dict[str, Any]]]): ...


@dataclass
class ObjectDetectionTarget(Generic[TArrayLike]):
    boxes: TArrayLike
    labels: TArrayLike
    scores: TArrayLike


class ObjectDetectionDataset(SizedDataset[tuple[TArrayLike, ObjectDetectionTarget[TArrayLike], dict[str, Any]]]): ...
