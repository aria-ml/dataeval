from __future__ import annotations

__all__ = []

from abc import ABC, abstractmethod
from typing import Any, Sequence, TypedDict, TypeVar, Union

from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset

TDatum = TypeVar("TDatum")
TArrayLike = TypeVar("TArrayLike", Sequence[Any], NDArray[Any], Tensor)
ArrayLike = Union[Sequence[Any], NDArray[Any], Tensor]


class SizedDataset(Dataset[TDatum], ABC):
    @abstractmethod
    def __getitem__(self, index: int) -> TDatum: ...

    @abstractmethod
    def __len__(self) -> int: ...


class InfoMixin:
    _image_set: str

    def info(self) -> str:
        """Pretty prints dataset name and info"""

        return f"{self._image_set.capitalize()}\n{'-' * len(self._image_set)}\n{str(self)}\n"


class ImageClassificationDataset(SizedDataset[tuple[TArrayLike, TArrayLike, dict[str, Any]]]): ...


class ObjectDetectionTarget(TypedDict):
    boxes: ArrayLike
    labels: ArrayLike
    scores: ArrayLike


class ObjectDetectionDataset(SizedDataset[tuple[TArrayLike, ObjectDetectionTarget, dict[str, Any]]]): ...
