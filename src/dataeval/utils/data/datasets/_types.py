from __future__ import annotations

__all__ = []

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypedDict, TypeVar

from torch.utils.data import Dataset

TDatum = TypeVar("TDatum")
TArray = TypeVar("TArray")


class InfoMixin:
    _image_set: str

    def info(self) -> str:
        """Pretty prints dataset name and information."""
        return f"{self._image_set.capitalize()}\n{'-' * len(self._image_set)}\n{self.__class__.__name__}\n{str(self)}\n"


class DatasetMetadata(TypedDict):
    id: str
    index2label: dict[int, str]
    split: str


TDatasetMetadata = TypeVar("TDatasetMetadata", bound=DatasetMetadata)


class SizedDataset(Dataset[TDatum], Generic[TDatum, TDatasetMetadata]):
    metadata: TDatasetMetadata

    @abstractmethod
    def __getitem__(self, index: int) -> TDatum: ...

    @abstractmethod
    def __len__(self) -> int: ...


class ImageClassificationDataset(SizedDataset[tuple[TArray, TArray, dict[str, Any]], TDatasetMetadata]): ...


@dataclass
class ObjectDetectionTarget(Generic[TArray]):
    boxes: TArray
    labels: TArray
    scores: TArray


class ObjectDetectionDataset(
    SizedDataset[tuple[TArray, ObjectDetectionTarget[TArray], dict[str, Any]], TDatasetMetadata]
): ...
