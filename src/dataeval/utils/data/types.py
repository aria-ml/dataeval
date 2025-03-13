from __future__ import annotations

__all__ = [
    "Dataset",
    "DatasetMetadata",
    "ImageClassificationDataset",
    "ObjectDetectionDataset",
    "ObjectDetectionTarget",
    "SegmentationDataset",
    "SegmentationTarget",
    "Transform",
]

import sys
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypedDict, TypeVar

if sys.version_info >= (3, 11):
    from typing import NotRequired, Required
else:
    from typing_extensions import NotRequired, Required

from torch.utils.data import Dataset as _Dataset

_TArray = TypeVar("_TArray")
_TData = TypeVar("_TData", covariant=True)
_TTarget = TypeVar("_TTarget", covariant=True)


class DatasetMetadata(TypedDict):
    id: Required[str]
    index2label: NotRequired[dict[int, str]]
    split: NotRequired[str]


class Dataset(_Dataset[tuple[_TData, _TTarget, dict[str, Any]]]):
    metadata: DatasetMetadata

    def __getitem__(self, index: Any) -> tuple[_TData, _TTarget, dict[str, Any]]: ...
    def __len__(self) -> int: ...


class ImageClassificationDataset(Dataset[_TArray, _TArray]): ...


@dataclass
class ObjectDetectionTarget(Generic[_TArray]):
    boxes: _TArray
    labels: _TArray
    scores: _TArray


class ObjectDetectionDataset(Dataset[_TArray, ObjectDetectionTarget[_TArray]]): ...


@dataclass
class SegmentationTarget(Generic[_TArray]):
    mask: _TArray
    labels: _TArray
    scores: _TArray


class SegmentationDataset(Dataset[_TArray, SegmentationTarget[_TArray]]): ...


class Transform(Generic[_TArray], Protocol):
    def __call__(self, data: _TArray, /) -> _TArray: ...
