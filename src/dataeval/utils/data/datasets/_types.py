from __future__ import annotations

__all__ = []

import sys
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypedDict, TypeVar

if sys.version_info >= (3, 11):
    from typing import NotRequired, Required
else:
    from typing_extensions import NotRequired, Required

from torch.utils.data import Dataset


class DatasetMetadata(TypedDict):
    id: Required[str]
    index2label: NotRequired[dict[int, str]]
    split: NotRequired[str]


_TDatum = TypeVar("_TDatum")
_TArray = TypeVar("_TArray")


class AnnotatedDataset(Dataset[_TDatum]):
    metadata: DatasetMetadata

    def __len__(self) -> int: ...


class ImageClassificationDataset(AnnotatedDataset[tuple[_TArray, _TArray, dict[str, Any]]]): ...


@dataclass
class ObjectDetectionTarget(Generic[_TArray]):
    boxes: _TArray
    labels: _TArray
    scores: _TArray


class ObjectDetectionDataset(AnnotatedDataset[tuple[_TArray, ObjectDetectionTarget[_TArray], dict[str, Any]]]): ...


@dataclass
class SegmentationTarget(Generic[_TArray]):
    mask: _TArray
    labels: _TArray
    scores: _TArray


class SegmentationDataset(AnnotatedDataset[tuple[_TArray, SegmentationTarget[_TArray], dict[str, Any]]]): ...


class Transform(Generic[_TArray], Protocol):
    def __call__(self, data: _TArray, /) -> _TArray: ...
