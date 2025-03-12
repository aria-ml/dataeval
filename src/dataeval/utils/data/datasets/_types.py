from __future__ import annotations

import sys
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypedDict, TypeVar

if sys.version_info >= (3, 11):
    from typing import NotRequired, Required
else:
    from typing_extensions import NotRequired, Required

from torch.utils.data import Dataset

_TArray = TypeVar("_TArray")
_TTarget = TypeVar("_TTarget")


class SizedDataset(Dataset[tuple[_TArray, _TTarget, dict[str, Any]]]):
    @abstractmethod
    def __getitem__(self, index: Any) -> tuple[_TArray, _TTarget, dict[str, Any]]: ...

    @abstractmethod
    def __len__(self) -> int: ...


class DatasetMetadata(TypedDict):
    id: Required[str]
    index2label: Required[dict[int, str]]
    split: NotRequired[str]


class AnnotatedDataset(SizedDataset[_TArray, _TTarget]):
    metadata: DatasetMetadata


class ImageClassificationDataset(AnnotatedDataset[_TArray, _TArray]): ...


@dataclass
class ObjectDetectionTarget(Generic[_TArray]):
    boxes: _TArray
    labels: _TArray
    scores: _TArray


class ObjectDetectionDataset(AnnotatedDataset[_TArray, ObjectDetectionTarget[_TArray]]): ...


@dataclass
class SegmentationTarget(Generic[_TArray]):
    mask: _TArray
    labels: _TArray
    scores: _TArray


class SegmentationDataset(AnnotatedDataset[_TArray, SegmentationTarget[_TArray]]): ...


class Transform(Protocol, Generic[_TArray]):
    def __call__(self, data: _TArray, /) -> _TArray: ...
