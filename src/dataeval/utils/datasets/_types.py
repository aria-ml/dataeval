from __future__ import annotations

__all__ = []

from dataclasses import dataclass
from typing import Generic, TypedDict, TypeVar

from torch.utils.data import Dataset
from typing_extensions import NotRequired, Required


class DatasetMetadata(TypedDict):
    id: Required[str]
    index2label: NotRequired[dict[int, str]]
    split: NotRequired[str]


class DatumMetadata(TypedDict, total=False):
    id: Required[str]


_TDatum = TypeVar("_TDatum")
_TArray = TypeVar("_TArray")


class AnnotatedDataset(Dataset[_TDatum]):
    metadata: DatasetMetadata

    def __len__(self) -> int: ...


class ImageClassificationDataset(AnnotatedDataset[tuple[_TArray, _TArray, DatumMetadata]]): ...


@dataclass
class ObjectDetectionTarget(Generic[_TArray]):
    boxes: _TArray
    labels: _TArray
    scores: _TArray


class ObjectDetectionDataset(AnnotatedDataset[tuple[_TArray, ObjectDetectionTarget[_TArray], DatumMetadata]]): ...


@dataclass
class SegmentationTarget(Generic[_TArray]):
    mask: _TArray
    labels: _TArray
    scores: _TArray


class SegmentationDataset(AnnotatedDataset[tuple[_TArray, SegmentationTarget[_TArray], DatumMetadata]]): ...
