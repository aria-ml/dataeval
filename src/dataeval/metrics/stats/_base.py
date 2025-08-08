from __future__ import annotations

__all__ = []

import re
from collections import ChainMap
from collections.abc import Iterator, Sequence
from copy import deepcopy
from itertools import tee
from typing import Any, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

from dataeval.outputs._stats import BaseStatsOutput
from dataeval.typing import Array, ArrayLike, Dataset, ObjectDetectionTarget
from dataeval.utils._array import as_numpy
from dataeval.utils._boundingbox import BoundingBox

DTYPE_REGEX = re.compile(r"NDArray\[np\.(.*?)\]")

TStatsOutput = TypeVar("TStatsOutput", bound=BaseStatsOutput, covariant=True)
T = TypeVar("T")


class SizedIterator(Generic[T]):
    def __init__(self, iterator: Iterator[T], length: int) -> None:
        self._iterator = iterator
        self._length = length

    def __iter__(self) -> Iterator[T]:
        return self._iterator

    def __next__(self) -> T:
        return next(self._iterator)

    def __len__(self) -> int:
        return self._length


def convert_output(
    output_cls: type[TStatsOutput],
    output_dict: dict[str, Any],
) -> TStatsOutput:
    output: dict[str, Any] = {}
    attrs = dict(ChainMap(*(getattr(c, "__annotations__", {}) for c in output_cls.__mro__)))
    for key in (key for key in output_dict if key in attrs):
        stat_type: str = attrs[key]
        dtype_match = re.match(DTYPE_REGEX, stat_type)
        if dtype_match is not None:
            output[key] = np.asarray(output_dict[key], dtype=np.dtype(dtype_match.group(1)))
        else:
            output[key] = output_dict[key]
    return output_cls(**output)  # , **base_attrs)


def unzip_dataset(
    dataset: Dataset[ArrayLike] | Dataset[tuple[ArrayLike, Any, Any]], per_box: bool
) -> tuple[Iterator[NDArray[Any]], Iterator[list[BoundingBox] | None] | None]:
    def _generate_pairs() -> Iterator[tuple[NDArray[Any], list[BoundingBox] | None]]:
        for i in range(len(dataset)):
            d = dataset[i]
            image = np.asarray(d[0] if isinstance(d, tuple) else d)
            if per_box and isinstance(d, tuple) and isinstance(d[1], ObjectDetectionTarget):
                try:
                    boxes = d[1].boxes if isinstance(d[1].boxes, Array) else as_numpy(d[1].boxes)
                    target = [BoundingBox(box[0], box[1], box[2], box[3], image_shape=image.shape) for box in boxes]
                except (ValueError, IndexError):
                    raise ValueError(f"Invalid bounding box format for image {i}: {d[1].boxes}")
            else:
                target = None
            yield image, target

    # Create two independent iterators from the generator
    iter1, iter2 = tee(_generate_pairs(), 2)

    # Extract images and targets separately
    images_iter = SizedIterator((pair[0] for pair in iter1), len(dataset))
    targets_iter = (pair[1] for pair in iter2) if per_box else None

    return images_iter, targets_iter


def add_stats(a: TStatsOutput, b: TStatsOutput) -> TStatsOutput:
    if type(a) is not type(b):
        raise TypeError(f"Types {type(a)} and {type(b)} cannot be added.")

    sum_dict = deepcopy(a.data())

    for k in sum_dict:
        if isinstance(sum_dict[k], Sequence):
            sum_dict[k].extend(b.data()[k])
        elif isinstance(sum_dict[k], Array):
            sum_dict[k] = np.concatenate((sum_dict[k], b.data()[k]))
        else:
            sum_dict[k] += b.data()[k]

    return type(a)(**sum_dict)


def combine_stats(stats: Sequence[TStatsOutput]) -> tuple[TStatsOutput, list[int]]:
    output = None
    dataset_steps = []
    cur_len = 0
    for s in stats:
        output = s if output is None else add_stats(output, s)
        cur_len += len(s)
        dataset_steps.append(cur_len)
    if output is None:
        raise TypeError("Cannot combine empty sequence of stats.")
    return output, dataset_steps


def get_dataset_step_from_idx(idx: int, dataset_steps: list[int]) -> tuple[int, int]:
    last_step = 0
    for i, step in enumerate(dataset_steps):
        if idx < step:
            return i, idx - last_step
        last_step = step
    return -1, idx
