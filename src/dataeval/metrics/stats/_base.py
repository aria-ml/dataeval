from __future__ import annotations

__all__ = []

import math
import re
import warnings
from collections import ChainMap
from collections.abc import Callable, Iterable, Iterator, Sequence
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Any, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from dataeval.config import get_max_processes
from dataeval.outputs._stats import BASE_ATTRS, BaseStatsOutput, SourceIndex
from dataeval.typing import Array, ArrayLike, Dataset, ObjectDetectionTarget
from dataeval.utils._array import as_numpy, to_numpy
from dataeval.utils._image import clip_and_pad, clip_box, is_valid_box, normalize_image_shape, rescale
from dataeval.utils._multiprocessing import PoolWrapper

DTYPE_REGEX = re.compile(r"NDArray\[np\.(.*?)\]")

TStatsOutput = TypeVar("TStatsOutput", bound=BaseStatsOutput, covariant=True)


@dataclass
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        # Test for invalid coordinates
        x_swap = self.x0 > self.x1
        y_swap = self.y0 > self.y1
        if x_swap or y_swap:
            warnings.warn(f"Invalid bounding box coordinates: {self} - swapping invalid coordinates.")
            if x_swap:
                self.x0, self.x1 = self.x1, self.x0
            if y_swap:
                self.y0, self.y1 = self.y1, self.y0

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    def to_int(self) -> tuple[int, int, int, int]:
        """
        Returns the bounding box as a tuple of integers.
        """
        x0_int = math.floor(self.x0)
        y0_int = math.floor(self.y0)
        x1_int = math.ceil(self.x1)
        y1_int = math.ceil(self.y1)
        return x0_int, y0_int, x1_int, y1_int


class StatsProcessor(Generic[TStatsOutput]):
    output_class: type[TStatsOutput]
    cache_keys: set[str] = set()
    image_function_map: dict[str, Callable[[StatsProcessor[TStatsOutput]], Any]] = {}
    channel_function_map: dict[str, Callable[[StatsProcessor[TStatsOutput]], Any]] = {}

    def __init__(self, image: NDArray[Any], box: BoundingBox | Iterable[Any] | None, per_channel: bool) -> None:
        self.raw = image
        self.width: int = image.shape[-1]
        self.height: int = image.shape[-2]
        box = (0, 0, self.width, self.height) if box is None else box
        self.box = box if isinstance(box, BoundingBox) else BoundingBox(*box)
        self._per_channel = per_channel
        self._image = None
        self._shape = None
        self._scaled = None
        self._cache = {}
        self._fn_map = self.channel_function_map if per_channel else self.image_function_map
        self._is_valid_box = is_valid_box(clip_box(image, self.box.to_int()))

    def get(self, fn_key: str) -> NDArray[Any]:
        if fn_key in self.cache_keys:
            if fn_key not in self._cache:
                self._cache[fn_key] = self._fn_map[fn_key](self)
            return self._cache[fn_key]
        return self._fn_map[fn_key](self)

    def process(self) -> dict[str, Any]:
        return {k: self._fn_map[k](self) for k in self._fn_map}

    @property
    def image(self) -> NDArray[Any]:
        if self._image is None:
            self._image = clip_and_pad(normalize_image_shape(self.raw), self.box.to_int())
        return self._image

    @property
    def shape(self) -> tuple[int, ...]:
        if self._shape is None:
            self._shape = self.image.shape
        return self._shape

    @property
    def scaled(self) -> NDArray[Any]:
        if self._scaled is None:
            self._scaled = rescale(self.image)
            if self._per_channel:
                self._scaled = self._scaled.reshape(self.image.shape[0], -1)
        return self._scaled

    @classmethod
    def convert_output(
        cls, source: dict[str, Any], source_index: list[SourceIndex], object_count: list[int], image_count: int
    ) -> TStatsOutput:
        output: dict[str, Any] = {}
        attrs = dict(ChainMap(*(getattr(c, "__annotations__", {}) for c in cls.output_class.__mro__)))
        for key in (key for key in source if key in attrs):
            stat_type: str = attrs[key]
            dtype_match = re.match(DTYPE_REGEX, stat_type)
            if dtype_match is not None:
                output[key] = np.asarray(source[key], dtype=np.dtype(dtype_match.group(1)))
            else:
                output[key] = source[key]
        base_attrs: dict[str, Any] = dict(
            zip(BASE_ATTRS, (source_index, np.asarray(object_count, dtype=np.uint16), image_count))
        )
        return cls.output_class(**output, **base_attrs)


@dataclass
class StatsProcessorOutput:
    results: list[dict[str, Any]]
    source_indices: list[SourceIndex]
    object_counts: list[int]
    warnings_list: list[str]


def process_stats(
    i: int,
    image: ArrayLike,
    boxes: list[BoundingBox] | None,
    per_channel: bool,
    stats_processor_cls: Iterable[type[StatsProcessor[TStatsOutput]]],
) -> StatsProcessorOutput:
    np_image = to_numpy(image)
    results_list: list[dict[str, Any]] = []
    source_indices: list[SourceIndex] = []
    box_counts: list[int] = []
    warnings_list: list[str] = []
    for i_b, box in [(None, None)] if boxes is None else enumerate(boxes):
        processor_list = [p(np_image, box, per_channel) for p in stats_processor_cls]
        if any(not p._is_valid_box for p in processor_list) and i_b is not None and box is not None:
            warnings_list.append(f"Bounding box [{i}][{i_b}]: {box} for image shape {np_image.shape} is invalid.")
        results_list.append({k: v for p in processor_list for k, v in p.process().items()})
        if per_channel:
            source_indices.extend([SourceIndex(i, i_b, c) for c in range(np_image.shape[-3])])
        else:
            source_indices.append(SourceIndex(i, i_b, None))
    box_counts.append(0 if boxes is None else len(boxes))
    return StatsProcessorOutput(results_list, source_indices, box_counts, warnings_list)


def process_stats_unpack(
    args: tuple[int, ArrayLike, list[BoundingBox] | None],
    per_channel: bool,
    stats_processor_cls: Iterable[type[StatsProcessor[TStatsOutput]]],
) -> StatsProcessorOutput:
    return process_stats(*args, per_channel=per_channel, stats_processor_cls=stats_processor_cls)


def _enumerate(
    dataset: Dataset[ArrayLike] | Dataset[tuple[ArrayLike, Any, Any]], per_box: bool
) -> Iterator[tuple[int, ArrayLike, Any]]:
    for i in range(len(dataset)):
        d = dataset[i]
        image = d[0] if isinstance(d, tuple) else d
        if per_box and isinstance(d, tuple) and isinstance(d[1], ObjectDetectionTarget):
            try:
                boxes = d[1].boxes if isinstance(d[1].boxes, Array) else as_numpy(d[1].boxes)
                target = [BoundingBox(*(float(box[i]) for i in range(4))) for box in boxes]
            except (ValueError, IndexError):
                raise ValueError(f"Invalid bounding box format for image {i}: {d[1].boxes}")
        else:
            target = None

        yield i, image, target


def run_stats(
    dataset: Dataset[ArrayLike] | Dataset[tuple[ArrayLike, Any, Any]],
    per_box: bool,
    per_channel: bool,
    stats_processor_cls: Iterable[type[StatsProcessor[TStatsOutput]]],
) -> list[TStatsOutput]:
    """
    Compute specified :term:`statistics<Statistics>` on a set of images.

    This function applies a set of statistical operations to each image in the input iterable,
    based on the specified output class. The function determines which statistics to apply
    using a function map. It also supports optional image flattening for pixel-wise calculations.

    Parameters
    ----------
    data : Dataset[ArrayLike] | Dataset[tuple[ArrayLike, Any, Any]]
        A dataset of images and targets to compute statistics on.
    per_box : bool
        A flag which determines if the statistics should be evaluated on a per-box basis or not.
        If the dataset does not include bounding boxes, this flag is ignored.
    per_channel : bool
        A flag which determines if the states should be evaluated on a per-channel basis or not.
    stats_processor_cls : Iterable[type[StatsProcessor]]
        An iterable of stats processor classes that calculate stats and return output classes.

    Returns
    -------
    list[TStatsOutput]
        A list of output classes containing the computed statistics

    Note
    ----
    - The function performs image normalization (rescaling the image values)
      before applying some of the statistics.
    - Pixel-level statistics (e.g., :term:`brightness<Brightness>`, entropy) are computed after
      rescaling and, optionally, flattening the images.
    - For statistics like histograms and entropy, intermediate results may
      be reused to avoid redundant computation.
    """
    results_list: list[dict[str, NDArray[np.float64]]] = []
    source_index: list[SourceIndex] = []
    object_count: list[int] = []
    image_count: int = len(dataset)

    warning_list = []
    stats_processor_cls = stats_processor_cls if isinstance(stats_processor_cls, Iterable) else [stats_processor_cls]

    with PoolWrapper(processes=get_max_processes()) as p:
        for r in tqdm(
            p.imap(
                partial(
                    process_stats_unpack,
                    per_channel=per_channel,
                    stats_processor_cls=stats_processor_cls,
                ),
                _enumerate(dataset, per_box),
            ),
            total=image_count,
        ):
            results_list.extend(r.results)
            source_index.extend(r.source_indices)
            object_count.extend(r.object_counts)
            warning_list.extend(r.warnings_list)

    # warnings are not emitted while in multiprocessing pools so we emit after gathering all warnings
    for w in warning_list:
        warnings.warn(w, UserWarning)

    output = {}
    for results in results_list:
        for stat, result in results.items():
            if per_channel:
                output.setdefault(stat, []).extend(result.tolist())
            else:
                output.setdefault(stat, []).append(result.tolist() if isinstance(result, np.ndarray) else result)

    return [s.convert_output(output, source_index, object_count, image_count) for s in stats_processor_cls]


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
