from __future__ import annotations

__all__ = []

import re
import warnings
from collections import ChainMap
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from typing import Any, Callable, Generic, Iterable, Sequence, TypeVar

import numpy as np
import tqdm
from numpy.typing import NDArray

from dataeval.config import get_max_processes
from dataeval.outputs._stats import BaseStatsOutput, SourceIndex
from dataeval.typing import Array, ArrayLike, Dataset, ObjectDetectionTarget
from dataeval.utils._array import to_numpy
from dataeval.utils._image import normalize_image_shape, rescale

DTYPE_REGEX = re.compile(r"NDArray\[np\.(.*?)\]")


def normalize_box_shape(bounding_box: NDArray[Any]) -> NDArray[Any]:
    """
    Normalizes the bounding box shape into (N,4).
    """
    ndim = bounding_box.ndim
    if ndim == 1:
        return np.expand_dims(bounding_box, axis=0)
    elif ndim > 2:
        raise ValueError("Bounding boxes must have 2 dimensions: (# of boxes in an image, [X,Y,W,H]) -> (N,4)")
    else:
        return bounding_box


TStatsOutput = TypeVar("TStatsOutput", bound=BaseStatsOutput, covariant=True)


class StatsProcessor(Generic[TStatsOutput]):
    output_class: type[TStatsOutput]
    cache_keys: list[str] = []
    image_function_map: dict[str, Callable[[StatsProcessor[TStatsOutput]], Any]] = {}
    channel_function_map: dict[str, Callable[[StatsProcessor[TStatsOutput]], Any]] = {}

    def __init__(self, image: NDArray[Any], box: NDArray[Any] | None, per_channel: bool) -> None:
        self.raw = image
        self.width: int = image.shape[-1]
        self.height: int = image.shape[-2]
        self.box: NDArray[np.int64] = np.array([0, 0, self.width, self.height]) if box is None else box.astype(np.int64)
        self._per_channel = per_channel
        self._image = None
        self._shape = None
        self._scaled = None
        self._cache = {}
        self._fn_map = self.channel_function_map if per_channel else self.image_function_map
        self._is_valid_slice = box is None or bool(
            box[0] >= 0 and box[1] >= 0 and box[2] <= image.shape[-1] and box[3] <= image.shape[-2]
        )

    def get(self, fn_key: str) -> NDArray[Any]:
        if fn_key in self.cache_keys:
            if fn_key not in self._cache:
                self._cache[fn_key] = self._fn_map[fn_key](self)
            return self._cache[fn_key]
        else:
            return self._fn_map[fn_key](self)

    def process(self) -> dict[str, Any]:
        return {k: self._fn_map[k](self) for k in self._fn_map}

    @property
    def image(self) -> NDArray[Any]:
        if self._image is None:
            if self._is_valid_slice:
                norm = normalize_image_shape(self.raw)
                self._image = norm[:, self.box[1] : self.box[3], self.box[0] : self.box[2]]
            else:
                self._image = np.zeros((self.raw.shape[0], self.box[3] - self.box[1], self.box[2] - self.box[0]))
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
        cls, source: dict[str, Any], source_index: list[SourceIndex], box_count: list[int]
    ) -> TStatsOutput:
        output = {}
        attrs = dict(ChainMap(*(getattr(c, "__annotations__", {}) for c in cls.output_class.__mro__)))
        for key in (key for key in source if key in attrs):
            stat_type: str = attrs[key]
            dtype_match = re.match(DTYPE_REGEX, stat_type)
            if dtype_match is not None:
                output[key] = np.asarray(source[key], dtype=np.dtype(dtype_match.group(1)))
            else:
                output[key] = source[key]
        return cls.output_class(**output, source_index=source_index, box_count=np.asarray(box_count, dtype=np.uint16))


@dataclass
class StatsProcessorOutput:
    results: list[dict[str, Any]]
    source_indices: list[SourceIndex]
    box_counts: list[int]
    warnings_list: list[str]


def process_stats(
    i: int,
    image: ArrayLike,
    target: Any,
    per_box: bool,
    per_channel: bool,
    stats_processor_cls: Iterable[type[StatsProcessor[TStatsOutput]]],
) -> StatsProcessorOutput:
    image = to_numpy(image)
    boxes = to_numpy(target.boxes) if isinstance(target, ObjectDetectionTarget) else None
    results_list: list[dict[str, Any]] = []
    source_indices: list[SourceIndex] = []
    box_counts: list[int] = []
    warnings_list: list[str] = []
    for i_b, box in [(None, None)] if boxes is None else enumerate(normalize_box_shape(boxes)):
        processor_list = [p(image, box, per_channel) for p in stats_processor_cls]
        if any(not p._is_valid_slice for p in processor_list) and i_b is not None and box is not None:
            warnings_list.append(f"Bounding box [{i}][{i_b}]: {box} is out of bounds of {image.shape}.")
        results_list.append({k: v for p in processor_list for k, v in p.process().items()})
        if per_channel:
            source_indices.extend([SourceIndex(i, i_b, c) for c in range(image.shape[-3])])
        else:
            source_indices.append(SourceIndex(i, i_b, None))
    box_counts.append(0 if boxes is None else len(boxes))
    return StatsProcessorOutput(results_list, source_indices, box_counts, warnings_list)


def process_stats_unpack(
    args: tuple[int, ArrayLike, Any],
    per_box: bool,
    per_channel: bool,
    stats_processor_cls: Iterable[type[StatsProcessor[TStatsOutput]]],
) -> StatsProcessorOutput:
    return process_stats(*args, per_box=per_box, per_channel=per_channel, stats_processor_cls=stats_processor_cls)


def run_stats(
    dataset: Dataset[Array] | Dataset[tuple[Array, Any, Any]],
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
    data : Dataset[Array] | Dataset[tuple[Array, Any, Any]]
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
    box_count: list[int] = []

    warning_list = []
    stats_processor_cls = stats_processor_cls if isinstance(stats_processor_cls, Iterable) else [stats_processor_cls]

    def _enumerate(dataset: Dataset[Array] | Dataset[tuple[Array, Any, Any]], per_box: bool):
        for i in range(len(dataset)):
            d = dataset[i]
            yield i, d[0] if isinstance(d, tuple) else d, d[1] if isinstance(d, tuple) and per_box else None

    with Pool(processes=get_max_processes()) as p:
        for r in tqdm.tqdm(
            p.imap(
                partial(
                    process_stats_unpack,
                    per_box=per_box,
                    per_channel=per_channel,
                    stats_processor_cls=stats_processor_cls,
                ),
                _enumerate(dataset, per_box),
            ),
            total=len(dataset),
        ):
            results_list.extend(r.results)
            source_index.extend(r.source_indices)
            box_count.extend(r.box_counts)
            warning_list.extend(r.warnings_list)
    p.close()
    p.join()

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

    outputs = [s.convert_output(output, source_index, box_count) for s in stats_processor_cls]
    return outputs


def add_stats(a: TStatsOutput, b: TStatsOutput) -> TStatsOutput:
    if type(a) is not type(b):
        raise TypeError(f"Types {type(a)} and {type(b)} cannot be added.")

    sum_dict = deepcopy(a.data())

    for k in sum_dict:
        if isinstance(sum_dict[k], list):
            sum_dict[k].extend(b.data()[k])
        else:
            sum_dict[k] = np.concatenate((sum_dict[k], b.data()[k]))

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
