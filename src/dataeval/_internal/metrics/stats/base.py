from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from typing import Any, Callable, Generic, Iterable, NamedTuple, Optional, TypeVar, Union

import numpy as np
import tqdm
from numpy.typing import ArrayLike, NDArray

from dataeval._internal.interop import to_numpy_iter
from dataeval._internal.metrics.utils import normalize_box_shape, normalize_image_shape, rescale
from dataeval._internal.output import OutputMetadata

DTYPE_REGEX = re.compile(r"NDArray\[np\.(.*?)\]")
SOURCE_INDEX = "source_index"
BOX_COUNT = "box_count"

OptionalRange = Optional[Union[int, Iterable[int]]]


def matches(index: int | None, opt_range: OptionalRange) -> bool:
    if index is None or opt_range is None:
        return True
    return index in opt_range if isinstance(opt_range, Iterable) else index == opt_range


class SourceIndex(NamedTuple):
    """
    Attributes
    ----------
    image: int
        Index of the source image
    box : int | None
        Index of the box of the source image
    channel : int | None
        Index of the channel of the source image
    """

    image: int
    box: int | None
    channel: int | None


@dataclass(frozen=True)
class BaseStatsOutput(OutputMetadata):
    """
    Attributes
    ----------
    source_index : List[SourceIndex]
        Mapping from statistic to source image, box and channel index
    box_count : NDArray[np.uint16]
    """

    source_index: list[SourceIndex]
    box_count: NDArray[np.uint16]

    def get_channel_mask(
        self,
        channel_index: OptionalRange,
        channel_count: OptionalRange = None,
    ) -> list[bool]:
        """
        Boolean mask for results filtered to specified channel index and optionally the count
        of the channels per image.

        Parameters
        ----------
        channel_index : int | Iterable[int] | None
            Index or indices of channel(s) to filter for
        channel_count : int | Iterable[int] | None
            Optional count(s) of channels to filter for
        """
        mask: list[bool] = []
        cur_mask: list[bool] = []
        cur_image = 0
        cur_max_channel = 0
        for source_index in list(self.source_index) + [None]:
            if source_index is None or source_index.image > cur_image:
                mask.extend(cur_mask if matches(cur_max_channel + 1, channel_count) else [False for _ in cur_mask])
                if source_index is None:
                    break
                cur_image = source_index.image
                cur_max_channel = 0
                cur_mask.clear()
            cur_mask.append(matches(source_index.channel, channel_index))
            cur_max_channel = max(cur_max_channel, source_index.channel or 0)
        return mask

    def __len__(self) -> int:
        return len(self.source_index)


TStatsOutput = TypeVar("TStatsOutput", bound=BaseStatsOutput, covariant=True)


class StatsProcessor(Generic[TStatsOutput]):
    output_class: type[TStatsOutput]
    cache_keys: list[str] = []
    image_function_map: dict[str, Callable[[StatsProcessor], Any]] = {}
    channel_function_map: dict[str, Callable[[StatsProcessor], Any]] = {}

    def __init__(self, image: NDArray, box: NDArray | None, per_channel: bool):
        self.raw = image
        self.width = image.shape[-1]
        self.height = image.shape[-2]
        self.box = np.array([0, 0, self.width, self.height]) if box is None else box
        self.per_channel = per_channel
        self._image = None
        self._shape = None
        self._scaled = None
        self.cache = {}
        self.fn_map = self.channel_function_map if per_channel else self.image_function_map
        self.is_valid_slice = box is None or bool(
            box[0] >= 0 and box[1] >= 0 and box[2] <= image.shape[-1] and box[3] <= image.shape[-2]
        )

    def get(self, fn_key: str) -> NDArray:
        if fn_key in self.cache_keys:
            if fn_key not in self.cache:
                self.cache[fn_key] = self.fn_map[fn_key](self)
            return self.cache[fn_key]
        else:
            return self.fn_map[fn_key](self)

    def process(self) -> dict:
        return {k: self.fn_map[k](self) for k in self.fn_map}

    @property
    def image(self) -> NDArray:
        if self._image is None:
            if self.is_valid_slice:
                norm = normalize_image_shape(self.raw)
                self._image = norm[:, self.box[1] : self.box[3], self.box[0] : self.box[2]]
            else:
                self._image = np.zeros((self.raw.shape[0], self.box[3] - self.box[1], self.box[2] - self.box[0]))
        return self._image

    @property
    def shape(self) -> tuple:
        if self._shape is None:
            self._shape = self.image.shape
        return self._shape

    @property
    def scaled(self) -> NDArray:
        if self._scaled is None:
            self._scaled = rescale(self.image)
            if self.per_channel:
                self._scaled = self._scaled.reshape(self.image.shape[0], -1)
        return self._scaled

    @classmethod
    def convert_output(
        cls, source: dict[str, Any], source_index: list[SourceIndex], box_count: list[int]
    ) -> TStatsOutput:
        output = {}
        for key in source:
            if key not in cls.output_class.__annotations__:
                continue
            stat_type: str = cls.output_class.__annotations__[key]
            dtype_match = re.match(DTYPE_REGEX, stat_type)
            if dtype_match is not None:
                output[key] = np.asarray(source[key], dtype=np.dtype(dtype_match.group(1)))
            else:
                output[key] = source[key]
        return cls.output_class(**output, source_index=source_index, box_count=np.asarray(box_count, dtype=np.uint16))


class StatsProcessorOutput(NamedTuple):
    results: list[dict[str, Any]]
    source_indices: list[SourceIndex]
    box_counts: list[int]
    warnings_list: list[tuple[int, int, NDArray, tuple[int, ...]]]


def process_stats(
    i: int,
    image_boxes: tuple[NDArray, NDArray | None],
    per_channel: bool,
    stats_processor_cls: Iterable[type[StatsProcessor]],
) -> StatsProcessorOutput:
    image, boxes = image_boxes
    results_list: list[dict[str, Any]] = []
    source_indices: list[SourceIndex] = []
    box_counts: list[int] = []
    warnings_list: list[tuple[int, int, NDArray, tuple[int, ...]]] = []
    nboxes = [None] if boxes is None else normalize_box_shape(boxes)
    for i_b, box in enumerate(nboxes):
        i_b = None if box is None else i_b
        processor_list = [p(image, box, per_channel) for p in stats_processor_cls]
        if any(not p.is_valid_slice for p in processor_list) and i_b is not None and box is not None:
            warnings_list.append((i, i_b, box, image.shape))
        results_list.append({k: v for p in processor_list for k, v in p.process().items()})
        if per_channel:
            source_indices.extend([SourceIndex(i, i_b, c) for c in range(image_boxes[0].shape[-3])])
        else:
            source_indices.append(SourceIndex(i, i_b, None))
    box_counts.append(0 if boxes is None else len(boxes))
    return StatsProcessorOutput(results_list, source_indices, box_counts, warnings_list)


def process_stats_unpack(args, per_channel: bool, stats_processor_cls: Iterable[type[StatsProcessor]]):
    return process_stats(*args, per_channel=per_channel, stats_processor_cls=stats_processor_cls)


def run_stats(
    images: Iterable[ArrayLike],
    bboxes: Iterable[ArrayLike] | None,
    per_channel: bool,
    stats_processor_cls: Iterable[type[StatsProcessor[TStatsOutput]]],
) -> list[TStatsOutput]:
    """
    Compute specified statistics on a set of images.

    This function applies a set of statistical operations to each image in the input iterable,
    based on the specified output class. The function determines which statistics to apply
    using a function map. It also supports optional image flattening for pixel-wise calculations.

    Parameters
    ----------
    images : Iterable[ArrayLike]
        An iterable of images (e.g., list of arrays), where each image is represented as an
        array-like structure (e.g., NumPy arrays).
    bboxes : Iterable[ArrayLike]
        An iterable of bounding boxes (e.g. list of arrays) where each bounding box is represented
        as an array-like structure in the format of (X0, Y0, X1, Y1). The length of the bounding boxes
        iterable should match the length of the input images.
    per_channel : bool
        A flag which determines if the states should be evaluated on a per-channel basis or not.
    stats_processor_cls : Iterable[type[StatsProcessor]]
        An iterable of stats processor classes that calculate stats and return output classes.

    Returns
    -------
    list[TStatsOutput]
        A list of output classes corresponding to the input processor types.

    Note
    ----
    - The function performs image normalization (rescaling the image values)
      before applying some of the statistics.
    - Pixel-level statistics (e.g., brightness, entropy) are computed after
      rescaling and, optionally, flattening the images.
    - For statistics like histograms and entropy, intermediate results may
      be reused to avoid redundant computation.
    """
    results_list: list[dict[str, NDArray]] = []
    source_index = []
    box_count = []
    bbox_iter = repeat(None) if bboxes is None else to_numpy_iter(bboxes)

    warning_list = []
    total_for_status = getattr(images, "__len__")() if hasattr(images, "__len__") else None
    stats_processor_cls = stats_processor_cls if isinstance(stats_processor_cls, Iterable) else [stats_processor_cls]

    # TODO: Introduce global controls for CPU job parallelism and GPU configurations
    with Pool(16) as p:
        for r in tqdm.tqdm(
            p.imap(
                partial(process_stats_unpack, per_channel=per_channel, stats_processor_cls=stats_processor_cls),
                enumerate(zip(to_numpy_iter(images), bbox_iter)),
            ),
            total=total_for_status,
        ):
            results_list.extend(r.results)
            source_index.extend(r.source_indices)
            box_count.extend(r.box_counts)
            warning_list.extend(r.warnings_list)
    p.close()
    p.join()

    # warnings are not emitted while in multiprocessing pools so we emit after gathering all warnings
    for w in warning_list:
        warnings.warn(f"Bounding box [{w[0]}][{w[1]}]: {w[2]} is out of bounds of {w[3]}.", UserWarning)

    output = {}
    for results in results_list:
        for stat, result in results.items():
            if per_channel:
                output.setdefault(stat, []).extend(result.tolist())
            else:
                output.setdefault(stat, []).append(result.tolist() if isinstance(result, np.ndarray) else result)

    outputs = [s.convert_output(output, source_index, box_count) for s in stats_processor_cls]
    return outputs
