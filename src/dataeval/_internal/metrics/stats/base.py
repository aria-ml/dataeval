from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Iterable, NamedTuple, Optional, Union

import numpy as np
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


class StatsProcessor:
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


def run_stats(
    images: Iterable[ArrayLike],
    bboxes: Iterable[ArrayLike] | None,
    per_channel: bool,
    stats_processor_cls: type,
    output_cls: type,
) -> dict:
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
    output_cls : type
        The output class for which stats values will be calculated.

    Returns
    -------
    dict[str, NDArray]]
        A dictionary containing the computed statistics for each image.
        The dictionary keys correspond to the names of the statistics, and the values are NumPy arrays
        with the results of the computations.

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
    output_list = list(output_cls.__annotations__)
    source_index = []
    box_count = []
    bbox_iter = (None for _ in images) if bboxes is None else to_numpy_iter(bboxes)

    for i, (boxes, image) in enumerate(zip(bbox_iter, to_numpy_iter(images))):
        nboxes = [None] if boxes is None else normalize_box_shape(boxes)
        for i_b, box in enumerate(nboxes):
            i_b = None if box is None else i_b
            processor: StatsProcessor = stats_processor_cls(image, box, per_channel)
            if not processor.is_valid_slice:
                warnings.warn(f"Bounding box {i_b}: {box} is out of bounds of image {i}: {image.shape}.")
            results_list.append({stat: processor.get(stat) for stat in output_list})
            if per_channel:
                source_index.extend([SourceIndex(i, i_b, c) for c in range(image.shape[-3])])
            else:
                source_index.append(SourceIndex(i, i_b, None))
        box_count.append(0 if boxes is None else len(boxes))

    output = {}
    if per_channel:
        for i, results in enumerate(results_list):
            for stat, result in results.items():
                output.setdefault(stat, []).extend(result.tolist())
    else:
        for results in results_list:
            for stat, result in results.items():
                output.setdefault(stat, []).append(result.tolist() if isinstance(result, np.ndarray) else result)

    for stat in output:
        stat_type: str = output_cls.__annotations__[stat]

        dtype_match = re.match(DTYPE_REGEX, stat_type)
        if dtype_match is not None:
            output[stat] = np.asarray(output[stat], dtype=np.dtype(dtype_match.group(1)))

    output[SOURCE_INDEX] = source_index
    output[BOX_COUNT] = np.asarray(box_count, dtype=np.uint16)

    return output
