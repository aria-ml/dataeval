from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dataeval._internal.interop import to_numpy_iter
from dataeval._internal.metrics.utils import normalize_box_shape, normalize_image_shape, rescale
from dataeval._internal.output import OutputMetadata

DTYPE_REGEX = re.compile(r"NDArray\[np\.(.*?)\]")
INDEX_MAP = "index_map"


class StatsFunctionMap:
    image: dict[str, Callable[[ImageProcessingCache], Any]] = {}
    channel: dict[str, Callable[[ImageProcessingCache], Any]] = {}


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
    index_map : List[SourceIndex]
        Mapping from statistic to source image, box and channel index
    """

    index_map: list[SourceIndex]

    def get_channel_mask(self, channel_index: int | None, channel_count: int | None = None) -> list[bool]:
        """
        Boolean mask for results filtered to specified channel index and optionally the count
        of the channels per image.

        Parameters
        ----------
        channel_index : int
            Index of channel to filter for
        channel_count : int or None
            Optional count of channels to filter for
        """
        mask: list[bool] = []
        cur_mask: list[bool] = []
        cur_image = 0
        cur_max_channel = 0
        for source_index in list(self.index_map) + [None]:
            if source_index is None or source_index.image > cur_image:
                mask.extend(
                    cur_mask
                    if channel_count is None or cur_max_channel == channel_count - 1
                    else [False for _ in cur_mask]
                )
                if source_index is None:
                    break
                cur_image = source_index.image
                cur_max_channel = 0
                cur_mask.clear()
            cur_mask.append(channel_index is None or source_index.channel == channel_index)
            cur_max_channel = max(cur_max_channel, source_index.channel or 0)
        return mask

    def __len__(self) -> int:
        for a in self.__annotations__:
            attr = getattr(self, a, None)
            if attr is not None and hasattr(a, "__len__") and len(attr) > 0:
                return len(attr)
        return 0


class ImageProcessingCache:
    def __init__(self, image: NDArray, box: NDArray | None, per_channel: bool, function_map: StatsFunctionMap):
        self.raw = image
        self.box = np.array([0, 0, image.shape[-1], image.shape[-2]]) if box is None else box
        self.per_channel = per_channel
        self._image = None
        self._shape = None
        self._scaled = None
        self._histogram = None
        self._percentiles = None
        self.fn_map = function_map.channel if per_channel else function_map.image

    def calculate(self, fn_key: str) -> NDArray:
        if fn_key == "percentiles":
            return self.percentiles
        elif fn_key == "histogram":
            return self.histogram
        else:
            return self.fn_map[fn_key](self)

    @property
    def image(self) -> NDArray:
        if self._image is None:
            norm = normalize_image_shape(self.raw)
            self._image = norm[:, self.box[1] : self.box[1] + self.box[3], self.box[0] : self.box[0] + self.box[2]]
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

    @property
    def histogram(self) -> NDArray:
        if self._histogram is None:
            self._histogram = self.fn_map["histogram"](self)
        return self._histogram

    @property
    def percentiles(self) -> NDArray:
        if self._percentiles is None:
            self._percentiles = self.fn_map["percentiles"](self)
        return self._percentiles


def run_stats(
    images: Iterable[ArrayLike],
    bboxes: Iterable[ArrayLike] | None,
    per_channel: bool,
    function_map: StatsFunctionMap,
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
        as an array-like structure in the format of (X, Y, W, H). The length of the bounding boxes
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

    Notes
    -----
    - The function performs image normalization (rescaling the image values)
      before applying some of the statistics.
    - Pixel-level statistics (e.g., brightness, entropy) are computed after
      rescaling and, optionally, flattening the images.
    - For statistics like histograms and entropy, intermediate results may
      be reused to avoid redundant computation.
    """
    results_list: list[dict[str, NDArray]] = []
    output_list = list(output_cls.__annotations__)
    index_map = []
    if bboxes is None:
        for i, image in enumerate(to_numpy_iter(images)):
            cache = ImageProcessingCache(image, None, per_channel, function_map)
            results_list.append({stat: cache.calculate(stat) for stat in output_list})
            if per_channel:
                index_map.extend([SourceIndex(i, None, c) for c in range(image.shape[-3] if per_channel else 1)])
            else:
                index_map.append(SourceIndex(i, None, None))
    else:
        for i, (boxes, image) in enumerate(zip(to_numpy_iter(bboxes), to_numpy_iter(images))):
            nboxes = normalize_box_shape(boxes)
            for i_b, box in enumerate(nboxes):
                cache = ImageProcessingCache(image, box, per_channel, function_map)
                results_list.append({stat: cache.calculate(stat) for stat in output_list})
                if per_channel:
                    index_map.extend([SourceIndex(i, i_b, c) for c in range(image.shape[-3] if per_channel else 1)])
                else:
                    index_map.append(SourceIndex(i, i_b, None))

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

    output[INDEX_MAP] = index_map

    return output
