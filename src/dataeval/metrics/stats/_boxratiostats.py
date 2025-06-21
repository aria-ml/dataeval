from __future__ import annotations

__all__ = []

import copy
from collections.abc import Callable
from typing import Any, Generic, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from dataeval.config import EPSILON
from dataeval.outputs._base import set_metadata
from dataeval.outputs._stats import BASE_ATTRS, BaseStatsOutput

TStatOutput = TypeVar("TStatOutput", bound=BaseStatsOutput, contravariant=True)
ArraySlice = tuple[int, int]


class BoxImageStatsOutputSlice(Generic[TStatOutput]):
    class StatSlicer:
        def __init__(self, stats: TStatOutput, slice: ArraySlice, channels: int = 0) -> None:  # noqa: A002
            self._stats = stats
            self._slice = slice
            self._channels = channels

        def __getitem__(self, key: str) -> NDArray[np.float64]:
            _stat = cast(np.ndarray, getattr(self._stats, key)).astype(np.float64)
            _shape = _stat[0].shape
            _slice = _stat[int(self._slice[0]) : int(self._slice[1])]
            return _slice.reshape(-1, self._channels, *_shape) if self._channels else _slice.reshape(-1, *_shape)

    box: StatSlicer
    img: StatSlicer
    channels: int

    def __init__(
        self, box_stats: TStatOutput, box_slice: ArraySlice, img_stats: TStatOutput, img_slice: ArraySlice
    ) -> None:
        self.channels = img_slice[1] - img_slice[0]
        self.box = self.StatSlicer(box_stats, box_slice, self.channels)
        self.img = self.StatSlicer(img_stats, img_slice)


RATIOSTATS_OVERRIDE_MAP: dict[str, Callable[[BoxImageStatsOutputSlice[Any]], NDArray[Any]]] = {
    "offset_x": lambda x: x.box["offset_x"] / x.img["width"],
    "offset_y": lambda x: x.box["offset_y"] / x.img["height"],
    "channels": lambda x: x.box["channels"],
    "depth": lambda x: x.box["depth"],
    "distance_center": lambda x: x.box["distance_center"]
    / (np.sqrt(np.square(x.img["width"]) + np.square(x.img["height"])) / 2),
    "distance_edge": lambda x: x.box["distance_edge"]
    / (
        x.img["width"]
        if np.min([np.abs(x.box["offset_x"]), np.abs((x.box["width"] + x.box["offset_x"]) - x.img["width"])])
        < np.min([np.abs(x.box["offset_y"]), np.abs((x.box["height"] + x.box["offset_y"]) - x.img["height"])])
        else x.img["height"]
    ),
}


def get_index_map(stats: BaseStatsOutput) -> list[int]:
    index_map: list[int] = []
    cur_index = -1
    for i, s in enumerate(stats.source_index):
        if s.image > cur_index:
            index_map.append(i)
            cur_index = s.image
    return index_map


def calculate_ratios(key: str, box_stats: BaseStatsOutput, img_stats: BaseStatsOutput) -> NDArray[np.float64]:
    if not hasattr(box_stats, key) or not hasattr(img_stats, key):
        raise KeyError("Invalid key for provided stats output object.")

    stats = getattr(box_stats, key)

    # Copy over base attributes
    if key in BASE_ATTRS:
        return copy.deepcopy(stats)

    # Calculate ratios for each stat
    out_stats: np.ndarray = np.copy(stats).astype(np.float64)

    box_map = get_index_map(box_stats)
    img_map = get_index_map(img_stats)
    for i, (box_i, img_i) in enumerate(zip(box_map, img_map)):
        box_j = len(box_stats) if i == len(box_map) - 1 else box_map[i + 1]
        img_j = len(img_stats) if i == len(img_map) - 1 else img_map[i + 1]
        stats = BoxImageStatsOutputSlice(box_stats, (box_i, box_j), img_stats, (img_i, img_j))
        use_override = key in RATIOSTATS_OVERRIDE_MAP
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = RATIOSTATS_OVERRIDE_MAP[key](stats) if use_override else stats.box[key] / (stats.img[key] + EPSILON)
        out_stats[box_i:box_j] = ratio.reshape(-1, *out_stats[box_i].shape)
    return out_stats


@set_metadata
def boxratiostats(
    boxstats: TStatOutput,
    imgstats: TStatOutput,
) -> TStatOutput:
    """
    Calculates ratio :term:`statistics<Statistics>` of box outputs over image outputs.

    Parameters
    ----------
    boxstats : DimensionStatsOutput | PixelStatsOutput | VisualStatsOutput
        Box statistics outputs to perform calculations on
    imgstats : DimensionStatsOutput | PixelStatsOutput | VisualStatsOutput
        Image statistics outputs to perform calculations on

    Returns
    -------
    DimensionStatsOutput | PixelStatsOutput | VisualStatsOutput
        A dictionary-like object containing the computed ratio of the box statistics divided by the
        image statistics.

    See Also
    --------
    dimensionstats, pixelstats, visualstats

    Note
    ----
    DimensionStatsOutput values for channels, depth and distances are the original values
    provided by the box outputs

    Examples
    --------
    Calculate the box ratio statistics using the dimension stats of the images and boxes
    on a dataset containing 15 targets.

    >>> from dataeval.metrics.stats import dimensionstats
    >>> imagestats = dimensionstats(dataset, per_box=False)
    >>> boxstats = dimensionstats(dataset, per_box=True)
    >>> ratiostats = boxratiostats(boxstats, imagestats)
    >>> print(ratiostats.aspect_ratio)
    [ 0.864  0.588 16.     0.857  1.27   0.438  0.667  3.833  1.95   0.833
      1.     0.6    0.522 15.     3.834]
    >>> print(ratiostats.size)
    [0.026 0.01  0.001 0.018 0.023 0.007 0.009 0.034 0.021 0.007 0.001 0.008
     0.017 0.001 0.008]
    """
    output_cls = type(boxstats)
    if type(boxstats) is not type(imgstats):
        raise TypeError("Must provide stats outputs of the same type.")
    if boxstats.image_count != imgstats.image_count:
        raise ValueError("Stats image count length mismatch. Check if the correct box and image stats were provided.")
    if any(src_idx.box is None for src_idx in boxstats.source_index):
        raise ValueError("Input for boxstats must contain box information.")
    if any(src_idx.box is not None for src_idx in imgstats.source_index):
        raise ValueError("Input for imgstats must not contain box information.")
    boxstats_has_channels = any(si.channel is None for si in boxstats.source_index)
    imgstats_has_channels = any(si.channel is None for si in imgstats.source_index)
    if boxstats_has_channels != imgstats_has_channels:
        raise ValueError("Input for boxstats and imgstats must have matching channel information.")

    output_dict = {}
    for key in boxstats.data():
        output_dict[key] = calculate_ratios(key, boxstats, imgstats)

    return output_cls(**output_dict)
