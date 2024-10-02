from __future__ import annotations

import copy
from typing import Callable, Generic, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from dataeval._internal.metrics.stats.base import BOX_COUNT, SOURCE_INDEX, BaseStatsOutput
from dataeval._internal.metrics.stats.dimensionstats import DimensionStatsOutput
from dataeval._internal.output import set_metadata

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
            _slice = _stat[self._slice[0] : self._slice[1]]
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


RATIOSTATS_OVERRIDE_MAP: dict[type, dict[str, Callable[[BoxImageStatsOutputSlice], NDArray]]] = {
    DimensionStatsOutput: {
        "left": lambda x: x.box["left"] / x.img["width"],
        "top": lambda x: x.box["top"] / x.img["height"],
        "channels": lambda x: x.box["channels"],
        "depth": lambda x: x.box["depth"],
        "distance": lambda x: x.box["distance"],
    }
}


def get_index_map(stats: BaseStatsOutput) -> list[int]:
    index_map: list[int] = []
    cur_index = -1
    for i, s in enumerate(stats.source_index):
        if s.image > cur_index:
            index_map.append(i)
            cur_index = s.image
    return index_map


def calculate_ratios(key: str, box_stats: BaseStatsOutput, img_stats: BaseStatsOutput) -> NDArray:
    if not hasattr(box_stats, key) or not hasattr(img_stats, key):
        raise KeyError("Invalid key for provided stats output object.")

    stats = getattr(box_stats, key)

    # Copy over stats index maps and box counts
    if key in (SOURCE_INDEX):
        return copy.deepcopy(stats)
    elif key == BOX_COUNT:
        return np.copy(stats)

    # Calculate ratios for each stat
    out_stats: np.ndarray = np.copy(stats).astype(np.float64)

    box_map = get_index_map(box_stats)
    img_map = get_index_map(img_stats)
    for i, (box_i, img_i) in enumerate(zip(box_map, img_map)):
        box_j = len(box_stats) if i == len(box_map) - 1 else box_map[i + 1]
        img_j = len(img_stats) if i == len(img_map) - 1 else img_map[i + 1]
        stats = BoxImageStatsOutputSlice(box_stats, (box_i, box_j), img_stats, (img_i, img_j))
        out_type = type(box_stats)
        use_override = out_type in RATIOSTATS_OVERRIDE_MAP and key in RATIOSTATS_OVERRIDE_MAP[out_type]
        ratio = (
            RATIOSTATS_OVERRIDE_MAP[out_type][key](stats)
            if use_override
            else np.nan_to_num(stats.box[key] / stats.img[key])
        )
        out_stats[box_i:box_j] = ratio.reshape(-1, *out_stats[box_i].shape)
    return out_stats


@set_metadata("dataeval.metrics")
def boxratiostats(
    boxstats: TStatOutput,
    imgstats: TStatOutput,
) -> TStatOutput:
    """
    Calculates ratio statistics of box outputs over image outputs

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
    Calculating the box ratio statistics using the dimension stats of the boxes and images

    >>> imagestats = dimensionstats(images)
    >>> boxstats = dimensionstats(images, bboxes)
    >>> ratiostats = boxratiostats(boxstats, imagestats)
    >>> print(ratiostats.aspect_ratio)
    [ 1.15169271  0.78450521 21.33333333  1.5234375   2.25651042  0.77799479
      0.88867188  3.40625     1.73307292  1.11132812  0.75018315  0.45018315
      0.69596354 20.          5.11197917  2.33333333  0.75        0.70019531]
    >>> print(ratiostats.size)
    [0.03401693 0.01383464 0.00130208 0.01822917 0.02327474 0.00683594
     0.01220703 0.0168457  0.01057943 0.00976562 0.00130208 0.01098633
     0.02246094 0.0012207  0.01123047 0.00911458 0.02636719 0.06835938]
    """
    output_cls = type(boxstats)
    if type(boxstats) is not type(imgstats):
        raise TypeError("Must provide stats outputs of the same type.")
    if boxstats.source_index[-1].image != imgstats.source_index[-1].image:
        raise ValueError("Stats index_map length mismatch. Check if the correct box and image stats were provided.")
    if all(count == 0 for count in boxstats.box_count):
        raise TypeError("Input for boxstats must contain box information.")
    if any(count != 0 for count in imgstats.box_count):
        raise TypeError("Input for imgstats must not contain box information.")
    boxstats_has_channels = any(si.channel is None for si in boxstats.source_index)
    imgstats_has_channels = any(si.channel is None for si in imgstats.source_index)
    if boxstats_has_channels != imgstats_has_channels:
        raise TypeError("Input for boxstats and imgstats must have matching channel information.")

    output_dict = {}
    for key in boxstats.dict():
        output_dict[key] = calculate_ratios(key, boxstats, imgstats)

    return output_cls(**output_dict)
