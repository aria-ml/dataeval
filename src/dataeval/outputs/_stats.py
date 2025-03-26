from __future__ import annotations

__all__ = []

import contextlib
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Union

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeAlias

with contextlib.suppress(ImportError):
    import pandas as pd

from dataeval.outputs._base import Output
from dataeval.utils._plot import channel_histogram_plot, histogram_plot

OptionalRange: TypeAlias = Optional[Union[int, Iterable[int]]]

SOURCE_INDEX = "source_index"
BOX_COUNT = "box_count"


@dataclass(frozen=True)
class SourceIndex:
    """
    The indices of the source image, box and channel.

    Attributes
    ----------
    image: int
        Index of the source image
    box : int | None
        Index of the box of the source image (if applicable)
    channel : int | None
        Index of the channel of the source image (if applicable)
    """

    image: int
    box: int | None
    channel: int | None


def matches(index: int | None, opt_range: OptionalRange) -> bool:
    if index is None or opt_range is None:
        return True
    return index in opt_range if isinstance(opt_range, Iterable) else index == opt_range


@dataclass(frozen=True)
class BaseStatsOutput(Output):
    """
    Attributes
    ----------
    source_index : List[SourceIndex]
        Mapping from statistic to source image, box and channel index
    box_count : NDArray[np.uint16]
    """

    source_index: list[SourceIndex]
    box_count: NDArray[np.uint16]

    def __post_init__(self) -> None:
        length = len(self.source_index)
        bad = {k: len(v) for k, v in self.data().items() if k not in [SOURCE_INDEX, BOX_COUNT] and len(v) != length}
        if bad:
            raise ValueError(f"All values must have the same length as source_index. Bad values: {str(bad)}.")

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
                if source_index is not None:
                    cur_image = source_index.image
                    cur_max_channel = 0
                    cur_mask.clear()
            if source_index is not None:
                cur_mask.append(matches(source_index.channel, channel_index))
                cur_max_channel = max(cur_max_channel, source_index.channel or 0)
        return mask

    def __len__(self) -> int:
        return len(self.source_index)

    def _get_channels(
        self, channel_limit: int | None = None, channel_index: int | Iterable[int] | None = None
    ) -> tuple[int, list[bool] | None]:
        source_index = self.data()[SOURCE_INDEX]
        raw_channels = int(max([si.channel or 0 for si in source_index])) + 1
        if isinstance(channel_index, int):
            max_channels = 1 if channel_index < raw_channels else raw_channels
            ch_mask = self.get_channel_mask(channel_index)
        elif isinstance(channel_index, Iterable) and all(isinstance(val, int) for val in list(channel_index)):
            max_channels = len(list(channel_index))
            ch_mask = self.get_channel_mask(channel_index)
        elif isinstance(channel_limit, int):
            max_channels = channel_limit
            ch_mask = self.get_channel_mask(None, channel_limit)
        else:
            max_channels = raw_channels
            ch_mask = None

        if max_channels > raw_channels:
            max_channels = raw_channels
        if ch_mask is not None and not any(ch_mask):
            ch_mask = None

        return max_channels, ch_mask

    def factors(self) -> dict[str, NDArray[Any]]:
        return {
            k: v
            for k, v in self.data().items()
            if k not in (SOURCE_INDEX, BOX_COUNT) and isinstance(v, np.ndarray) and v[v != 0].size > 0 and v.ndim == 1
        }

    def plot(
        self, log: bool, channel_limit: int | None = None, channel_index: int | Iterable[int] | None = None
    ) -> None:
        max_channels, ch_mask = self._get_channels(channel_limit, channel_index)
        if max_channels == 1:
            histogram_plot(self.factors(), log)
        else:
            channel_histogram_plot(self.factors(), log, max_channels, ch_mask)


@dataclass(frozen=True)
class DimensionStatsOutput(BaseStatsOutput):
    """
    Output class for :func:`.dimensionstats` stats metric.

    Attributes
    ----------
    left : NDArray[np.int32]
        Offsets from the left edge of images in pixels
    top : NDArray[np.int32]
        Offsets from the top edge of images in pixels
    width : NDArray[np.uint32]
        Width of the images in pixels
    height : NDArray[np.uint32]
        Height of the images in pixels
    channels : NDArray[np.uint8]
        Channel count of the images in pixels
    size : NDArray[np.uint32]
        Size of the images in pixels
    aspect_ratio : NDArray[np.float16]
        :term:`ASspect Ratio<Aspect Ratio>` of the images (width/height)
    depth : NDArray[np.uint8]
        Color depth of the images in bits
    center : NDArray[np.uint16]
        Offset from center in [x,y] coordinates of the images in pixels
    distance : NDArray[np.float16]
        Distance in pixels from center
    """

    left: NDArray[np.int32]
    top: NDArray[np.int32]
    width: NDArray[np.uint32]
    height: NDArray[np.uint32]
    channels: NDArray[np.uint8]
    size: NDArray[np.uint32]
    aspect_ratio: NDArray[np.float16]
    depth: NDArray[np.uint8]
    center: NDArray[np.int16]
    distance: NDArray[np.float16]


@dataclass(frozen=True)
class HashStatsOutput(BaseStatsOutput):
    """
    Output class for :func:`.hashstats` stats metric.

    Attributes
    ----------
    xxhash : List[str]
        xxHash hash of the images as a hex string
    pchash : List[str]
        :term:`Perception-based Hash` of the images as a hex string
    """

    xxhash: list[str]
    pchash: list[str]


@dataclass(frozen=True)
class LabelStatsOutput(Output):
    """
    Output class for :func:`.labelstats` stats metric.

    Attributes
    ----------
    label_counts_per_class : dict[int, int]
        Dictionary whose keys are the different label classes and
        values are total counts of each class
    label_counts_per_image : list[int]
        Number of labels per image
    image_counts_per_class : dict[int, int]
        Dictionary whose keys are the different label classes and
        values are total counts of each image the class is present in
    image_indices_per_class : dict[int, list]
        Dictionary whose keys are the different label classes and
        values are lists containing the images that have that label
    image_count : int
        Total number of images present
    class_count : int
        Total number of classes present
    label_count : int
        Total number of labels present
    class_names : list[str]
    """

    label_counts_per_class: list[int]
    label_counts_per_image: list[int]
    image_counts_per_class: list[int]
    image_indices_per_class: list[list[int]]
    image_count: int
    class_count: int
    label_count: int
    class_names: list[str]

    def to_table(self) -> str:
        """
        Formats the label statistics output results as a table.

        Returns
        -------
        str
        """
        max_char = max(len(name) if isinstance(name, str) else name // 10 + 1 for name in self.class_names)
        max_char = max(max_char, 5)
        max_label = max(list(self.label_counts_per_class))
        max_img = max(list(self.image_counts_per_class))
        max_num = int(np.ceil(np.log10(max(max_label, max_img))))
        max_num = max(max_num, 11)

        # Display basic counts
        table_str = [f"Class Count: {self.class_count}"]
        table_str += [f"Label Count: {self.label_count}"]
        table_str += [f"Average # Labels per Image: {round(np.mean(self.label_counts_per_image), 2)}"]
        table_str += ["--------------------------------------"]

        # Display counts per class
        table_str += [f"{'Label':>{max_char}}: Total Count - Image Count"]
        for cls in range(len(self.class_names)):
            table_str += [
                f"{self.class_names[cls]:>{max_char}}: {self.label_counts_per_class[cls]:^{max_num}}"
                + " - "
                + f"{self.image_counts_per_class[cls]:^{max_num}}".rstrip()
            ]

        return "\n".join(table_str)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Exports the label statistics output results to a pandas DataFrame.

        Notes
        -----
        This method requires `pandas <https://pandas.pydata.org/>`_ to be installed.

        Returns
        -------
        pd.DataFrame
        """
        import pandas as pd

        total_count = []
        image_count = []
        for cls in range(len(self.class_names)):
            total_count.append(self.label_counts_per_class[cls])
            image_count.append(self.image_counts_per_class[cls])

        return pd.DataFrame(
            {
                "Label": self.class_names,
                "Total Count": total_count,
                "Image Count": image_count,
            }
        )


@dataclass(frozen=True)
class PixelStatsOutput(BaseStatsOutput):
    """
    Output class for :func:`.pixelstats` stats metric.

    Attributes
    ----------
    mean : NDArray[np.float16]
        Mean of the pixel values of the images
    std : NDArray[np.float16]
        Standard deviation of the pixel values of the images
    var : NDArray[np.float16]
        :term:`Variance` of the pixel values of the images
    skew : NDArray[np.float16]
        Skew of the pixel values of the images
    kurtosis : NDArray[np.float16]
        Kurtosis of the pixel values of the images
    histogram : NDArray[np.uint32]
        Histogram of the pixel values of the images across 256 bins scaled between 0 and 1
    entropy : NDArray[np.float16]
        Entropy of the pixel values of the images
    """

    mean: NDArray[np.float16]
    std: NDArray[np.float16]
    var: NDArray[np.float16]
    skew: NDArray[np.float16]
    kurtosis: NDArray[np.float16]
    histogram: NDArray[np.uint32]
    entropy: NDArray[np.float16]


@dataclass(frozen=True)
class VisualStatsOutput(BaseStatsOutput):
    """
    Output class for :func:`.visualstats` stats metric.

    Attributes
    ----------
    brightness : NDArray[np.float16]
        Brightness of the images
    contrast : NDArray[np.float16]
        Image contrast ratio
    darkness : NDArray[np.float16]
        Darkness of the images
    missing : NDArray[np.float16]
        Percentage of the images with missing pixels
    sharpness : NDArray[np.float16]
        Sharpness of the images
    zeros : NDArray[np.float16]
        Percentage of the images with zero value pixels
    percentiles : NDArray[np.float16]
        Percentiles of the pixel values of the images with quartiles of (0, 25, 50, 75, 100)
    """

    brightness: NDArray[np.float16]
    contrast: NDArray[np.float16]
    darkness: NDArray[np.float16]
    missing: NDArray[np.float16]
    sharpness: NDArray[np.float16]
    zeros: NDArray[np.float16]
    percentiles: NDArray[np.float16]


@dataclass(frozen=True)
class ImageStatsOutput(DimensionStatsOutput, PixelStatsOutput, VisualStatsOutput):
    """
    Output class for :func:`.imagestats` stats metric with `per_channel=False`.

    This class represents the combined outputs of various stats functions against a
    single dataset, such that each index across all stat outputs are representative
    of the same source image. Modifying or mixing outputs will result in inaccurate
    outlier calculations if not created correctly.

    The attributes and methods are a superset of :class:`.DimensionStatsOutput`,
    :class:`.PixelStatsOutput` and :class:`.VisualStatsOutput`.
    """


@dataclass(frozen=True)
class ChannelStatsOutput(PixelStatsOutput, VisualStatsOutput):
    """
    Output class for :func:`.imagestats` stats metric with `per_channel=True`.

    This class represents the outputs of various per-channel stats functions against
    a single dataset, such that each index across all stat outputs are representative
    of the same source image. Modifying or mixing outputs will result in inaccurate
    outlier calculations if not created correctly.

    The attributes and methods are a superset of :class:`.PixelStatsOutput` and
    :class:`.VisualStatsOutput`.
    """
