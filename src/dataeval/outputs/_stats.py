from __future__ import annotations

__all__ = []

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias

import numpy as np
import polars as pl
from numpy.typing import NDArray

from dataeval.outputs._base import Output
from dataeval.utils._plot import channel_histogram_plot, histogram_plot

if TYPE_CHECKING:
    from matplotlib.figure import Figure

OptionalRange: TypeAlias = int | Iterable[int] | None

SOURCE_INDEX = "source_index"
OBJECT_COUNT = "object_count"
IMAGE_COUNT = "image_count"

BASE_ATTRS = [SOURCE_INDEX, OBJECT_COUNT, IMAGE_COUNT]


class SourceIndex(NamedTuple):
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
    object_count : NDArray[np.uint16]
        The number of detected objects in each image
    """

    source_index: Sequence[SourceIndex]
    object_count: NDArray[np.uint16]
    image_count: int

    def __post_init__(self) -> None:
        si_length = len(self.source_index)
        mismatch = {k: len(v) for k, v in self.data().items() if k not in BASE_ATTRS and len(v) != si_length}
        if mismatch:
            raise ValueError(f"All values must have the same length as source_index. Bad values: {str(mismatch)}.")
        oc_length = len(self.object_count)
        if oc_length != self.image_count:
            raise ValueError(
                f"Total object counts per image does not match image count. {oc_length} != {self.image_count}."
            )

    def get_channel_mask(
        self,
        channel_index: OptionalRange,
        channel_count: OptionalRange = None,
    ) -> Sequence[bool]:
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
        mask: Sequence[bool] = []
        cur_mask: Sequence[bool] = []
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
    ) -> tuple[int, Sequence[bool] | None]:
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

    def factors(
        self,
        filter: str | Sequence[str] | None = None,  # noqa: A002
        exclude_constant: bool = False,
    ) -> Mapping[str, NDArray[Any]]:
        """
        Returns all 1-dimensional data as a dictionary of numpy arrays.

        Parameters
        ----------
        filter : str, Sequence[str] or None, default None:
            If provided, only returns keys that match the filter.
        exclude_constant : bool, default False
            If True, exclude arrays that contain only a single unique value.

        Returns
        -------
        Mapping[str, NDArray[Any]]
        """
        filter_ = [filter] if isinstance(filter, str) else filter

        """
        Performs validation checks to ensure selected keys and constant or 1-D values
        Each set of checks returns True if a valid value.
        Only one set of final checks needs to be True to allow the value through
        """
        return {
            k: v
            for k, v in self.data().items()
            if (
                k not in BASE_ATTRS  # Ignore BaseStatsOutput attributes
                and (filter_ is None or k in filter_)  # Key is selected
                and (isinstance(v, np.ndarray) and v.ndim == 1)  # Check valid array
                and (not exclude_constant or len(np.unique(v)) > 1)  # Check valid numpy "constant"
            )
        }

    def plot(
        self, log: bool, channel_limit: int | None = None, channel_index: int | Iterable[int] | None = None
    ) -> Figure:
        """
        Plots the statistics as a set of histograms.

        Parameters
        ----------
        log : bool
            If True, plots the histograms on a logarithmic scale.
        channel_limit : int or None
            The maximum number of channels to plot. If None, all channels are plotted.
        channel_index : int, Iterable[int] or None
            The index or indices of the channels to plot. If None, all channels are plotted.

        Returns
        -------
        matplotlib.Figure
        """
        from matplotlib.figure import Figure

        max_channels, ch_mask = self._get_channels(channel_limit, channel_index)
        factors = self.factors(exclude_constant=True)
        if not factors:
            return Figure()
        if max_channels == 1:
            return histogram_plot(factors, log)
        return channel_histogram_plot(factors, log, max_channels, ch_mask)

    def to_dataframe(self) -> pl.DataFrame:
        """Returns the processed factors a polars dataframe of shape (factors, samples)"""

        return pl.DataFrame(self.factors())


@dataclass(frozen=True)
class DimensionStatsOutput(BaseStatsOutput):
    """
    Output class for :func:`.dimensionstats` stats metric.

    Attributes
    ----------
    offset_x : NDArray[np.int32]
        Offsets from the left edge of images in pixels
    offset_y : NDArray[np.int32]
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
        :term:`Aspect Ratio<Aspect Ratio>` of the images (width/height)
    depth : NDArray[np.uint8]
        Color depth of the images in bits
    center : NDArray[np.uint32]
        Offset from center in [x,y] coordinates of the images in pixels
    distance_center : NDArray[np.float32]
        Distance in pixels from center
    distance_edge : NDArray[np.uint32]
        Distance in pixels from nearest edge
    """

    offset_x: NDArray[np.int32]
    offset_y: NDArray[np.int32]
    width: NDArray[np.uint32]
    height: NDArray[np.uint32]
    channels: NDArray[np.uint8]
    size: NDArray[np.uint32]
    aspect_ratio: NDArray[np.float16]
    depth: NDArray[np.uint8]
    center: NDArray[np.int32]
    distance_center: NDArray[np.float32]
    distance_edge: NDArray[np.uint32]


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

    xxhash: Sequence[str]
    pchash: Sequence[str]

    def to_dataframe(self) -> pl.DataFrame:
        """
        Returns a polars dataframe for the xxhash and pchash attributes of each sample

        Note
        ----
        xxhash and pchash do not follow the normal definition of factors but are
        helpful attributes of the data

        Examples
        --------
        Display the hashes of a dataset of images, whose shape is (C, H, W),
        as a polars DataFrame

        >>> from dataeval.metrics.stats import hashstats
        >>> results = hashstats(dataset)
        >>> print(results.to_dataframe())
        shape: (8, 2)
        ┌──────────────────┬──────────────────┐
        │ xxhash           ┆ pchash           │
        │ ---              ┆ ---              │
        │ str              ┆ str              │
        ╞══════════════════╪══════════════════╡
        │ 69b50a5f06af238c ┆ e666999999266666 │
        │ 5a861d7a23d1afe7 ┆ e666999999266666 │
        │ 7ffdb4990ad44ac6 ┆ e666999966666299 │
        │ 4f0c366a3298ceac ┆ e666999999266666 │
        │ c5519e36ac1f8839 ┆ 96e91656e91616e9 │
        │ e7e92346159a4567 ┆ e666999999266666 │
        │ 9a538f797a5ba8ee ┆ e666999999266666 │
        │ 1a658bd2a1baee25 ┆ e666999999266666 │
        └──────────────────┴──────────────────┘
        """
        data = {"xxhash": self.xxhash, "pchash": self.pchash}
        schema = {"xxhash": str, "pchash": str}
        return pl.DataFrame(data=data, schema=schema)


@dataclass(frozen=True)
class LabelStatsOutput(Output):
    """
    Output class for :func:`.labelstats` stats metric.

    Attributes
    ----------
    label_counts_per_class : Mapping[int, int]
        Dictionary whose keys are the different label classes and
        values are total counts of each class
    label_counts_per_image : Sequence[int]
        Number of labels per image
    image_counts_per_class : Mapping[int, int]
        Dictionary whose keys are the different label classes and
        values are total counts of each image the class is present in
    image_indices_per_class : Mapping[int, Sequence[int]]
        Dictionary whose keys are the different label classes and
        values are lists containing the images that have that label
    image_count : int
        Total number of images present
    class_count : int
        Total number of classes present
    label_count : int
        Total number of labels present
    class_names : Sequence[str]
    """

    label_counts_per_class: Mapping[int, int]
    label_counts_per_image: Sequence[int]
    image_counts_per_class: Mapping[int, int]
    image_indices_per_class: Mapping[int, Sequence[int]]
    image_count: int
    class_count: int
    label_count: int
    class_names: Sequence[str]

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

    def to_dataframe(self) -> pl.DataFrame:
        """
        Exports the label statistics output results to a polars DataFrame.

        Returns
        -------
        pl.DataFrame
        """
        total_count = []
        image_count = []
        for cls in range(len(self.class_names)):
            total_count.append(self.label_counts_per_class[cls])
            image_count.append(self.image_counts_per_class[cls])

        return pl.DataFrame(
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
