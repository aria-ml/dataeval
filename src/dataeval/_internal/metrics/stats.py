from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import entropy, kurtosis, skew

from dataeval._internal.interop import to_numpy_iter
from dataeval._internal.metrics.utils import (
    edge_filter,
    get_bitdepth,
    normalize_box_shape,
    normalize_image_shape,
    pchash,
    rescale,
    xxhash,
)
from dataeval._internal.output import OutputMetadata, set_metadata

DTYPE_REGEX = re.compile(r"NDArray\[np\.(.*?)\]")
INDEX_MAP = "index_map"
QUARTILES = (0, 25, 50, 75, 100)
STATS_FN_MAP: dict[str, dict[str, Callable[[NDArray], Any]]] = {
    "image": {
        # hash stats
        "xxhash": lambda x: xxhash(x),
        "pchash": lambda x: pchash(x),
        # dimension stats
        "width": lambda x: np.uint16(x.shape[-1]),
        "height": lambda x: np.uint16(x.shape[-2]),
        "channels": lambda x: np.uint8(x.shape[-3]),
        "size": lambda x: np.uint32(np.prod(x.shape[-2:])),
        "aspect_ratio": lambda x: np.float16(x.shape[-1] / x.shape[-2]),
        "depth": lambda x: np.uint8(get_bitdepth(x).depth),
        "box_count": lambda x: np.uint16(x.shape[0]),
        "box_center": lambda x: np.asarray([(x[0] + x[2]) / 2, (x[1] + x[3]) / 2], dtype=np.uint16),
        # visual stats
        "brightness": lambda x: x[-2],
        "blurriness": lambda x: np.float16(np.std(edge_filter(np.mean(x, axis=0)))),
        "contrast": lambda x: np.float16((np.max(x) - np.min(x)) / np.mean(x)),
        "darkness": lambda x: x[1],
        "missing": lambda x: np.float16(np.sum(np.isnan(x)) / np.prod(x.shape[-2:])),
        "zeros": lambda x: np.float16(np.count_nonzero(x == 0) / np.prod(x.shape[-2:])),
        # pixel stats
        "mean": lambda x: np.float16(np.mean(x)),
        "std": lambda x: np.float16(np.std(x)),
        "var": lambda x: np.float16(np.var(x)),
        "skew": lambda x: np.float16(skew(x.ravel())),
        "kurtosis": lambda x: np.float16(kurtosis(x.ravel())),
        "percentiles": lambda x: np.float16(np.nanpercentile(x, q=QUARTILES)),
        "histogram": lambda x: np.uint32(np.histogram(x, 256, (0, 1))[0]),
        "entropy": lambda x: np.float16(entropy(x)),
    },
    "channel": {
        # visual stats
        "brightness": lambda x: x[:, -2],
        "blurriness": lambda x: np.float16(np.std(np.vectorize(edge_filter, signature="(m,n)->(m,n)")(x), axis=(1, 2))),
        "contrast": lambda x: np.float16((np.max(x, axis=1) - np.min(x, axis=1)) / np.mean(x, axis=1)),
        "darkness": lambda x: x[:, 1],
        "missing": lambda x: np.float16(np.sum(np.isnan(x), axis=(1, 2)) / np.prod(x.shape[-2:])),
        "zeros": lambda x: np.float16(np.count_nonzero(x == 0, axis=(1, 2)) / np.prod(x.shape[-2:])),
        # pixel stats
        "mean": lambda x: np.float16(np.mean(x, axis=1)),
        "std": lambda x: np.float16(np.std(x, axis=1)),
        "var": lambda x: np.float16(np.var(x, axis=1)),
        "skew": lambda x: np.float16(skew(x, axis=1)),
        "kurtosis": lambda x: np.float16(kurtosis(x, axis=1)),
        "percentiles": lambda x: np.float16(np.nanpercentile(x, q=QUARTILES, axis=1).T),
        "histogram": lambda x: np.uint32(np.apply_along_axis(lambda y: np.histogram(y, 256, (0, 1))[0], 1, x)),
        "entropy": lambda x: np.float16(entropy(x, axis=1)),
    },
}


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


@dataclass(frozen=True)
class HashStatsOutput(BaseStatsOutput):
    """
    Attributes
    ----------
    xxhash : List[str]
        xxHash hash of the images as a hex string
    pchash : List[str]
        Perception hash of the images as a hex string
    """

    xxhash: list[str]
    pchash: list[str]


@dataclass(frozen=True)
class DimensionStatsOutput(BaseStatsOutput):
    """
    Attributes
    ----------
    width : NDArray[np.uint16]
        Width of the images in pixels
    height : NDArray[np.uint16]
        Height of the images in pixels
    channels : NDArray[np.uint8]
        Channel count of the images in pixels
    size : NDArray[np.uint32]
        Size of the images in pixels
    aspect_ratio : NDArray[np.float16]
        Aspect ratio of the images (width/height)
    depth : NDArray[np.uint8]
        Color depth of the images in bits
    """

    width: NDArray[np.uint16]
    height: NDArray[np.uint16]
    channels: NDArray[np.uint8]
    size: NDArray[np.uint32]
    aspect_ratio: NDArray[np.float16]
    depth: NDArray[np.uint8]


@dataclass(frozen=True)
class VisualStatsOutput(BaseStatsOutput):
    """
    Attributes
    ----------
    brightness : NDArray[np.float16]
        Brightness of the images
    blurriness : NDArray[np.float16]
        Blurriness of the images
    contrast : NDArray[np.float16]
        Image contrast ratio
    darkness : NDArray[np.float16]
        Darkness of the images
    missing : NDArray[np.float16]
        Percentage of the images with missing pixels
    zeros : NDArray[np.float16]
        Percentage of the images with zero value pixels
    """

    brightness: NDArray[np.float16]
    blurriness: NDArray[np.float16]
    contrast: NDArray[np.float16]
    darkness: NDArray[np.float16]
    missing: NDArray[np.float16]
    zeros: NDArray[np.float16]


@dataclass(frozen=True)
class PixelStatsOutput(BaseStatsOutput):
    """
    Attributes
    ----------
    mean : NDArray[np.float16]
        Mean of the pixel values of the images
    std : NDArray[np.float16]
        Standard deviation of the pixel values of the images
    var : NDArray[np.float16]
        Variance of the pixel values of the images
    skew : NDArray[np.float16]
        Skew of the pixel values of the images
    kurtosis : NDArray[np.float16]
        Kurtosis of the pixel values of the images
    percentiles : NDArray[np.float16]
        Percentiles of the pixel values of the images with quartiles of (0, 25, 50, 75, 100)
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
    percentiles: NDArray[np.float16]
    histogram: NDArray[np.uint32]
    entropy: NDArray[np.float16]


class ImageProcessingCache:
    def __init__(self, image: NDArray, per_channel: bool):
        self.image = image
        self.per_channel = per_channel
        self._normalized = None
        self._scaled = None
        self._percentiles = None
        self._norm_bboxes = None
        self.fn_map = STATS_FN_MAP["channel" if per_channel else "image"]

    @property
    def normalized(self) -> NDArray:
        if self._normalized is None:
            self._normalized = normalize_image_shape(self.image)
        return self._normalized

    @property
    def scaled(self) -> NDArray:
        if self._scaled is None:
            self._scaled = rescale(self.normalized)
            if self.per_channel:
                self._scaled = self._scaled.reshape(self.image.shape[0], -1)
        return self._scaled

    @property
    def percentiles(self) -> NDArray:
        if self._percentiles is None:
            self._percentiles = self.fn_map["percentiles"](self.scaled)
        return self._percentiles

    def get_data_for_stat(self, stat: str) -> NDArray:
        if stat in ("mean", "std", "var", "skew", "kurtosis", "percentiles", "histogram"):
            return self.scaled
        if stat in ("entropy"):
            # This can be cached if additional functions need histogram input
            return self.fn_map["histogram"](self.scaled)
        if stat in ("brightness", "contrast", "darkness"):
            return self.percentiles
        return self.normalized


def run_stats(
    images: Iterable[ArrayLike],
    bboxes: Iterable[ArrayLike] | None,
    per_channel: bool,
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
    fn_map = STATS_FN_MAP["channel" if per_channel else "image"]
    index_map = []
    if bboxes is None:
        for i, image in enumerate(to_numpy_iter(images)):
            cache = ImageProcessingCache(image, per_channel)
            results_list.append({stat: fn_map[stat](cache.get_data_for_stat(stat)) for stat in output_list})
            if per_channel:
                index_map.extend([SourceIndex(i, None, c) for c in range(image.shape[-3] if per_channel else 1)])
            else:
                index_map.append(SourceIndex(i, None, None))
    else:
        for i, (boxes, image) in enumerate(zip(to_numpy_iter(bboxes), to_numpy_iter(images))):
            nboxes = normalize_box_shape(boxes)
            for i_b, box in enumerate(nboxes):
                cache = ImageProcessingCache(image[:, box[1] : box[1] + box[3], box[0] : box[0] + box[2]], per_channel)
                results_list.append({stat: fn_map[stat](cache.get_data_for_stat(stat)) for stat in output_list})
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


@set_metadata("dataeval.metrics")
def hashstats(images: Iterable[ArrayLike]) -> HashStatsOutput:
    """
    Calculates hashes for each image

    This function computes hashes from the images including exact hashes and perception-based
    hashes. These hash values can be used to determine if images are exact or near matches.

    Parameters
    ----------
    images : ArrayLike
        Images to run statistical tests on

    Returns
    -------
    HashStatsOutput
        A dictionary-like object containing the computed hashes for each image.

    See Also
    --------
    Duplicates

    Examples
    --------
    Calculating the statistics on the images, whose shape is (C, H, W)

    >>> results = hashstats(images)
    >>> print(results.xxhash)
    ['a72434443d6e7336', 'efc12c2f14581d79', '4a1e03483a27d674', '3a3ecedbcf814226']
    >>> print(results.pchash)
    ['8f25506af46a7c6a', '8000808000008080', '8e71f18e0ef18e0e', 'a956d6a956d6a928']
    """
    output = run_stats(images, None, False, HashStatsOutput)
    return HashStatsOutput(**output)


@set_metadata("dataeval.metrics")
def dimensionstats(
    images: Iterable[ArrayLike],
    bboxes: Iterable[ArrayLike] | None = None,
) -> DimensionStatsOutput:
    """
    Calculates dimension statistics for each image

    This function computes various dimensional metrics (e.g., width, height, channels)
    on the images or individual bounding boxes for each image.

    Parameters
    ----------
    images : Iterable[ArrayLike]
        Images to perform calculations on
    bboxes : Iterable[ArrayLike] or None
        Bounding boxes for each image to perform calculations on

    Returns
    -------
    DimensionStatsOutput
        A dictionary-like object containing the computed dimension statistics for each image or bounding
        box. The keys correspond to the names of the statistics (e.g., 'width', 'height'), and the values
        are lists of results for each image or numpy arrays when the results are multi-dimensional.

    See Also
    --------
    pixelstats, visualstats, Outliers

    Examples
    --------
    Calculating the dimension statistics on the images, whose shape is (C, H, W)

    >>> results = dimensionstats(images)
    >>> print(results.aspect_ratio)
    [0.75  0.75  0.75  0.75  0.75  0.75  1.333 0.75  0.75  1.   ]
    >>> print(results.channels)
    [1 1 1 1 1 1 3 1 1 3]
    """
    output = run_stats(images, bboxes, False, DimensionStatsOutput)
    return DimensionStatsOutput(**output)


@set_metadata("dataeval.metrics")
def visualstats(
    images: Iterable[ArrayLike],
    bboxes: Iterable[ArrayLike] | None = None,
    per_channel: bool = False,
) -> VisualStatsOutput:
    """
    Calculates visual statistics for each image

    This function computes various visual metrics (e.g., brightness, darkness, contrast, blurriness)
    on the images as a whole.

    Parameters
    ----------
    images : Iterable[ArrayLike]
        Images to perform calculations on
    bboxes : Iterable[ArrayLike] or None
        Bounding boxes for each image to perform calculations on

    Returns
    -------
    VisualStatsOutput
        A dictionary-like object containing the computed visual statistics for each image. The keys correspond
        to the names of the statistics (e.g., 'brightness', 'blurriness'), and the values are lists of results for
        each image or numpy arrays when the results are multi-dimensional.

    See Also
    --------
    dimensionstats, pixelstats, Outliers

    Notes
    -----
    - `zeros` and `missing` are presented as a percentage of total pixel counts

    Examples
    --------
    Calculating the statistics on the images, whose shape is (C, H, W)

    >>> results = visualstats(images)
    >>> print(results.brightness)
    [0.0737 0.607  0.0713 0.1046 0.138  0.1713 0.2046 0.2379 0.2712 0.3047
     0.338  0.3713 0.4045 0.438  0.4712 0.5044 0.538  0.5713 0.6045 0.638
     0.6714 0.7046 0.738  0.7715 0.8047 0.838  0.871  0.905  0.938  0.971 ]
    >>> print(results.contrast)
    [2.041 1.332 1.293 1.279 1.271 1.269 1.265 1.264 1.261 1.26  1.259 1.258
     1.258 1.257 1.256 1.256 1.256 1.255 1.255 1.255 1.254 1.254 1.254 1.254
     1.254 1.253 1.254 1.253 1.254 1.254]
    """
    output = run_stats(images, bboxes, per_channel, VisualStatsOutput)
    return VisualStatsOutput(**output)


@set_metadata("dataeval.metrics")
def pixelstats(
    images: Iterable[ArrayLike],
    bboxes: Iterable[ArrayLike] | None = None,
    per_channel: bool = False,
) -> PixelStatsOutput:
    """
    Calculates pixel statistics for each image

    This function computes various statistical metrics (e.g., mean, standard deviation, entropy)
    on the images as a whole.

    Parameters
    ----------
    images : Iterable[ArrayLike]
        Images to perform calculations on
    bboxes : Iterable[ArrayLike] or None
        Bounding boxes for each image to perform calculations on

    Returns
    -------
    PixelStatsOutput
        A dictionary-like object containing the computed statistics for each image. The keys correspond
        to the names of the statistics (e.g., 'mean', 'std'), and the values are lists of results for
        each image or numpy arrays when the results are multi-dimensional.

    See Also
    --------
    dimensionstats, visualstats, Outliers

    Notes
    -----
    - All metrics are scaled based on the perceived bit depth (which is derived from the largest pixel value)
      to allow for better comparison between images stored in different formats and different resolutions.

    Examples
    --------
    Calculating the statistics on the images, whose shape is (C, H, W)

    >>> results = pixelstats(images)
    >>> print(results.mean)
    [0.04828 0.562   0.06726 0.09937 0.1315  0.1636  0.1957  0.2278  0.26
     0.292   0.3242  0.3562  0.3884  0.4204  0.4526  0.4846  0.5166  0.549
     0.581   0.6133  0.6455  0.6772  0.7095  0.7417  0.774   0.8057  0.838
     0.87    0.9023  0.934  ]
    >>> print(results.entropy)
    [3.238  3.303  0.8125 1.028  0.8223 1.046  0.8247 1.041  0.8203 1.012
     0.812  0.9883 0.795  0.9243 0.9243 0.795  0.9907 0.8125 1.028  0.8223
     1.046  0.8247 1.041  0.8203 1.012  0.812  0.9883 0.795  0.9243 0.9243]
    """
    output = run_stats(images, bboxes, per_channel, PixelStatsOutput)
    return PixelStatsOutput(**output)
