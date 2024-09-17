from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import entropy, kurtosis, skew

from dataeval._internal.flags import ImageStat, to_distinct, verify_supported
from dataeval._internal.interop import to_numpy_iter
from dataeval._internal.metrics.utils import edge_filter, get_bitdepth, normalize_image_shape, pchash, rescale, xxhash
from dataeval._internal.output import OutputMetadata, populate_defaults, set_metadata

CH_IDX_MAP = "ch_idx_map"


@dataclass(frozen=True)
class StatsOutput(OutputMetadata):
    """
    Attributes
    ----------
    xxhash : List[str]
        xxHash hash of the images as a hex string
    pchash : List[str]
        Perception hash of the images as a hex string
    width: NDArray[np.uint16]
        Width of the images in pixels
    height: NDArray[np.uint16]
        Height of the images in pixels
    channels: NDArray[np.uint8]
        Channel count of the images in pixels
    size: NDArray[np.uint32]
        Size of the images in pixels
    aspect_ratio: NDArray[np.float16]
        Aspect ratio of the images (width/height)
    depth: NDArray[np.uint8]
        Color depth of the images in bits
    brightness: NDArray[np.float16]
        Brightness of the images
    blurriness: NDArray[np.float16]
        Blurriness of the images
    missing: NDArray[np.float16]
        Percentage of the images with missing pixels
    zero: NDArray[np.float16]
        Percentage of the images with zero value pixels
    mean: NDArray[np.float16]
        Mean of the pixel values of the images
    std: NDArray[np.float16]
        Standard deviation of the pixel values of the images
    var: NDArray[np.float16]
        Variance of the pixel values of the images
    skew: NDArray[np.float16]
        Skew of the pixel values of the images
    kurtosis: NDArray[np.float16]
        Kurtosis of the pixel values of the images
    percentiles: NDArray[np.float16]
        Percentiles of the pixel values of the images with quartiles of (0, 25, 50, 75, 100)
    histogram: NDArray[np.uint32]
        Histogram of the pixel values of the images across 256 bins scaled between 0 and 1
    entropy: NDArray[np.float16]
        Entropy of the pixel values of the images
    ch_idx_map: Dict[int, List[int]]
        Per-channel mapping of indices for each metric
    """

    xxhash: list[str]
    pchash: list[str]
    width: NDArray[np.uint16]
    height: NDArray[np.uint16]
    channels: NDArray[np.uint8]
    size: NDArray[np.uint32]
    aspect_ratio: NDArray[np.float16]
    depth: NDArray[np.uint8]
    brightness: NDArray[np.float16]
    blurriness: NDArray[np.float16]
    missing: NDArray[np.float16]
    zero: NDArray[np.float16]
    mean: NDArray[np.float16]
    std: NDArray[np.float16]
    var: NDArray[np.float16]
    skew: NDArray[np.float16]
    kurtosis: NDArray[np.float16]
    percentiles: NDArray[np.float16]
    histogram: NDArray[np.uint32]
    entropy: NDArray[np.float16]
    ch_idx_map: dict[int, list[int]]

    def dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_") and len(v) > 0}

    def __len__(self) -> int:
        if self.ch_idx_map:
            return sum([len(idxs) for idxs in self.ch_idx_map.values()])
        else:
            for a in self.__annotations__:
                attr = getattr(self, a, None)
                if attr is not None and hasattr(a, "__len__") and len(attr) > 0:
                    return len(attr)
        return 0


QUARTILES = (0, 25, 50, 75, 100)

IMAGESTATS_FN_MAP: dict[ImageStat, Callable[[NDArray], Any]] = {
    ImageStat.XXHASH: lambda x: xxhash(x),
    ImageStat.PCHASH: lambda x: pchash(x),
    ImageStat.WIDTH: lambda x: np.uint16(x.shape[-1]),
    ImageStat.HEIGHT: lambda x: np.uint16(x.shape[-2]),
    ImageStat.CHANNELS: lambda x: np.uint8(x.shape[-3]),
    ImageStat.SIZE: lambda x: np.uint32(np.prod(x.shape[-2:])),
    ImageStat.ASPECT_RATIO: lambda x: np.float16(x.shape[-1] / x.shape[-2]),
    ImageStat.DEPTH: lambda x: np.uint8(get_bitdepth(x).depth),
    ImageStat.BRIGHTNESS: lambda x: np.float16(np.mean(x)),
    ImageStat.BLURRINESS: lambda x: np.float16(np.std(edge_filter(np.mean(x, axis=0)))),
    ImageStat.MISSING: lambda x: np.float16(np.sum(np.isnan(x)) / np.prod(x.shape[-2:])),
    ImageStat.ZERO: lambda x: np.float16(np.count_nonzero(x == 0) / np.prod(x.shape[-2:])),
    ImageStat.MEAN: lambda x: np.float16(np.mean(x)),
    ImageStat.STD: lambda x: np.float16(np.std(x)),
    ImageStat.VAR: lambda x: np.float16(np.var(x)),
    ImageStat.SKEW: lambda x: np.float16(skew(x.ravel())),
    ImageStat.KURTOSIS: lambda x: np.float16(kurtosis(x.ravel())),
    ImageStat.PERCENTILES: lambda x: np.float16(np.percentile(x, q=QUARTILES)),
    ImageStat.HISTOGRAM: lambda x: np.uint32(np.histogram(x, 256, (0, 1))[0]),
    ImageStat.ENTROPY: lambda x: np.float16(entropy(x)),
}

CHANNELSTATS_FN_MAP: dict[ImageStat, Callable[[NDArray], Any]] = {
    ImageStat.MEAN: lambda x: np.float16(np.mean(x, axis=1)),
    ImageStat.STD: lambda x: np.float16(np.std(x, axis=1)),
    ImageStat.VAR: lambda x: np.float16(np.var(x, axis=1)),
    ImageStat.SKEW: lambda x: np.float16(skew(x, axis=1)),
    ImageStat.KURTOSIS: lambda x: np.float16(kurtosis(x, axis=1)),
    ImageStat.PERCENTILES: lambda x: np.float16(np.percentile(x, q=QUARTILES, axis=1).T),
    ImageStat.HISTOGRAM: lambda x: np.uint32(np.apply_along_axis(lambda y: np.histogram(y, 256, (0, 1))[0], 1, x)),
    ImageStat.ENTROPY: lambda x: np.float16(entropy(x, axis=1)),
}


def run_stats(
    images: Iterable[ArrayLike],
    flags: ImageStat,
    fn_map: dict[ImageStat, Callable[[NDArray], Any]],
    flatten: bool,
):
    """
    Compute specified statistics on a set of images.

    This function applies a set of statistical operations to each image in the input iterable,
    based on the specified flags. The function dynamically determines which statistics to apply
    using a flag system and a corresponding function map. It also supports optional image
    flattening for pixel-wise calculations.

    Parameters
    ----------
    images : ArrayLike
        An iterable of images (e.g., list of arrays), where each image is represented as an
        array-like structure (e.g., NumPy arrays).
    flags : ImageStat
        A bitwise flag or set of flags specifying the statistics to compute for each image.
        These flags determine which functions in `fn_map` to apply.
    fn_map : dict[ImageStat, Callable]
        A dictionary mapping `ImageStat` flags to functions that compute the corresponding statistics.
        Each function accepts a NumPy array (representing an image or rescaled pixel data) and returns a result.
    flatten : bool
        If True, the image is flattened into a 2D array for pixel-wise operations. Otherwise, the
        original image dimensions are preserved.

    Returns
    -------
    list[dict[str, NDArray]]
        A list of dictionaries, where each dictionary contains the computed statistics for an image.
        The dictionary keys correspond to the names of the statistics, and the values are NumPy arrays
        with the results of the computations.

    Raises
    ------
    ValueError
        If unsupported flags are provided that are not present in `fn_map`.

    Notes
    -----
    - The function performs image normalization (rescaling the image values)
      before applying some of the statistics.
    - Pixel-level statistics (e.g., brightness, entropy) are computed after
      rescaling and, optionally, flattening the images.
    - For statistics like histograms and entropy, intermediate results may
      be reused to avoid redundant computation.
    """
    verify_supported(flags, fn_map)
    flag_dict = to_distinct(flags)

    results_list: list[dict[str, NDArray]] = []
    for image in to_numpy_iter(images):
        normalized = normalize_image_shape(image)
        scaled = None
        hist = None
        output: dict[str, NDArray] = {}
        for flag, stat in flag_dict.items():
            if flag & (ImageStat.ALL_PIXELSTATS | ImageStat.BRIGHTNESS):
                if scaled is None:
                    scaled = rescale(normalized).reshape(image.shape[0], -1) if flatten else rescale(normalized)
                if flag & (ImageStat.HISTOGRAM | ImageStat.ENTROPY):
                    if hist is None:
                        hist = fn_map[ImageStat.HISTOGRAM](scaled)
                    output[stat] = hist if flag & ImageStat.HISTOGRAM else fn_map[flag](hist)
                else:
                    output[stat] = fn_map[flag](scaled)
            else:
                output[stat] = fn_map[flag](normalized)
        results_list.append(output)
    return results_list


@set_metadata("dataeval.metrics")
def imagestats(images: Iterable[ArrayLike], flags: ImageStat = ImageStat.ALL_STATS) -> StatsOutput:
    """
    Calculates image and pixel statistics for each image

    This function computes various statistical metrics (e.g., mean, standard deviation, entropy)
    on the images as a whole, based on the specified flags. It supports multiple types of statistics
    that can be selected using the `flags` argument.

    Parameters
    ----------
    images : ArrayLike
        Images to run statistical tests on
    flags : ImageStat, default ImageStat.ALL_STATS
        Metric(s) to calculate for each image. The default flag ``ImageStat.ALL_STATS``
        computes all available statistics.

    Returns
    -------
    StatsOutput
        A dictionary-like object containing the computed statistics for each image. The keys correspond
        to the names of the statistics (e.g., 'mean', 'std'), and the values are lists of results for
        each image or numpy arrays when the results are multi-dimensional.

    Notes
    -----
    - All metrics in the ImageStat.ALL_PIXELSTATS flag are scaled based on the perceived bit depth
      (which is derived from the largest pixel value) to allow for better comparison
      between images stored in different formats and different resolutions.
    - ImageStat.ZERO and ImageStat.MISSING are presented as a percentage of total pixel counts

    Examples
    --------
    Calculating the statistics on the images, whose shape is (C, H, W)

    >>> results = imagestats(images, flags=ImageStat.MEAN | ImageStat.ALL_VISUALS)
    >>> print(results.mean)
    [0.16650391 0.52050781 0.05471802 0.07702637 0.09875488 0.12188721
     0.14440918 0.16711426 0.18859863 0.21264648 0.2355957  0.25854492
     0.27978516 0.3046875  0.32788086 0.35131836 0.37255859 0.39819336
     0.42163086 0.4453125  0.46630859 0.49267578 0.51660156 0.54052734
     0.56152344 0.58837891 0.61230469 0.63671875 0.65771484 0.68505859
     0.70947266 0.73388672 0.75488281 0.78271484 0.80712891 0.83203125
     0.85302734 0.88134766 0.90625    0.93115234]
    >>> print(results.zero)
    [0.12561035 0.         0.         0.         0.11730957 0.
     0.         0.         0.10986328 0.         0.         0.
     0.10266113 0.         0.         0.         0.09570312 0.
     0.         0.         0.08898926 0.         0.         0.
     0.08251953 0.         0.         0.         0.07629395 0.
     0.         0.         0.0703125  0.         0.         0.
     0.0645752  0.         0.         0.        ]
    """
    stats = run_stats(images, flags, IMAGESTATS_FN_MAP, False)
    output = {}
    length = len(stats)
    for i, results in enumerate(stats):
        for stat, result in results.items():
            if not isinstance(result, (np.ndarray, np.generic)):
                output.setdefault(stat, []).append(result)
            else:
                shape = () if np.isscalar(result) else result.shape
                output.setdefault(stat, np.empty((length,) + shape))[i] = result
    return StatsOutput(**populate_defaults(output, StatsOutput))


@set_metadata("dataeval.metrics")
def channelstats(images: Iterable[ArrayLike], flags=ImageStat.ALL_PIXELSTATS) -> StatsOutput:
    """
    Calculates pixel statistics for each image per channel

    This function computes pixel-level statistics (e.g., mean, variance, etc.) on a per-channel basis
    for each image. The statistics can be selected using the `flags` argument, and the results will
    be grouped by the number of channels (e.g., RGB channels) in each image.

    Parameters
    ----------
    images : ArrayLike
        Images to run statistical tests on
    flags: ImageStat, default ImageStat.ALL_PIXELSTATS
        Metric(s) to calculate for each image per channel.
        Only flags within the ``ImageStat.ALL_PIXELSTATS`` category are supported.

    Returns
    -------
    StatsOutput
        A dictionary-like object containing the computed statistics for each image per channel. The keys
        correspond to the names of the statistics (e.g., 'mean', 'variance'), and the values are numpy arrays
        with results for each channel of each image.

    Notes
    -----
    - All metrics in the ImageStat.ALL_PIXELSTATS flag are scaled based on the perceived bit depth
      (which is derived from the largest pixel value) to allow for better comparison
      between images stored in different formats and different resolutions.

    Examples
    --------
    Calculating the statistics on a per channel basis for images, whose shape is (N, C, H, W)

    >>> results = channelstats(images, flags=ImageStat.MEAN | ImageStat.VAR)
    >>> print(results.mean)
    {3: array([[0.01617, 0.5303 , 0.06525, 0.09735, 0.1295 , 0.1616 , 0.1937 ,
            0.2258 , 0.2578 , 0.29   , 0.322  , 0.3542 , 0.3865 , 0.4185 ,
            0.4507 , 0.4827 , 0.5146 , 0.547  , 0.579  , 0.6113 , 0.643  ,
            0.6753 , 0.7075 , 0.7397 , 0.7715 , 0.8037 , 0.836  , 0.868  ,
            0.9004 , 0.932  ],
           [0.04828, 0.562  , 0.06726, 0.09937, 0.1315 , 0.1636 , 0.1957 ,
            0.2278 , 0.26   , 0.292  , 0.3242 , 0.3562 , 0.3884 , 0.4204 ,
            0.4526 , 0.4846 , 0.5166 , 0.549  , 0.581  , 0.6133 , 0.6455 ,
            0.6772 , 0.7095 , 0.7417 , 0.774  , 0.8057 , 0.838  , 0.87   ,
            0.9023 , 0.934  ],
           [0.0804 , 0.594  , 0.0693 , 0.1014 , 0.1334 , 0.1656 , 0.1978 ,
            0.2299 , 0.262  , 0.294  , 0.3262 , 0.3584 , 0.3904 , 0.4226 ,
            0.4546 , 0.4868 , 0.519  , 0.551  , 0.583  , 0.615  , 0.6475 ,
            0.679  , 0.7114 , 0.7437 , 0.776  , 0.808  , 0.84   , 0.872  ,
            0.9043 , 0.9365 ]], dtype=float16)}
    >>> print(results.var)
    {3: array([[0.00010103, 0.01077   , 0.0001621 , 0.0003605 , 0.0006375 ,
            0.000993  , 0.001427  , 0.001939  , 0.00253   , 0.003199  ,
            0.003944  , 0.004772  , 0.005676  , 0.006657  , 0.007717  ,
            0.00886   , 0.01008   , 0.01137   , 0.01275   , 0.0142    ,
            0.01573   , 0.01733   , 0.01903   , 0.0208    , 0.02264   ,
            0.02457   , 0.02657   , 0.02864   , 0.0308    , 0.03305   ],
           [0.0001798 , 0.0121    , 0.0001721 , 0.0003753 , 0.0006566 ,
            0.001017  , 0.001455  , 0.001972  , 0.002565  , 0.003239  ,
            0.00399   , 0.00482   , 0.00573   , 0.006714  , 0.007782  ,
            0.00893   , 0.01015   , 0.011444  , 0.012825  , 0.01428   ,
            0.01581   , 0.01743   , 0.01912   , 0.02089   , 0.02274   ,
            0.02466   , 0.02667   , 0.02875   , 0.03091   , 0.03314   ],
           [0.000337  , 0.0135    , 0.0001824 , 0.0003903 , 0.0006766 ,
            0.00104   , 0.001484  , 0.002005  , 0.002604  , 0.00328   ,
            0.004036  , 0.00487   , 0.005783  , 0.006775  , 0.00784   ,
            0.00899   , 0.010216  , 0.01152   , 0.0129    , 0.01436   ,
            0.0159    , 0.01752   , 0.01921   , 0.02098   , 0.02283   ,
            0.02477   , 0.02676   , 0.02885   , 0.03102   , 0.03326   ]],
          dtype=float16)}
    """
    stats = run_stats(images, flags, CHANNELSTATS_FN_MAP, True)

    output = {}
    for i, results in enumerate(stats):
        for stat, result in results.items():
            channels = result.shape[0]
            output.setdefault(stat, {}).setdefault(channels, []).append(result)
            output.setdefault(CH_IDX_MAP, {}).setdefault(channels, {})[i] = None

    # Concatenate list of channel statistics numpy
    for stat in output:
        if stat == CH_IDX_MAP:
            continue
        for channel in output[stat]:
            output[stat][channel] = np.array(output[stat][channel]).T

    for channel in output[CH_IDX_MAP]:
        output[CH_IDX_MAP][channel] = list(output[CH_IDX_MAP][channel].keys())

    return StatsOutput(**populate_defaults(output, StatsOutput))
