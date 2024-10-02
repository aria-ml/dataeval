from dataclasses import dataclass
from enum import IntFlag, auto
from functools import reduce
from typing import Any, Callable, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import entropy, kurtosis, skew

from dataeval._internal.flags import ImageStat, to_distinct, verify_supported
from dataeval._internal.interop import to_numpy_iter
from dataeval._internal.metrics.stats.stats import StatsOutput
from dataeval._internal.metrics.utils import edge_filter, normalize_image_shape, rescale
from dataeval._internal.output import OutputMetadata, populate_defaults, set_metadata


class BoxStat(IntFlag):
    """
    Flags for calculating bounding box statistics
    """

    # PROPERTIES
    WIDTH = auto()
    HEIGHT = auto()
    SIZE = auto()
    ASPECT_RATIO = auto()

    # VISUALS
    BRIGHTNESS = auto()
    BLURRINESS = auto()
    CONTRAST = auto()
    DARKNESS = auto()
    MISSING = auto()
    ZEROS = auto()

    # PIXEL STATS
    MEAN = auto()
    STD = auto()
    VAR = auto()
    SKEW = auto()
    KURTOSIS = auto()
    ENTROPY = auto()
    PERCENTILES = auto()
    HISTOGRAM = auto()

    # BOX STATS
    BOX_COUNT = auto()
    CENTER = auto()

    # JOINT FLAGS
    ALL_PROPERTIES = WIDTH | HEIGHT | SIZE | ASPECT_RATIO
    ALL_VISUALS = BRIGHTNESS | BLURRINESS | CONTRAST | DARKNESS | MISSING | ZEROS
    ALL_PIXELSTATS = MEAN | STD | VAR | SKEW | KURTOSIS | ENTROPY | PERCENTILES | HISTOGRAM
    ALL_BOXSTATS = BOX_COUNT | CENTER | ALL_PROPERTIES | ALL_VISUALS | ALL_PIXELSTATS


def translate_comparison_flag_to_result(stat: str):
    # Remove 'box_' prefix and '_ratio' suffix from stat
    # except for aspect ratio
    key = stat.replace("box_", "").replace("_ratio", "")
    if key == "aspect":
        key = "aspect_ratio"
    return key


def names_to_flags(flags, supported_flags, image=False) -> ImageStat | BoxStat:
    verify_supported(flags, supported_flags)
    flag_dict = to_distinct(flags)
    selected_flags = []
    for flag, stat in flag_dict.items():
        key = translate_comparison_flag_to_result(stat)
        if image:
            selected_flags.extend([f for f in ImageStat.ALL_STATS if f.name.lower() == key])  # type: ignore
        else:
            selected_flags.extend([f for f in BoxStat.ALL_STATS if f.name.lower() == key])  # type: ignore
    corresponding_flags = reduce(lambda a, b: a | b, selected_flags)

    return corresponding_flags


def normalize_box_shape(bounding_box: NDArray) -> NDArray:
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


@dataclass(frozen=True)
class BaseStatsOutput(OutputMetadata):
    def dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_") and len(v) > 0}

    def __len__(self) -> int:
        for a in self.__annotations__:
            attr = getattr(self, a, None)
            if attr is not None and hasattr(a, "__len__") and len(attr) > 0:
                return len(attr)
        return 0


@dataclass(frozen=True)
class BoxStatsOutput(BaseStatsOutput):
    """
    Attributes
    ----------
    box_count : NDArray[np.uint16]
        Total number of bounding boxes per image
    center : NDArray[np.uint16]
        Box center for each bounding box
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

    box_count: NDArray[np.uint16]
    center: NDArray[np.uint16]
    width: NDArray[np.uint16]
    height: NDArray[np.uint16]
    channels: NDArray[np.uint8]
    size: NDArray[np.uint32]
    aspect_ratio: NDArray[np.float16]
    depth: NDArray[np.uint8]
    brightness: NDArray[np.float16]
    blurriness: NDArray[np.float16]
    contrast: NDArray[np.float16]
    darkness: NDArray[np.float16]
    missing: NDArray[np.float16]
    zeros: NDArray[np.float16]
    mean: NDArray[np.float16]
    std: NDArray[np.float16]
    var: NDArray[np.float16]
    skew: NDArray[np.float16]
    kurtosis: NDArray[np.float16]
    percentiles: NDArray[np.float16]
    histogram: NDArray[np.uint32]
    entropy: NDArray[np.float16]

@dataclass(frozen=True)
class RatioStatsOutput(BaseStatsOutput):
    """
    Attributes
    ----------
    density : NDArray[np.float16]
        Total box pixels divided by the image size
    count_density : NDArray[np.float16]
        Box count divided by density
    center_ratio : NDArray[np.float16]
        Ratio of box center to image center for each box, to the left and below image center is negative
    width_ratio : NDArray[np.float16]
        Ratio of box width to image width for each box
    height_ratio : NDArray[np.float16]
        Ratio of box height to image height for each box
    size_ratio : NDArray[np.float16]
        Ratio of box size to image size for each box
    aspect_ratio_ratio : NDArray[np.float16]
        Ratio of box aspect ratio to image aspect ratio for each box
    brightness_ratio : NDArray[np.float16]
        Ratio of box brightness to image brightness for each box
    blurriness_ratio : NDArray[np.float16]
        Ratio of box blurriness to image blurriness for each box
    contrast_ratio : NDArray[np.float16]
        Ratio of box contrast to image contrast for each box
    darkness_ratio : NDArray[np.float16]
        Ratio of box darkness to image darkness for each box
    zeros_ratio : NDArray[np.float16]
        Ratio of box zeros to image zeros for each box
    mean_ratio : NDArray[np.float16]
        Ratio of box mean to image mean for each box
    std_ratio : NDArray[np.float16]
        Ratio of box standard deviation to image standard deviation for each box
    var_ratio : NDArray[np.float16]
        Ratio of box variance to image variance for each box
    skew_ratio : NDArray[np.float16]
        Ratio of box skew to image skew for each box
    kurtosis_ratio : NDArray[np.float16]
        Ratio of box kurtosis to image kurtosis for each box
    entropy_ratio : NDArray[np.float16]
        Ratio of box entropy to image entropy for each box
    """

    density: NDArray[np.float16]
    count_density: NDArray[np.float16]
    center_ratio: NDArray[np.float16]
    width_ratio: NDArray[np.float16]
    height_ratio: NDArray[np.float16]
    size_ratio: NDArray[np.float16]
    aspect_ratio_ratio: NDArray[np.float16]
    brightness_ratio: NDArray[np.float16]
    blurriness_ratio: NDArray[np.float16]
    contrast_ratio: NDArray[np.float16]
    darkness_ratio: NDArray[np.float16]
    zeros_ratio: NDArray[np.float16]
    mean_ratio: NDArray[np.float16]
    std_ratio: NDArray[np.float16]
    var_ratio: NDArray[np.float16]
    skew_ratio: NDArray[np.float16]
    kurtosis_ratio: NDArray[np.float16]
    entropy_ratio: NDArray[np.float16]


QUARTILES = (0, 25, 50, 75, 100)

BOXSTATS_FN_MAP: dict[BoxStat, Callable[[NDArray], Any]] = {
    BoxStat.BOX_COUNT: lambda x: np.uint16(x.shape[0]),
    BoxStat.CENTER: lambda x: np.asarray([(x[0] + x[2]) / 2, (x[1] + x[3]) / 2], dtype=np.uint16),
    BoxStat.WIDTH: lambda x: np.uint16(x.shape[-1]),
    BoxStat.HEIGHT: lambda x: np.uint16(x.shape[-2]),
    BoxStat.SIZE: lambda x: np.uint32(np.prod(x.shape[-2:])),
    BoxStat.ASPECT_RATIO: lambda x: np.float16(x.shape[-1] / x.shape[-2]),
    BoxStat.BRIGHTNESS: lambda x: np.float16((np.max(x) - np.mean(x)) / np.var(x)),
    BoxStat.BLURRINESS: lambda x: np.float16(np.std(edge_filter(np.mean(x, axis=0)))),
    BoxStat.CONTRAST: lambda x: np.float16((np.max(x) - np.min(x)) / np.mean(x)),
    BoxStat.DARKNESS: lambda x: np.float16((np.mean(x) - np.min(x)) / np.var(x)),
    BoxStat.MISSING: lambda x: np.float16(np.sum(np.isnan(x)) / np.prod(x.shape[-2:])),
    BoxStat.ZEROS: lambda x: np.float16(np.count_nonzero(x == 0) / np.prod(x.shape[-2:])),
    BoxStat.MEAN: lambda x: np.float16(np.mean(x)),
    BoxStat.STD: lambda x: np.float16(np.std(x)),
    BoxStat.VAR: lambda x: np.float16(np.var(x)),
    BoxStat.SKEW: lambda x: np.float16(skew(x.ravel())),
    BoxStat.KURTOSIS: lambda x: np.float16(kurtosis(x.ravel())),
    BoxStat.PERCENTILES: lambda x: np.float16(np.nanpercentile(x, q=QUARTILES)),
    BoxStat.HISTOGRAM: lambda x: np.uint32(np.histogram(x, 256, (0, 1))[0]),
    BoxStat.ENTROPY: lambda x: np.float16(entropy(x)),
}


def run_boxstats(
    bounding_boxes: Iterable[ArrayLike],
    images: Iterable[ArrayLike],
    flags: BoxStat,
    fn_map: dict[BoxStat, Callable[[NDArray], Any]],
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
    bounding_boxes : ArrayLike or None
        An iterable of bounding boxes with each box in the format of (X, Y, W, H).
        X, Y is the upperleft corner of the box, with W being the width and H being the height of the box.
        If more than one box is present in an image, then the outermost dimension of the
        ArrayLike must match the outermost dimension of the images array.

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
    max_boxes = 0
    for boxes, image in zip(to_numpy_iter(bounding_boxes), to_numpy_iter(images)):
        norm_boxes = normalize_box_shape(boxes)
        normalized = normalize_image_shape(image)
        max_boxes = max(max_boxes, norm_boxes.shape[0])
        scaled = None
        hist = None
        output: dict[str, NDArray] = {}
        for flag, stat in flag_dict.items():
            if flag & BoxStat.BOX_COUNT:
                output[stat] = np.asarray([fn_map[flag](norm_boxes)])
            else:
                boxes_output = []
                for box in norm_boxes:
                    box_image = normalized[:, box[1] : box[1] + box[3], box[0] : box[0] + box[2]]
                    if flag & (BoxStat.ALL_PIXELSTATS | BoxStat.BRIGHTNESS):
                        if scaled is None:
                            scaled = rescale(box_image)
                        if flag & (BoxStat.HISTOGRAM | BoxStat.ENTROPY):
                            if hist is None:
                                hist = fn_map[BoxStat.HISTOGRAM](scaled)
                            boxes_output.append(hist if flag & BoxStat.HISTOGRAM else fn_map[flag](hist))
                        else:
                            boxes_output.append(fn_map[flag](scaled))
                    elif flag & BoxStat.CENTER:
                        boxes_output.append(fn_map[flag](box))
                    else:
                        boxes_output.append(fn_map[flag](box_image))
                output[stat] = np.asarray(boxes_output)
        results_list.append(output)
    return results_list, max_boxes


@set_metadata("dataeval.metrics")
def boxstats(
    bounding_boxes: Iterable[ArrayLike],
    images: Iterable[ArrayLike],
    flags=BoxStat.ALL_BOXSTATS,
) -> BoxStatsOutput:
    """
    Calculates box and pixel statistics for each bounding box

    This function computes various statistical metrics (e.g., mean, standard deviation, entropy)
    on the bounding boxes, based on the specified flags. It supports multiple types of statistics
    that can be selected using the `flags` argument.

    Parameters
    ----------
    bounding boxes : ArrayLike, shape - [[(X, Y, W, H)]] or (N,M,4) or (N,4)
        Lists of lists or numpy array with individual bounding boxes in the format of (X, Y, W, H)
        where (X, Y) is the top left corner of the bounding box.
        A set of lists where each list contains all bounding boxes per image. If a numpy array,
        N is the number of images, M is the number of boxes per image.
    images : ArrayLike
        Images to compute the bounding box statistical tests on
    flags : BoxStat, default BoxStat.ALL_BOXSTATS
        Metric(s) to calculate for each bounding box. The default flag ``BoxStat.ALL_BOXSTATS``
        computes all available statistics for each box.

    Returns
    -------
    StatsOutput
        A dictionary-like object containing the computed statistics for each image. The keys correspond
        to the names of the statistics (e.g., 'mean', 'std'), and the values are lists of results for
        each image or numpy arrays when the results are multi-dimensional.

    Notes
    -----
    - All metrics in the BoxStat.ALL_PIXELSTATS flag are scaled based on the perceived bit depth
      (which is derived from the largest pixel value) to allow for better comparison
      between images stored in different formats and different resolutions.
    - BoxStat.ZERO and BoxStat.MISSING are presented as a percentage of total pixel counts

    Examples
    --------
    Calculating the statistics on the bounding boxes (X, Y, W, H) for a set of images with shape (C, H, W)

    >>> results = boxstats(bounding_boxes, images)
    >>> print(results.size)
    >>> print(results.darkness)
    """
    stats, max_boxes = run_boxstats(bounding_boxes, images, flags, BOXSTATS_FN_MAP)
    output = {}
    length = len(stats)
    for i, results in enumerate(stats):
        for stat, result in results.items():
            if stat == "box_count":
                output.setdefault(stat, np.zeros((length), dtype=result.dtype))
                output[stat][i] = result
            else:
                if np.issubdtype(result.dtype, np.floating):
                    output.setdefault(stat, np.full((length, max_boxes) + result.shape[1:], np.nan, dtype=result.dtype))
                else:
                    output.setdefault(stat, np.zeros((length, max_boxes) + result.shape[1:], dtype=result.dtype))
                nboxes = result.shape[0]
                output[stat][i, :nboxes] = result
    return BoxStatsOutput(**populate_defaults(output, BoxStatsOutput))


def compare_stats(box_stats: dict, image_stats: dict):
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
        A dictionary mapping `ImageStat` flags to functions that compute the corresponding comparison statistics.
        Each function accepts 2 NumPy arrays (representing a bounding box and an image or rescaled pixel data)
        and returns the resulting ratio.
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
    flag_dict = to_distinct(BoxStat.ALL_BOXSTATS)

    output: dict[str, NDArray] = {}
    for _, stat in flag_dict.items():
        result = stat + "_ratio"
        if stat == 'size':
            if stat in box_stats and stat in image_stats:
                if not np.all(image_stats[stat]):
                    image_divide = image_stats[stat]
                    mask = image_divide == 0
                    image_divide[mask] += 0.001  # TODO: test with real data and see what this value does
                    output[result] = (box_stats[stat] / image_divide[:, np.newaxis]).astype(np.float16)
                else:
                    output[result] = (box_stats[stat] / image_stats[stat][:, np.newaxis]).astype(np.float16)
                output['density'] = (np.sum(box_stats[stat], axis=1) / image_stats[stat]).astype(np.float16)
        elif stat == 'box_count':
            if stat in box_stats and 'size' in box_stats and 'size' in image_stats:
                output['count_density'] = (box_stats[stat] / (np.sum(box_stats["size"], axis=1) / image_stats["size"])).astype(np.float16)
        elif stat == 'center':
            if stat in box_stats and 'width' in image_stats and 'height' in image_stats:
                width_ratio = (box_stats[stat][:, :, 0] - image_stats["width"][:, np.newaxis] / 2) / (
                    image_stats["width"][:, np.newaxis] / 2
                )
                height_ratio = (image_stats["height"][:, np.newaxis] / 2 - box_stats[stat][:, :, 1]) / (
                    image_stats["height"][:, np.newaxis] / 2
                )
                output[result] = np.stack([width_ratio, height_ratio], axis=2, dtype=np.float16)
        elif stat != 'histogram' and stat != 'percentiles':
            if stat in box_stats and stat in image_stats:
                if not np.all(image_stats[stat]):
                    image_divide = image_stats[stat]
                    mask = image_divide == 0
                    image_divide[mask] += 0.001  # TODO: test with real data and see what this value does
                    output[result] = (box_stats[stat] / image_divide[:, np.newaxis]).astype(np.float16)
                else:
                    output[result] = (box_stats[stat] / image_stats[stat][:, np.newaxis]).astype(np.float16)
    return output


@set_metadata("dataeval.metrics")
def box_image_ratio_stats(
    bounding_boxes: BoxStatsOutput,
    images: StatsOutput,
) -> RatioStatsOutput:
    """
    Calculates ratios between the box and image statistics

    This function creates ratios between the calculated box and image statistical metrics
    (e.g., mean, standard deviation, entropy), based on the specified flags.
    It supports multiple types of metrics that can be selected using the `flags` argument.

    Parameters
    ----------
    bounding_boxes : StatsBoxOutput | ArrayLike
        Output from ``boxstats`` or data in the form of a lists of lists or a numpy array
        with individual bounding boxes in the format of (X, Y, W, H) where (X, Y) is
        the top left corner of the bounding box. If data is provided, ``boxstats``
        will be run on the data.
    images : StatsOutput | ArrayLike
        Output from ``imagestats`` or images to compute the image and bounding box
        statistical metrics with. Images must be provided if data was provided to the
        bounding_boxes parameter. If images are provided, ``imagestats`` will be run on the images.
    flags : ImageStat, default BoxStat.ALL_RATIO
        Metric(s) to calculate ratios for between the bouding boxes and the images.
        The default flag ``BoxStat.ALL_RATIO`` compares statistics from the Box, Pixelstats,
        Properties, and Visuals flags.
        Metric(s) beginning with "BOX" are calculated per box while the rest are calculated
        using all boxes. If data and images are provided, the flags for running the ``boxstats``
        and ``imagestats`` functions, will be determined based on the metrics needed to calculate
        the comparison metrics.

    Returns
    -------
    StatsRatioOutput | (StatsRatioOutput, StatsBoxOutput, StatsOutput)
        A dictionary-like object containing the computed ratios for each statistic. The keys correspond
        to the names of the statistics (e.g., 'mean_ratio', 'std_ratio', 'density'), and the
        values are lists of results for each image or numpy arrays when the results are multi-dimensional.
        If data and images are provided, the resulting ``StatsOutputs`` from calling the ``boxstats``
        and ``imagestats`` functions will also be provided.

    Examples
    --------
    Calculating comparisons between the bounding box statistics and the image statistics

    >>> results = box_image_comparisonstats(bounding_boxes, images)
    >>> print(results.density)
    >>> print(results.density)
    """
    box_output = bounding_boxes.dict()
    image_output = images.dict()

    stats = compare_stats(box_output, image_output)
    return RatioStatsOutput(**populate_defaults(stats, RatioStatsOutput))
