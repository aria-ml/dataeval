from typing import Any, Callable, Dict, Iterable, List

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import entropy, kurtosis, skew

from dataeval._internal.flags import ImageStat, to_distinct, verify_supported
from dataeval._internal.interop import to_numpy_iter
from dataeval._internal.metrics.utils import edge_filter, get_bitdepth, normalize_image_shape, pchash, rescale, xxhash

QUARTILES = (0, 25, 50, 75, 100)

IMAGESTATS_FN_MAP: Dict[ImageStat, Callable[[np.ndarray], Any]] = {
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

CHANNELSTATS_FN_MAP: Dict[ImageStat, Callable[[np.ndarray], Any]] = {
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
    fn_map: Dict[ImageStat, Callable[[np.ndarray], Any]],
    flatten: bool,
):
    verify_supported(flags, fn_map)
    flag_dict = to_distinct(flags)

    results_list: List[Dict[str, np.ndarray]] = []
    for image in to_numpy_iter(images):
        normalized = normalize_image_shape(image)
        scaled = None
        hist = None
        output: Dict[str, np.ndarray] = {}
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


def imagestats(images: Iterable[ArrayLike], flags: ImageStat = ImageStat.ALL_STATS) -> Dict[str, Any]:
    """
    Calculates image and pixel statistics for each image

    Parameters
    ----------
    images : Iterable[ArrayLike]
        Images to run statistical tests on
    flags : ImageStat, default ImageStat.ALL_STATS
        Metric(s) to calculate for each image

    Returns
    -------
    Dict[str, Any]
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
    return output


def channelstats(images: Iterable[ArrayLike], flags=ImageStat.ALL_PIXELSTATS) -> Dict[str, Any]:
    """
    Calculates pixel statistics for each image per channel

    Parameters
    ----------
    images : Iterable[ArrayLike]
        Images to run statistical tests on
    flags: ImageStat, default ImageStat.ALL_PIXELSTATS
        Statistic(s) to calculate for each image per channel
        Only flags in the ImageStat.ALL_PIXELSTATS category are supported

    Returns
    -------
    Dict[str, Any]
    """
    IDX_MAP = "idx_map"
    stats = run_stats(images, flags, CHANNELSTATS_FN_MAP, True)

    output = {}
    for i, results in enumerate(stats):
        for stat, result in results.items():
            channels = result.shape[0]
            output.setdefault(IDX_MAP, {}).setdefault(channels, {})[i] = None
            output.setdefault(stat, {}).setdefault(channels, []).append(result)

    # Concatenate list of channel statistics numpy
    for stat in output:
        if stat == IDX_MAP:
            continue
        for channel in output[stat]:
            output[stat][channel] = np.array(output[stat][channel]).T

    for channel in output[IDX_MAP]:
        output[IDX_MAP][channel] = list(output[IDX_MAP][channel].keys())

    return output
