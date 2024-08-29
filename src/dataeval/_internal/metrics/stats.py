from enum import Flag
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, TypeVar, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import entropy, kurtosis, skew

from dataeval._internal.flags import ImageHash, ImageProperty, ImageStatistics, ImageStatsFlags, ImageVisuals
from dataeval._internal.interop import to_numpy_iter
from dataeval._internal.metrics.utils import edge_filter, get_bitdepth, normalize_image_shape, pchash, rescale, xxhash

QUARTILES = (0, 25, 50, 75, 100)

TFlag = TypeVar("TFlag", bound=Flag)

IMAGESTATS_FN_MAP: Dict[Flag, Callable[[np.ndarray], Any]] = {
    ImageHash.XXHASH: lambda x: xxhash(x),
    ImageHash.PCHASH: lambda x: pchash(x),
    ImageProperty.WIDTH: lambda x: np.int32(x.shape[-1]),
    ImageProperty.HEIGHT: lambda x: np.int32(x.shape[-2]),
    ImageProperty.SIZE: lambda x: np.int32(x.shape[-1] * x.shape[-2]),
    ImageProperty.ASPECT_RATIO: lambda x: x.shape[-1] / np.int32(x.shape[-2]),
    ImageProperty.CHANNELS: lambda x: x.shape[-3],
    ImageProperty.DEPTH: lambda x: get_bitdepth(x).depth,
    ImageVisuals.BRIGHTNESS: lambda x: np.mean(rescale(x)),
    ImageVisuals.BLURRINESS: lambda x: np.std(edge_filter(np.mean(x, axis=0))),
    ImageVisuals.MISSING: lambda x: np.sum(np.isnan(x)),
    ImageVisuals.ZERO: lambda x: np.int32(np.count_nonzero(x == 0)),
    ImageStatistics.MEAN: lambda x: np.mean(x),
    ImageStatistics.STD: lambda x: np.std(x),
    ImageStatistics.VAR: lambda x: np.var(x),
    ImageStatistics.SKEW: lambda x: np.float32(skew(x.ravel())),
    ImageStatistics.KURTOSIS: lambda x: np.float32(kurtosis(x.ravel())),
    ImageStatistics.PERCENTILES: lambda x: np.percentile(x, q=QUARTILES),
    ImageStatistics.HISTOGRAM: lambda x: np.histogram(x, 256, (0, 1))[0],
    ImageStatistics.ENTROPY: lambda x: np.float32(entropy(x)),
}

CHANNELSTATS_FN_MAP: Dict[Flag, Callable[[np.ndarray], Any]] = {
    ImageStatistics.MEAN: lambda x: np.mean(x, axis=1),
    ImageStatistics.STD: lambda x: np.std(x, axis=1),
    ImageStatistics.VAR: lambda x: np.var(x, axis=1),
    ImageStatistics.SKEW: lambda x: skew(x, axis=1),
    ImageStatistics.KURTOSIS: lambda x: kurtosis(x, axis=1),
    ImageStatistics.PERCENTILES: lambda x: np.percentile(x, q=QUARTILES, axis=1).T,
    ImageStatistics.HISTOGRAM: lambda x: np.apply_along_axis(lambda y: np.histogram(y, 256, (0, 1))[0], 1, x),
    ImageStatistics.ENTROPY: lambda x: entropy(x, axis=1),
}

IMAGESTATS_ALL_FLAGS = [ImageHash.ALL, ImageProperty.ALL, ImageStatistics.ALL, ImageVisuals.ALL]
CHANNELSTATS_ALL_FLAGS = [ImageStatistics.ALL]


def run_stats(
    images: Iterable[ArrayLike],
    flags: Union[ImageStatsFlags, Sequence[ImageStatsFlags]],
    fn_map: Dict[Flag, Callable[[np.ndarray], Any]],
    flatten: bool,
):
    flags = [f for flag in (flags if isinstance(flags, Sequence) else [flags]) for f in flag]
    results_list: List[Dict[str, np.ndarray]] = []
    for image in to_numpy_iter(images):
        normalized = normalize_image_shape(image)
        scaled = None
        hist = None
        output: Dict[str, np.ndarray] = {}
        for flag in flags:
            if flag.name is None:
                raise TypeError(f"Unrecognized stat flag {flag}")
            stat = flag.name.lower()
            if isinstance(flag, ImageStatistics):
                if scaled is None:
                    scaled = rescale(normalized).reshape(image.shape[0], -1) if flatten else rescale(normalized)
                if flag & (ImageStatistics.HISTOGRAM | ImageStatistics.ENTROPY):
                    if hist is None:
                        hist = fn_map[ImageStatistics.HISTOGRAM](scaled)
                    output[stat] = hist if flag & ImageStatistics.HISTOGRAM else fn_map[flag](hist)
                else:
                    output[stat] = fn_map[flag](scaled)
            else:
                output[stat] = fn_map[flag](normalized)
        results_list.append(output)
    return results_list


def imagestats(
    images: Iterable[ArrayLike], flags: Optional[Union[ImageStatsFlags, Sequence[ImageStatsFlags]]] = None
) -> Dict[str, Any]:
    """
    Calculates various image property statistics

    Parameters
    ----------
    images : Iterable[ArrayLike]
        Images to run statistical tests on
    flags : [ImageHash | ImageProperty | ImageStatistics | ImageVisuals], default None
        Metric(s) to calculate for each image - calculates all metrics if None

    Returns
    -------
    Dict[str, Any]
    """

    stats = run_stats(images, flags or IMAGESTATS_ALL_FLAGS, IMAGESTATS_FN_MAP, False)
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


def channelstats(images: Iterable[ArrayLike], flags: Optional[ImageStatistics] = None) -> Dict[str, Any]:
    """
    Calculates various image statistics per channel

    Parameters
    ----------
    images : Iterable[ArrayLike]
        Images to run statistical tests on
    flags: ImageStatistics, default None
        Statistic(s) to calculate for each image per channel - calculates all metrics if None

    Returns
    -------
    Dict[str, Any]
    """
    IDX_MAP = "idx_map"
    stats = run_stats(images, flags or CHANNELSTATS_ALL_FLAGS, CHANNELSTATS_FN_MAP, True)

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
