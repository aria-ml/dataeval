from enum import Flag
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, Sequence, TypeVar, Union

import numpy as np
from scipy.stats import entropy, kurtosis, skew

from dataeval._internal.flags import ImageHash, ImageProperty, ImageStatistics, ImageStatsFlags, ImageVisuals
from dataeval._internal.functional.hash import pchash, xxhash
from dataeval._internal.functional.utils import edge_filter, get_bitdepth, normalize_image_shape, rescale
from dataeval._internal.interop import ArrayLike, to_numpy_iter
from dataeval._internal.metrics.base import EvaluateMixin, MetricMixin

QUARTILES = (0, 25, 50, 75, 100)

TBatch = TypeVar("TBatch", bound=Sequence[ArrayLike])
TFlag = TypeVar("TFlag", bound=Flag)


class BaseStatsMetric(EvaluateMixin, MetricMixin, Generic[TBatch, TFlag]):
    def __init__(self, flags: TFlag):
        self.flags = flags
        self.results = []

    def update(self, images: TBatch) -> None:
        """
        Updates internal metric cache for later calculation

        Parameters
        ----------
        batch : Sequence
            Sequence of images to be processed
        """

    def compute(self) -> Dict[str, Any]:
        """
        Computes the specified measures on the cached values

        Returns
        -------
        Dict[str, Any]
            Dictionary results of the specified measures
        """
        return {stat: [result[stat] for result in self.results] for stat in self.results[0]}

    def reset(self) -> None:
        """
        Resets the internal metric cache
        """
        self.results = []

    def _map(self, func_map: Dict[Flag, Callable]) -> Dict[str, Any]:
        """Calculates the measures for each flag if it is selected."""
        results = {}
        for flag, func in func_map.items():
            if not flag.name:
                raise ValueError("Provided flag to set value does not have a name.")
            if flag & self.flags:
                results[flag.name.lower()] = func()
        return results

    def _keys(self) -> List[str]:
        """Returns the list of measures to be calculated."""
        flags = (
            self.flags
            if isinstance(self.flags, Iterable)  # py3.11
            else [flag for flag in list(self.flags.__class__) if flag & self.flags]
        )
        return [flag.name.lower() for flag in flags if flag.name is not None]

    def evaluate(self, images: TBatch) -> Dict[str, Any]:
        """Calculate metric results given a single batch of images"""
        if self.results:
            raise RuntimeError("Call reset before calling evaluate")

        self.update(images)
        results = self.compute()
        self.reset()
        return results


class ImageHashMetric(BaseStatsMetric):
    """
    Hashes images using the specified algorithms

    Parameters
    ----------
    flags : ImageHash
        Algorithm(s) to calculate a hash as hex digest
    """

    def __init__(self, flags: ImageHash = ImageHash.ALL):
        super().__init__(flags)

    def update(self, images: Iterable[ArrayLike]) -> None:
        for image in to_numpy_iter(images):
            results = self._map(
                {
                    ImageHash.XXHASH: lambda: xxhash(image),
                    ImageHash.PCHASH: lambda: pchash(image),
                }
            )
            self.results.append(results)


class ImagePropertyMetric(BaseStatsMetric):
    """
    Calculates specified image properties

    Parameters
    ----------
    flags: ImageProperty
        Property(ies) to calculate for each image
    """

    def __init__(self, flags: ImageProperty = ImageProperty.ALL):
        super().__init__(flags)

    def update(self, images: Iterable[ArrayLike]) -> None:
        for image in to_numpy_iter(images):
            results = self._map(
                {
                    ImageProperty.WIDTH: lambda: np.int32(image.shape[-1]),
                    ImageProperty.HEIGHT: lambda: np.int32(image.shape[-2]),
                    ImageProperty.SIZE: lambda: np.int32(image.shape[-1] * image.shape[-2]),
                    ImageProperty.ASPECT_RATIO: lambda: image.shape[-1] / np.int32(image.shape[-2]),
                    ImageProperty.CHANNELS: lambda: image.shape[-3],
                    ImageProperty.DEPTH: lambda: get_bitdepth(image).depth,
                }
            )
            self.results.append(results)


class ImageVisualsMetric(BaseStatsMetric):
    """
    Calculates specified visual image properties

    Parameters
    ----------
    flags: ImageVisuals
        Property(ies) to calculate for each image
    """

    def __init__(self, flags: ImageVisuals = ImageVisuals.ALL):
        super().__init__(flags)

    def update(self, images: Iterable[ArrayLike]) -> None:
        for image in to_numpy_iter(images):
            results = self._map(
                {
                    ImageVisuals.BRIGHTNESS: lambda: np.mean(rescale(image)),
                    ImageVisuals.BLURRINESS: lambda: np.std(edge_filter(np.mean(image, axis=0))),
                    ImageVisuals.MISSING: lambda: np.sum(np.isnan(image)),
                    ImageVisuals.ZERO: lambda: np.int32(np.count_nonzero(image == 0)),
                }
            )
            self.results.append(results)


class ImageStatisticsMetric(BaseStatsMetric):
    """
    Calculates descriptive statistics for each image

    Parameters
    ----------
    flags: ImageStatistics
        Statistic(s) to calculate for each image
    """

    def __init__(self, flags: ImageStatistics = ImageStatistics.ALL):
        super().__init__(flags)

    def update(self, images: Iterable[ArrayLike]) -> None:
        for image in to_numpy_iter(images):
            scaled = rescale(image)
            if (ImageStatistics.HISTOGRAM | ImageStatistics.ENTROPY) & self.flags:
                hist = np.histogram(scaled, bins=256, range=(0, 1))[0]

            results = self._map(
                {
                    ImageStatistics.MEAN: lambda: np.mean(scaled),
                    ImageStatistics.STD: lambda: np.std(scaled),
                    ImageStatistics.VAR: lambda: np.var(scaled),
                    ImageStatistics.SKEW: lambda: np.float32(skew(scaled.ravel())),
                    ImageStatistics.KURTOSIS: lambda: np.float32(kurtosis(scaled.ravel())),
                    ImageStatistics.PERCENTILES: lambda: np.percentile(scaled, q=QUARTILES),
                    ImageStatistics.HISTOGRAM: lambda: hist,
                    ImageStatistics.ENTROPY: lambda: np.float32(entropy(hist)),
                }
            )
            self.results.append(results)


class ChannelStatisticsMetric(BaseStatsMetric):
    """
    Calculates descriptive statistics for each image per channel

    Parameters
    ----------
    flags: ImageStatistics
        Statistic(s) to calculate for each image per channel
    """

    def __init__(self, flags: ImageStatistics = ImageStatistics.ALL):
        super().__init__(flags)

    def update(self, images: Iterable[ArrayLike]) -> None:
        for image in to_numpy_iter(images):
            scaled = rescale(image)
            flattened = scaled.reshape(image.shape[0], -1)

            if (ImageStatistics.HISTOGRAM | ImageStatistics.ENTROPY) & self.flags:
                hist = np.apply_along_axis(lambda x: np.histogram(x, bins=256, range=(0, 1))[0], 1, flattened)

            results = self._map(
                {
                    ImageStatistics.MEAN: lambda: np.mean(flattened, axis=1),
                    ImageStatistics.STD: lambda: np.std(flattened, axis=1),
                    ImageStatistics.VAR: lambda: np.var(flattened, axis=1),
                    ImageStatistics.SKEW: lambda: skew(flattened, axis=1),
                    ImageStatistics.KURTOSIS: lambda: kurtosis(flattened, axis=1),
                    ImageStatistics.PERCENTILES: lambda: np.percentile(flattened, q=QUARTILES, axis=1).T,
                    ImageStatistics.HISTOGRAM: lambda: hist,
                    ImageStatistics.ENTROPY: lambda: entropy(hist, axis=1),
                }
            )
            self.results.append(results)


class BaseAggregateMetric(BaseStatsMetric, Generic[TFlag]):
    FLAG_METRIC_MAP: Dict[type, type]
    DEFAULT_FLAGS: Sequence[TFlag]

    def __init__(self, flags: Optional[Union[TFlag, Sequence[TFlag]]] = None):
        flag_dict = {}
        for flag in flags if isinstance(flags, Sequence) else self.DEFAULT_FLAGS if not flags else [flags]:
            flag_dict[type(flag)] = flag_dict.setdefault(type(flag), type(flag)(0)) | flag
        self._metrics_dict = {
            metric: []
            for metric in (
                self.FLAG_METRIC_MAP[flag_class](flag) for flag_class, flag in flag_dict.items() if flag.value != 0
            )
        }


class ImageStats(BaseAggregateMetric):
    """
    Calculates various image property statistics

    Parameters
    ----------
    flags: [ImageHash | ImageProperty | ImageStatistics | ImageVisuals], default None
        Metric(s) to calculate for each image per channel - calculates all metrics if None
    """

    FLAG_METRIC_MAP = {
        ImageHash: ImageHashMetric,
        ImageProperty: ImagePropertyMetric,
        ImageStatistics: ImageStatisticsMetric,
        ImageVisuals: ImageVisualsMetric,
    }
    DEFAULT_FLAGS = [ImageHash.ALL, ImageProperty.ALL, ImageStatistics.ALL, ImageVisuals.ALL]

    def __init__(self, flags: Optional[Union[ImageStatsFlags, Sequence[ImageStatsFlags]]] = None):
        super().__init__(flags)
        self._length = 0

    def update(self, images: Iterable[ArrayLike]) -> None:
        for image in to_numpy_iter(images):
            self._length += 1
            img = normalize_image_shape(image)
            for metric in self._metrics_dict:
                metric.update([img])

    def compute(self) -> Dict[str, Any]:
        for metric in self._metrics_dict:
            self._metrics_dict[metric] = metric.results

        stats = {}
        for metric, results in self._metrics_dict.items():
            for i, result in enumerate(results):
                for stat in metric._keys():
                    value = result[stat]
                    if not isinstance(value, (np.ndarray, np.generic)):
                        if stat not in stats:
                            stats[stat] = []
                        stats[stat].append(result[stat])
                    else:
                        if stat not in stats:
                            shape = () if np.isscalar(result[stat]) else result[stat].shape
                            stats[stat] = np.empty((self._length,) + shape)
                        stats[stat][i] = result[stat]
        return stats

    def reset(self):
        self._length = 0
        for metric in self._metrics_dict:
            metric.reset()
            self._metrics_dict[metric] = []


class ChannelStats(BaseAggregateMetric):
    FLAG_METRIC_MAP = {ImageStatistics: ChannelStatisticsMetric}
    DEFAULT_FLAGS = [ImageStatistics.ALL]
    IDX_MAP = "idx_map"

    def __init__(self, flags: Optional[ImageStatistics] = None) -> None:
        super().__init__(flags)

    def update(self, images: Iterable[ArrayLike]) -> None:
        for image in to_numpy_iter(images):
            img = normalize_image_shape(image)
            for metric in self._metrics_dict:
                metric.update([img])

        for metric in self._metrics_dict:
            self._metrics_dict[metric] = metric.results

    def compute(self) -> Dict[str, Any]:
        # Aggregate all metrics into a single dictionary
        stats = {}
        channel_stats = set()
        for metric, results in self._metrics_dict.items():
            for i, result in enumerate(results):
                for stat in metric._keys():
                    channel_stats.update(metric._keys())
                    channels = result[stat].shape[0]
                    stats.setdefault(self.IDX_MAP, {}).setdefault(channels, {})[i] = None
                    stats.setdefault(stat, {}).setdefault(channels, []).append(result[stat])

        # Concatenate list of channel statistics numpy
        for stat in channel_stats:
            for channel in stats[stat]:
                stats[stat][channel] = np.array(stats[stat][channel]).T

        for channel in stats[self.IDX_MAP]:
            stats[self.IDX_MAP][channel] = list(stats[self.IDX_MAP][channel].keys())

        return stats

    def reset(self) -> None:
        for metric in self._metrics_dict:
            metric.reset()
            self._metrics_dict[metric] = []
