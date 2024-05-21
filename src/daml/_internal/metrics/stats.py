from enum import Flag
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, Sequence, TypeVar

import numpy as np
from scipy.stats import entropy, kurtosis, skew

from daml._internal.metrics.flags import ImageHash, ImageProperty, ImageStatistics, ImageVisuals
from daml._internal.metrics.hash import pchash, xxhash
from daml._internal.metrics.utils import edge_filter, get_bitdepth, normalize_image_shape, rescale

QUARTILES = (0, 25, 50, 75, 100)

TBatch = TypeVar("TBatch", bound=Sequence)
TFlag = TypeVar("TFlag", bound=Flag)


class BaseStatsMetric(Generic[TBatch, TFlag]):
    def __init__(self, flags: TFlag):
        self.flags = flags
        self.results = []

    def update(self, batch: TBatch) -> None:
        """Update internal metric cache for later calculation."""

    def compute(self) -> list:
        return self.results

    def reset(self) -> None:
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


class ImageHashMetric(BaseStatsMetric):
    def __init__(self, flags: ImageHash = ImageHash.ALL):
        super().__init__(flags)

    def update(self, batch: Sequence[np.ndarray]):
        for data in batch:
            results = self._map(
                {
                    ImageHash.XXHASH: lambda: xxhash(data),
                    ImageHash.PCHASH: lambda: pchash(data),
                }
            )
            self.results.append(results)


class ImagePropertyMetric(BaseStatsMetric):
    def __init__(self, flags: ImageProperty = ImageProperty.ALL):
        super().__init__(flags)

    def update(self, batch: Sequence[np.ndarray]):
        for data in batch:
            results = self._map(
                {
                    ImageProperty.WIDTH: lambda: np.int32(data.shape[-1]),
                    ImageProperty.HEIGHT: lambda: np.int32(data.shape[-2]),
                    ImageProperty.SIZE: lambda: np.int32(data.shape[-1] * data.shape[-2]),
                    ImageProperty.ASPECT_RATIO: lambda: data.shape[-1] / np.int32(data.shape[-2]),
                    ImageProperty.CHANNELS: lambda: data.shape[-3],
                    ImageProperty.DEPTH: lambda: get_bitdepth(data).depth,
                }
            )
            self.results.append(results)


class ImageVisualsMetric(BaseStatsMetric):
    def __init__(self, flags: ImageVisuals = ImageVisuals.ALL):
        super().__init__(flags)

    def update(self, batch: Sequence[np.ndarray]):
        for data in batch:
            results = self._map(
                {
                    ImageVisuals.MISSING: lambda: np.sum(np.isnan(data)),
                    ImageVisuals.BRIGHTNESS: lambda: np.mean(rescale(data)),
                    ImageVisuals.BLURRINESS: lambda: np.std(edge_filter(np.mean(data, axis=0))),
                }
            )
            self.results.append(results)


class ImageStatisticsMetric(BaseStatsMetric):
    def __init__(self, flags: ImageStatistics = ImageStatistics.ALL):
        super().__init__(flags)

    def update(self, batch: Sequence[np.ndarray]):
        for data in batch:
            scaled = rescale(data)
            if (ImageStatistics.HISTOGRAM | ImageStatistics.ENTROPY) & self.flags:
                hist = np.histogram(scaled, bins=256, range=(0, 1))[0]

            results = self._map(
                {
                    ImageStatistics.MEAN: lambda: np.mean(scaled),
                    ImageStatistics.ZERO: lambda: np.int32(np.count_nonzero(scaled == 0)),
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
    def __init__(self, flags: ImageStatistics = ImageStatistics.ALL):
        super().__init__(flags)

    def update(self, batch: Sequence[np.ndarray]):
        for data in batch:
            scaled = rescale(data)
            flattened = scaled.reshape(data.shape[0], -1)

            if (ImageStatistics.HISTOGRAM | ImageStatistics.ENTROPY) & self.flags:
                hist = np.apply_along_axis(lambda x: np.histogram(x, bins=256, range=(0, 1))[0], 1, flattened)

            results = self._map(
                {
                    ImageStatistics.MEAN: lambda: np.mean(flattened, axis=1),
                    ImageStatistics.ZERO: lambda: np.count_nonzero(flattened == 0, axis=1),
                    ImageStatistics.VAR: lambda: np.var(flattened, axis=1),
                    ImageStatistics.SKEW: lambda: skew(flattened, axis=1),
                    ImageStatistics.KURTOSIS: lambda: kurtosis(flattened, axis=1),
                    ImageStatistics.PERCENTILES: lambda: np.percentile(flattened, q=QUARTILES, axis=1).T,
                    ImageStatistics.HISTOGRAM: lambda: hist,
                    ImageStatistics.ENTROPY: lambda: entropy(hist, axis=1),
                }
            )
            self.results.append(results)


class ImageStats(BaseStatsMetric):
    IMAGESTATS_METRICS = [ImageHashMetric, ImagePropertyMetric, ImageVisualsMetric, ImageStatisticsMetric]

    def __init__(self, metrics: Optional[Sequence[BaseStatsMetric]] = None) -> None:
        metrics_dict: Dict[BaseStatsMetric, List[Dict[str, Any]]] = {
            metric: [] for metric in (metrics if metrics else [metric() for metric in self.IMAGESTATS_METRICS])
        }
        self.metrics_dict = metrics_dict
        self.length = 0

    def update(self, batch: Sequence[np.ndarray]):
        # Run the images through each metric
        for image in batch:
            self.length += 1
            img = normalize_image_shape(image)
            for metric in self.metrics_dict:
                metric.update([img])

    def compute(self):
        # Compute each metric
        for metric in self.metrics_dict:
            self.metrics_dict[metric] = metric.compute()

        # Aggregate all metrics into a single dictionary
        self.stats = {}
        for metric, results in self.metrics_dict.items():
            for i, result in enumerate(results):
                for stat in metric._keys():
                    value = result[stat]
                    if not isinstance(value, (np.ndarray, np.generic)):
                        if stat not in self.stats:
                            self.stats[stat] = []
                        self.stats[stat].append(result[stat])
                    else:
                        if stat not in self.stats:
                            shape = () if np.isscalar(result[stat]) else result[stat].shape
                            self.stats[stat] = np.empty((self.length,) + shape)
                        self.stats[stat][i] = result[stat]

    def reset(self):
        self.length = 0
        for metric in self.metrics_dict:
            metric.reset()
            self.metrics_dict[metric] = []


class ChannelStats(BaseStatsMetric):
    CHANNELSTATS_METRICS = [ChannelStatisticsMetric]
    IDX_MAP = "idx_map"

    def __init__(self, metrics: Optional[ChannelStatisticsMetric] = None) -> None:
        if not metrics:
            metrics = ChannelStatisticsMetric()

        metrics_dict: Dict[ChannelStatisticsMetric, List[Dict[str, Any]]] = {metric: [] for metric in [metrics]}
        self.metrics_dict = metrics_dict
        self.length = 0

    def update(self, batch: Sequence[np.ndarray]):
        # Run the images through each metric
        for image in batch:
            self.length += 1
            img = normalize_image_shape(image)
            for metric in self.metrics_dict:
                metric.update([img])

        # Compute each metric
        for metric in self.metrics_dict:
            self.metrics_dict[metric] = metric.compute()

    def compute(self):
        # Aggregate all metrics into a single dictionary
        stats = {}
        channel_stats = set()
        for metric, results in self.metrics_dict.items():
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

    def reset(self):
        self.length = 0
        for metric in self.metrics_dict:
            metric.reset()
            self.metrics_dict[metric] = []
