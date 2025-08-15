from __future__ import annotations

from collections.abc import Iterable
from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import entropy, kurtosis, skew

from dataeval.core._processor import BaseProcessor, process
from dataeval.typing import ArrayLike
from dataeval.utils._boundingbox import BoxLike


class PixelStatsProcessor(BaseProcessor):
    @cached_property
    def histogram(self) -> NDArray[np.float64]:
        return np.histogram(self.scaled, bins=256, range=(0, 1))[0]

    def process(self) -> dict[str, list[Any]]:
        return {
            "mean": [float(np.nanmean(self.scaled))],
            "std": [float(np.nanstd(self.scaled))],
            "var": [float(np.nanvar(self.scaled))],
            "skew": [float(skew(self.scaled.ravel(), nan_policy="omit"))],
            "kurtosis": [float(kurtosis(self.scaled.ravel(), nan_policy="omit"))],
            "entropy": [float(entropy(self.histogram))],
            "missing": [float(np.count_nonzero(np.isnan(self.image)) / np.prod(self.shape[-2:]))],
            "zeros": [float(np.count_nonzero(np.sum(self.image, axis=0) == 0) / np.prod(self.shape[-2:]))],
            "histogram": [self.histogram.tolist()],
        }


class PixelStatsPerChannelProcessor(BaseProcessor):
    @cached_property
    def histogram(self) -> NDArray[np.float64]:
        return np.apply_along_axis(lambda y: np.histogram(y, bins=256, range=(0, 1))[0], 1, self.per_channel)

    def process(self) -> dict[str, list[Any]]:
        return {
            "mean": np.nanmean(self.per_channel, axis=1).tolist(),
            "std": np.nanstd(self.per_channel, axis=1).tolist(),
            "var": np.nanvar(self.per_channel, axis=1).tolist(),
            "skew": skew(self.per_channel, axis=1, nan_policy="omit").tolist(),
            "kurtosis": kurtosis(self.per_channel, axis=1, nan_policy="omit").tolist(),
            "entropy": np.asarray(entropy(self.histogram, axis=1)).tolist(),
            "missing": (np.count_nonzero(np.isnan(self.image), axis=(1, 2)) / np.prod(self.shape[-2:])).tolist(),
            "zeros": (np.count_nonzero(self.image == 0, axis=(1, 2)) / np.prod(self.shape[-2:])).tolist(),
            "histogram": self.histogram.tolist(),
        }


def pixelstats(
    images: Iterable[ArrayLike],
    boxes: Iterable[Iterable[BoxLike] | None] | None,
    per_channel: bool = False,
) -> dict[str, Any]:
    return process(images, boxes, [PixelStatsPerChannelProcessor if per_channel else PixelStatsProcessor])
