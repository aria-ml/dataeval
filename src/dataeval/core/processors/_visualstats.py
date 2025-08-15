from __future__ import annotations

from collections.abc import Iterable
from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval.config import EPSILON
from dataeval.core._processor import BaseProcessor, process
from dataeval.typing import ArrayLike
from dataeval.utils._boundingbox import BoxLike
from dataeval.utils._image import edge_filter

QUARTILES = (0, 25, 50, 75, 100)


class VisualStatsProcessor(BaseProcessor):
    @cached_property
    def percentiles(self) -> NDArray[np.float64]:
        return np.nanpercentile(self.scaled, q=QUARTILES).astype(np.float64)

    def process(self) -> dict[str, list[Any]]:
        return {
            "brightness": [float(self.percentiles[1])],
            "contrast": [
                float(np.max(self.percentiles) - np.min(self.percentiles)) / float(np.mean(self.percentiles) + EPSILON)
            ],
            "darkness": [float(self.percentiles[-2])],
            "sharpness": [float(np.nanstd(edge_filter(np.mean(self.image, axis=0))))],
            "percentiles": [self.percentiles.tolist()],
        }


class VisualStatsPerChannelProcessor(BaseProcessor):
    @cached_property
    def percentiles(self) -> NDArray[np.float64]:
        return np.nanpercentile(self.per_channel, q=QUARTILES, axis=1).T.astype(np.float64)

    def process(self) -> dict[str, list[Any]]:
        return {
            "brightness": self.percentiles[:, 1].tolist(),
            "contrast": (
                (np.max(self.percentiles, axis=1) - np.min(self.percentiles, axis=1))
                / (np.mean(self.percentiles, axis=1) + EPSILON)
            ).tolist(),
            "darkness": self.percentiles[:, -2].tolist(),
            "missing": (np.count_nonzero(np.isnan(self.image), axis=(1, 2)) / np.prod(self.shape[-2:])).tolist(),
            "sharpness": np.nanstd(
                np.vectorize(edge_filter, signature="(m,n)->(m,n)")(self.image), axis=(1, 2)
            ).tolist(),
            "percentiles": self.percentiles.tolist(),
        }


def visualstats(
    images: Iterable[ArrayLike],
    boxes: Iterable[Iterable[BoxLike] | None] | None,
    per_channel: bool = False,
) -> dict[str, Any]:
    return process(images, boxes, [VisualStatsPerChannelProcessor if per_channel is None else VisualStatsProcessor])
