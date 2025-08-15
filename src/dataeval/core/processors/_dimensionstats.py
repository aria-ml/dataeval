from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from dataeval.core._processor import BaseProcessor, process
from dataeval.typing import ArrayLike
from dataeval.utils._boundingbox import BoxLike
from dataeval.utils._image import get_bitdepth


class DimensionStatsProcessor(BaseProcessor):
    def process(self) -> dict[str, list[Any]]:
        return {
            "offset_x": [self.box.x0],
            "offset_y": [self.box.y0],
            "width": [self.box.width],
            "height": [self.box.height],
            "channels": [self.shape[-3]],
            "size": [self.box.width * self.box.height],
            "aspect_ratio": [0.0 if self.box.height == 0 else self.box.width / self.box.height],
            "depth": [get_bitdepth(self.raw).depth],
            "center": [[(self.box.x0 + self.box.x1) / 2, (self.box.y0 + self.box.y1) / 2]],
            "distance_center": [
                float(
                    np.sqrt(
                        np.square(((self.box.x0 + self.box.x1) / 2) - (self.raw.shape[-1] / 2))
                        + np.square(((self.box.y0 + self.box.y1) / 2) - (self.raw.shape[-2] / 2))
                    )
                )
            ],
            "distance_edge": [
                float(
                    np.min(
                        [
                            np.abs(self.box.x0),
                            np.abs(self.box.y0),
                            np.abs(self.box.x1 - self.raw.shape[-1]),
                            np.abs(self.box.y1 - self.raw.shape[-2]),
                        ]
                    )
                )
            ],
            "invalid_box": [not self.box.is_valid()],
        }


def dimensionstats(
    images: Iterable[ArrayLike],
    boxes: Iterable[Iterable[BoxLike] | None] | None,
) -> dict[str, Any]:
    return process(images, boxes, [DimensionStatsProcessor])
