"""Per-detection geometry features from a detection Model (drift indicator)."""

__all__ = ["DetectionGeometryExtractor"]

from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval.types import ReprMixin
from dataeval.utils._internal import iter_images

_EPS = 1e-6


class DetectionGeometryExtractor(ReprMixin):
    """Turn a detection Model's normalized boxes into per-detection geometry rows.

    Each kept detection becomes ``[center_x, center_y, width, height, area, aspect]``,
    derived from normalized ``(x0, y0, x1, y1)`` boxes (resolution-independent).
    Detections whose max class score is below ``confidence`` are dropped.
    """

    def __init__(self, model: Any, *, confidence: float = 0.0) -> None:
        self._model = model
        self._confidence = confidence

    def _repr_overrides(self) -> dict[str, str]:
        return {"model": self._model.__class__.__name__}

    def __call__(self, data: Any) -> NDArray[np.float32]:
        rows: list[list[float]] = []
        for pred in self._model(list(iter_images(data))):
            boxes = np.asarray(pred.boxes, dtype=np.float32)
            scores = np.asarray(pred.scores, dtype=np.float32)
            conf = scores.max(axis=1) if scores.ndim == 2 else scores
            for box, c in zip(boxes, conf, strict=False):
                if c < self._confidence:
                    continue
                x0, y0, x1, y1 = box
                w = float(x1 - x0)
                h = float(y1 - y0)
                rows.append([
                    float((x0 + x1) / 2),
                    float((y0 + y1) / 2),
                    w,
                    h,
                    w * h,
                    w / (h + _EPS),
                ])
        if not rows:
            return np.empty((0, 6), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)
