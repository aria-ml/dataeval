"""Adapter: a MAITE Model's predictions -> per-row class scores (FeatureExtractor)."""

__all__ = ["ScoresExtractor"]

from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval.types import ReprMixin
from dataeval.utils._internal import iter_images


class ScoresExtractor(ReprMixin):
    """Wrap a MAITE classification/detection Model as a ``FeatureExtractor``.

    Classification: stacks per-image ``(nClasses,)`` -> ``(n_images, nClasses)``.
    Detection: flattens per-detection class scores -> ``(n_detections, nClasses)``.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    def _repr_overrides(self) -> dict[str, str]:
        return {"model": self._model.__class__.__name__}

    def __call__(self, data: Any) -> NDArray[np.float32]:
        images = list(iter_images(data))
        preds = self._model(images)
        rows: list[NDArray[np.float32]] = []
        for pred in preds:
            if hasattr(pred, "scores"):  # ObjectDetectionTarget
                scores = np.asarray(pred.scores, dtype=np.float32)
                if scores.ndim == 1:
                    scores = scores[:, None]
                rows.extend(scores[i] for i in range(scores.shape[0]))
            else:  # classification: a (nClasses,) array
                rows.append(np.asarray(pred, dtype=np.float32))
        if not rows:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack(rows).astype(np.float32)
