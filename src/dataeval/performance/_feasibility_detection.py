"""Object detection for the feasibility evaluator: the checks on a detector's predictions, made before any data pass."""

__all__ = []

from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from dataeval.core._detection_counts import check_boxes
from dataeval.types._target import whole_ids
from dataeval.utils._array import as_numpy

_SHAPE = "boxes (N, 4) as (x0, y0, x1, y1), labels (N,) and scores (N,) or (N, n_classes)"


class ImageDetections(NamedTuple):
    """One image's checked detections: float64 ``boxes`` ``(N, 4)``, ``labels`` ``(N,)`` and float64 ``scores``."""

    boxes: NDArray[np.float64]
    labels: NDArray[np.intp]
    scores: NDArray[np.float64]


def _labels(where: str, raw: Any, n_classes: int) -> NDArray[np.intp]:
    values = as_numpy(raw).reshape(-1)
    if values.dtype.kind == "b":  # whole_ids would read True and False as the classes 1 and 0
        raise ValueError(f"{where}: labels are booleans; pass integer indices of the declared classes.")
    labels = whole_ids(values, f"{where}: labels", "Pass integer indices of the declared classes.")
    outside = labels[(labels < 0) | (labels >= n_classes)]
    if len(outside):
        raise ValueError(
            f"{where}: label {int(outside[0])} is outside the declared classes [0, {n_classes}). Pass indices of the "
            "dataset's index2label, in its order."
        )
    return labels


def _scores(where: str, raw: Any, labels: NDArray[np.intp], n_classes: int) -> NDArray[np.float64]:
    """One score per detection: a per-class layout gives each detection the column of its own label."""
    scores = np.asarray(as_numpy(raw), dtype=np.float64)
    n = len(labels)
    if scores.ndim == 2 and scores.shape == (n, n_classes):
        scores = scores[np.arange(n), labels]
    elif scores.shape != (n,):
        raise ValueError(f"{where}: scores of shape {scores.shape} for {n} boxes; give ({n},) or ({n}, {n_classes}).")
    if not np.isfinite(scores).all():
        raise ValueError(f"{where}: scores must be finite; found NaN or infinite scores. Fix or drop those detections.")
    return scores


def check_detections(predictions: Sequence[Any], n_items: int, n_classes: int) -> list[ImageDetections]:
    """
    Check a detector's predictions, one detections object per dataset item in dataset order, and return them read.

    Each object carries ``boxes``, ``labels`` and ``scores``, the MAITE ``ObjectDetectionTarget`` shape. Labels are
    indices of the declared classes; a float array of whole values is accepted. Scores of shape ``(N, n_classes)``
    give each detection the score in its own label's column. Boxes are not clipped.

    Raises
    ------
    ValueError
        Naming the item and the fix, if there is not one detections object per item, an object lacks ``boxes``,
        ``labels`` or ``scores``, a box is not finite or has no positive width and height, labels are booleans, not
        whole numbers, not one per box, or outside ``[0, n_classes)``, or scores are not finite or neither ``(N,)``
        nor ``(N, n_classes)``.
    """
    if isinstance(predictions, np.ndarray) or not hasattr(predictions, "__len__") or len(predictions) != n_items:
        got = f"an array of shape {predictions.shape}" if isinstance(predictions, np.ndarray) else _count(predictions)
        raise ValueError(
            f"predictions for a detection dataset are one detections object per image ({n_items}), got {got}; "
            f"each holds {_SHAPE}."
        )
    return [_image(f"predictions[{i}]", detections, n_classes) for i, detections in enumerate(predictions)]


def _image(where: str, detections: Any, n_classes: int) -> ImageDetections:
    for name in ("boxes", "labels", "scores"):
        if not hasattr(detections, name):
            raise ValueError(f"{where} has no {name!r}; each image's detections hold {_SHAPE}.")
    boxes = check_boxes(as_numpy(detections.boxes), where)
    labels = _labels(where, detections.labels, n_classes)
    if len(labels) != len(boxes):
        raise ValueError(f"{where}: {len(labels)} labels for {len(boxes)} boxes; give one label and score per box.")
    return ImageDetections(boxes, labels, _scores(where, detections.scores, labels, n_classes))


def _count(predictions: Any) -> str:
    return str(len(predictions)) if hasattr(predictions, "__len__") else f"a {type(predictions).__name__}"
