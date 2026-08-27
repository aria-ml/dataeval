"""Reading detections off a MAITE target, one way for every reader.

A target's fields are loosely specified on purpose — ``scores`` may be one per box or one
per box per class, ``track_ids`` is absent on the tasks that have no tracks — and every
reader that guesses differently is a place two parts of the library disagree about the
same detection. These are the one reading, shared by the structuring walk behind
:class:`~dataeval.Metadata` and by anything that follows a
:class:`~dataeval.types.SourceIndex` back to a detection — the two must agree about *which*
detection a ``target_index`` names.
"""

__all__ = []

from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval.protocols import ObjectDetectionTarget, SingleFrameObjectTrackingTarget
from dataeval.utils._internal import as_numpy


def detection_count(target: Any) -> int:
    """How many detections one target holds.

    **Labels are the authority.** Every other per-detection array is read against this
    count, which is what makes a detection's position mean the same thing to every reader —
    and a ``target_index`` name the same row whether it is being assigned or followed.
    """
    return len(as_numpy(target.labels).reshape(-1))


def instance_arrays(
    target: "ObjectDetectionTarget | SingleFrameObjectTrackingTarget",
) -> tuple[NDArray[np.intp], NDArray[np.float32], NDArray[np.float32]]:
    """Read one target's per-detection labels, boxes and scores.

    Parameters
    ----------
    target : ObjectDetectionTarget or SingleFrameObjectTrackingTarget
        The target to read. A tracking frame's target is a detection target plus
        ``track_ids``, so both tasks read boxes and labels the same way.

    Returns
    -------
    tuple
        ``(labels, boxes, scores)`` with one entry per detection. **Labels establish the
        count** — every other per-detection array is read against it, which is what makes
        one detection's position mean the same thing to every reader. ``scores`` keeps its
        original shape, which may be per-detection or per-detection-per-class; use
        :func:`detection_score` to read one.
    """
    labels = as_numpy(target.labels).reshape(-1).astype(np.intp)
    count = len(labels)
    boxes = as_numpy(target.boxes).astype(np.float32).reshape(count, 4) if count else np.empty((0, 4), dtype=np.float32)
    scores = as_numpy(target.scores).astype(np.float32) if count else np.empty(0, dtype=np.float32)
    return labels, boxes, scores


def track_ids_of(target: Any, count: int) -> NDArray[np.intp]:
    """Read one frame's per-detection track ids, ``-1`` where a detection has no track.

    Parameters
    ----------
    target : Any
        A tracking frame's target.
    count : int
        Detections in this frame, as :func:`instance_arrays` established it.

    Returns
    -------
    NDArray[np.intp]
        One id per detection. ``-1`` marks a detection no tracker linked, and stands in
        wherever the target supplies no id at all: a frame holding no detections need not
        carry ``track_ids``, and one carrying too few is padded rather than read past its
        end. Tolerant here because ``count`` is the authority on how many rows exist, so a
        disagreeing ``track_ids`` must not be allowed to change that count.
    """
    if not count:
        return np.empty(0, dtype=np.intp)
    ids = getattr(target, "track_ids", None)
    if ids is None:
        return np.full(count, -1, dtype=np.intp)
    values = as_numpy(ids).reshape(-1).astype(np.intp)
    if len(values) == count:
        return values
    if len(values) > count:
        return values[:count]
    return np.concatenate([values, np.full(count - len(values), -1, dtype=np.intp)])


def detection_score(target: Any, index: int, label: int) -> float | None:
    """Read one detection's confidence, from either score layout a target may carry.

    Parameters
    ----------
    target : Any
        The target the detection sits in.
    index : int
        The detection's position within `target`.
    label : int
        The detection's class, which picks the column under a per-class layout.

    Returns
    -------
    float or None
        The confidence, or ``None`` where the target carries none. A ground-truth target
        scores ``1.0``. Where scores are ``(N, CLASSES)`` this is the score of the box's
        **own** class rather than its highest, so a detection's score is about the class it
        was labelled with.
    """
    scores = getattr(target, "scores", None)
    if scores is None:
        return None
    values = as_numpy(scores)
    if values.ndim == 1:
        return float(values[index])
    if values.ndim == 2:
        return float(values[index, label])
    return None
