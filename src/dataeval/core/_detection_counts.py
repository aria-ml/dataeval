"""
A frozen detector's true and false positives on one class, per image, at an IoU and a grid of score thresholds.

These are the counts a detector witness bounds. They come from the evaluator's own greedy matching, which is what a
witness must use: the maximum matching of the DFE replica envelope is an optimistic relaxation and would overstate a
detector.
"""

__all__ = []

from collections.abc import Sequence
from typing import TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dataeval.utils.preprocessing import compute_iou

# pycocotools caps the IoU threshold just below 1, so a box equal to its target matches at ``iou=1``.
_IOU_CAP = 1 - 1e-10


class DetectionCountsResult(TypedDict):
    """
    Per-image counts from :func:`detection_counts`.

    - true_positives : NDArray[np.intp] - Shape ``(n_images, n_thresholds)``, detections that matched a target.
    - false_positives : NDArray[np.intp] - Shape ``(n_images, n_thresholds)``, detections that matched none.
    """

    true_positives: NDArray[np.intp]
    false_positives: NDArray[np.intp]


def check_boxes(boxes: ArrayLike, where: str) -> NDArray[np.float64]:
    """
    Return ``boxes`` as float64 of shape ``(N, 4)``, refusing any that is not finite or has no positive area.

    An empty 1-D array is no boxes, ``(0, 4)``. Boxes are (x0, y0, x1, y1) and are not clipped to an image.

    Raises
    ------
    ValueError
        Prefixed with ``where`` and naming the first bad box, if the boxes are not ``(N, 4)``, a box is not finite, or
        a box has ``x1 <= x0`` or ``y1 <= y0``.
    """
    arr = np.asarray(boxes, dtype=np.float64)
    if arr.shape == (0,):
        return np.empty((0, 4), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"{where}: boxes must have shape (N, 4) in (x0, y0, x1, y1) order, got {arr.shape}.")
    infinite = np.flatnonzero(~np.isfinite(arr).all(axis=1))
    if len(infinite):
        raise ValueError(f"{where}: box {infinite[0]} is {arr[infinite[0]].tolist()}, not finite. Fix or drop it.")
    flat = np.flatnonzero((arr[:, 2] <= arr[:, 0]) | (arr[:, 3] <= arr[:, 1]))
    if len(flat):
        raise ValueError(
            f"{where}: box {flat[0]} is {arr[flat[0]].tolist()}, with x1 <= x0 or y1 <= y0. Boxes need positive width "
            "and height, in (x0, y0, x1, y1) order; fix or drop it."
        )
    return arr


def _matched(targets: NDArray[np.float64], boxes: NDArray[np.float64], iou: float) -> NDArray[np.bool_]:
    """Whether each detection, taken in the order given, matches a target under pycocotools' greedy rule."""
    matched = np.zeros(len(boxes), dtype=bool)
    if not len(targets) or not len(boxes):
        return matched
    overlaps = compute_iou(boxes, targets)
    floor = min(iou, _IOU_CAP)
    free = np.ones(len(targets), dtype=bool)
    last = len(targets) - 1
    # ponytail: a Python loop over detections, O(K x G) per image; vectorize if dense scenes are slow.
    for d, row in enumerate(overlaps):
        candidates = np.where(free & (row >= floor), row, -1.0)
        g = last - int(np.argmax(candidates[::-1]))  # the later of two equal IoUs wins, as in pycocotools
        if candidates[g] >= 0:
            free[g] = False
            matched[d] = True
    return matched


def _check_arguments(
    target_boxes: Sequence[ArrayLike],
    detection_boxes: Sequence[ArrayLike],
    detection_scores: Sequence[ArrayLike],
    iou: float,
    max_detections: int,
) -> int:
    """Check what both public functions take, and return the number of images."""
    n_images = len(target_boxes)
    if not len(detection_boxes) == len(detection_scores) == n_images:
        raise ValueError(
            "target_boxes, detection_boxes and detection_scores need one entry per image, got "
            f"{n_images}, {len(detection_boxes)} and {len(detection_scores)}."
        )
    if not 0 < iou <= 1:
        raise ValueError(f"iou must be in (0, 1], got {iou}.")
    if isinstance(max_detections, bool) or not isinstance(max_detections, int | np.integer) or max_detections < 1:
        raise ValueError(f"max_detections must be an integer of at least 1, got {max_detections!r}.")
    return n_images


def _kept(
    i: int,
    target_boxes: ArrayLike,
    detection_boxes: ArrayLike,
    detection_scores: ArrayLike,
    iou: float,
    max_detections: int,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """One image's kept detections, as their scores in kept order and whether each matched a target."""
    targets = check_boxes(target_boxes, f"image {i} targets")
    boxes = check_boxes(detection_boxes, f"image {i} detections")
    scores = np.asarray(detection_scores, dtype=np.float64).reshape(-1)
    if len(scores) != len(boxes):
        raise ValueError(f"image {i}: {len(scores)} scores for {len(boxes)} detection boxes; give one each.")
    if not np.isfinite(scores).all():
        raise ValueError(f"image {i}: scores must be finite; found NaN or infinite scores.")
    order = np.argsort(-scores, kind="stable")[:max_detections]
    return scores[order], _matched(targets, boxes[order], iou)


def detection_counts(
    target_boxes: Sequence[ArrayLike],
    detection_boxes: Sequence[ArrayLike],
    detection_scores: Sequence[ArrayLike],
    *,
    iou: float,
    thresholds: ArrayLike,
    max_detections: int,
) -> DetectionCountsResult:
    """
    Count each image's true and false positives at each score threshold, matching as the COCO evaluator does.

    The caller selects one class: every box passed is of that class. Per image, the detections are sorted by score,
    highest first, ties kept in the order given, and the top ``max_detections`` are kept (COCO's ``maxDets``, per image
    and class). Each detection in turn takes the unmatched target of highest IoU among those of IoU at least ``iou``
    (capped at ``1 - 1e-10``); on an IoU tie the later target wins. A detection with no such target is a false
    positive. This is pycocotools' ``evaluateImg`` without crowd, ignore or area ranges [1]_. A detection counts at
    threshold ``t`` when its score is at least ``t``. Matching in score order makes the detections counted at ``t`` a
    prefix of one pass, so both counts rise as ``t`` falls. Boxes are not clipped to the image.

    Parameters
    ----------
    target_boxes : Sequence[ArrayLike]
        Per image, the targets of the class, shape ``(G_i, 4)`` as (x0, y0, x1, y1).
    detection_boxes : Sequence[ArrayLike]
        Per image, the detections of the class, shape ``(D_i, 4)`` as (x0, y0, x1, y1).
    detection_scores : Sequence[ArrayLike]
        Per image, shape ``(D_i,)`` the detections' scores.
    iou : float
        The IoU a detection needs with a target to match it, in ``(0, 1]``.
    thresholds : ArrayLike
        Shape ``(n_thresholds,)`` score thresholds; the result's columns follow them as given. A threshold of ``-inf``
        counts every kept detection.
    max_detections : int
        The most detections kept per image, at least 1.

    Returns
    -------
    DetectionCountsResult
        ``true_positives`` and ``false_positives``, each of shape ``(n_images, n_thresholds)``.

    Raises
    ------
    ValueError
        If the three sequences differ in length, a box is not finite or has no positive width and height, scores are
        not finite or not one per detection box, ``thresholds`` is not a nonempty 1-D array without NaN or ``+inf``,
        ``iou`` is outside ``(0, 1]``, or ``max_detections`` is not an integer of at least 1.

    References
    ----------
    .. [1] Lin, T.-Y. et al. (2014). Microsoft COCO: Common Objects in Context. ECCV. arXiv:1405.0312.
    """
    n_images = _check_arguments(target_boxes, detection_boxes, detection_scores, iou, max_detections)
    grid = np.asarray(thresholds, dtype=np.float64)
    if grid.ndim != 1 or not grid.size or np.isnan(grid).any() or np.isposinf(grid).any():
        raise ValueError(
            "thresholds must be a nonempty 1-D array of scores; -inf counts every kept detection, NaN and +inf are "
            f"refused. Got shape {grid.shape}."
        )
    tp = np.zeros((n_images, len(grid)), dtype=np.intp)
    fp = np.zeros((n_images, len(grid)), dtype=np.intp)
    for i in range(n_images):
        kept, matched = _kept(i, target_boxes[i], detection_boxes[i], detection_scores[i], iou, max_detections)
        hits = np.r_[0, np.cumsum(matched)]
        counted = np.searchsorted(-kept, -grid, side="right")  # detections scoring at least each threshold
        tp[i] = hits[counted]
        fp[i] = counted - tp[i]
    return DetectionCountsResult(true_positives=tp, false_positives=fp)


def kept_matches(
    target_boxes: Sequence[ArrayLike],
    detection_boxes: Sequence[ArrayLike],
    detection_scores: Sequence[ArrayLike],
    *,
    iou: float,
    max_detections: int,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """
    Return the score of every kept detection and whether it matched a target, as :func:`detection_counts` matches.

    The detections kept are the top ``max_detections`` of each image, and the match is the same greedy pass, from the
    same code, so a detection counted as a true positive by :func:`detection_counts` at some threshold is a ``True``
    here. The arguments are those of :func:`detection_counts` without ``thresholds``.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.bool_]]
        The scores and the matched flags, each of shape ``(n_kept,)``, over the images in order and each image's
        detections in kept order (highest score first). Both are empty when no detection is kept.

    Raises
    ------
    ValueError
        As :func:`detection_counts`, apart from ``thresholds``.
    """
    n_images = _check_arguments(target_boxes, detection_boxes, detection_scores, iou, max_detections)
    parts = [
        _kept(i, target_boxes[i], detection_boxes[i], detection_scores[i], iou, max_detections) for i in range(n_images)
    ]
    if not parts:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=bool)
    return np.concatenate([scores for scores, _ in parts]), np.concatenate([matched for _, matched in parts])
