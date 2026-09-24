"""Reducing a datum's annotation to a digest.

Duplicate detection compares pixels. Two data can hold the same pixels under different
annotations (for example, one collection labeled twice). They can also hold the same
annotation over different pixels (a synthetically augmented copy). Neither case is visible to a
hash of the frames.

This reduces the annotation itself to a digest. The same grouping that finds pixel copies
therefore finds annotation copies. Reading it costs no decode. The boxes and labels come from
the target, not the video.
"""

__all__ = []

from collections.abc import Sequence
from typing import Any, NamedTuple, TypedDict

import numpy as np
from numpy.typing import NDArray

_AnnotatedFrame = tuple[Any, Any, Any]


def _normalized(boxes: Any, image_hw: tuple[int, int] | None) -> NDArray[np.float64]:
    """
    Reshape boxes to ``(N, 4)`` float64 and, if given, normalize by image size.

    Parameters
    ----------
    boxes : ArrayLike
        Shape ``(N, 4)`` in XYXY order.
    image_hw : tuple[int, int] or None
        ``(height, width)``. When given, coordinates are divided by it. This makes a rescaled
        copy of the same annotation compare as identical. ``None`` means the boxes are already
        normalized. It is a no-op past the reshape, not an error case.

    Returns
    -------
    NDArray[np.float64]
        The reshaped boxes, normalized when requested.

    Raises
    ------
    ValueError
        If image_hw is given with a non-positive dimension. Both :func:`frame_annotation_hash`
        and :func:`annotation_divergence` reject it here, once.
    """
    box_array = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if image_hw is not None:
        height, width = float(image_hw[0]), float(image_hw[1])
        if height <= 0 or width <= 0:
            raise ValueError(f"image_hw dimensions must be positive, got {image_hw}")
        box_array = box_array / np.array([width, height, width, height], dtype=np.float64)
    return box_array


def _validated_frame(
    boxes: Any,
    labels: Any,
    track_ids: Any,
    image_hw: tuple[int, int] | None,
) -> tuple[NDArray[np.float64], NDArray[Any], NDArray[np.int64]]:
    """
    Normalize one frame's boxes and check that boxes, labels, and track_ids agree in row count.

    Shared by :func:`frame_annotation_hash` and :func:`annotation_divergence`. A ragged frame
    (one whose boxes, labels, and track_ids disagree on row count) is rejected the same way by
    both. This avoids one raising cleanly while the other silently miscounts or indexes out of
    bounds.

    Parameters
    ----------
    boxes : ArrayLike
        Shape ``(N, 4)`` in XYXY order.
    labels : ArrayLike
        Shape ``(N,)``, aligned with **boxes**.
    track_ids : ArrayLike or None
        Shape ``(N,)``, aligned with **boxes**. ``None`` fills a same-length ``-1`` sentinel, so
        "no track ids given" is never itself a length mismatch.
    image_hw : tuple[int, int] or None
        Forwarded to :func:`_normalized`.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[Any], NDArray[np.int64]]
        ``(box_array, label_array, track_array)``, all agreeing in row count.

    Raises
    ------
    ValueError
        If boxes, labels, and track_ids (when given) do not have matching row counts, or if
        image_hw has a non-positive dimension (via :func:`_normalized`).
    """
    box_array = _normalized(boxes, image_hw)
    label_array = np.asarray(labels).reshape(-1)
    track_array: NDArray[np.int64] = (
        np.full(len(label_array), -1, dtype=np.int64)
        if track_ids is None
        else np.asarray(track_ids, dtype=np.int64).reshape(-1)
    )

    num_boxes, num_labels, num_tracks = len(box_array), len(label_array), len(track_array)
    if not (num_boxes == num_labels == num_tracks):
        raise ValueError(
            f"boxes, labels, and track_ids must have matching lengths, got "
            f"boxes={num_boxes}, labels={num_labels}, track_ids={num_tracks}"
        )
    return box_array, label_array, track_array


def _canonical_track_ids(frames: Sequence[_AnnotatedFrame]) -> list[NDArray[np.int64]]:
    """Renumber track ids by order of first appearance across the whole sequence.

    A re-export that keeps the tracks but renumbers them is the same annotation. So ``{0, 1}``
    and ``{7, 9}`` must agree. Renumbering is sequence-wide, not per frame. A track spans
    frames, so numbering each frame alone would lose the structure that tells two tracks apart.
    """
    seen: dict[int, int] = {}
    canonical: list[NDArray[np.int64]] = []
    for _, _, track_ids in frames:
        if track_ids is None:
            canonical.append(np.empty(0, dtype=np.int64))
            continue
        row = np.asarray(track_ids, dtype=np.int64).reshape(-1)
        canonical.append(np.array([seen.setdefault(int(t), len(seen)) for t in row], dtype=np.int64))
    return canonical


def frame_annotation_hash(
    boxes: Any,
    labels: Any,
    *,
    track_ids: Any = None,
    image_hw: tuple[int, int] | None = None,
    decimals: int = 6,
) -> str:
    """
    Reduce one frame's annotation to a digest.

    Parameters
    ----------
    boxes : ArrayLike
        Shape ``(N, 4)`` in XYXY order. An empty array is a frame with no annotation. That is a
        fact about the annotation and hashes to a stable value.
    labels : ArrayLike
        Shape ``(N,)`` class labels, aligned with **boxes**.
    track_ids : ArrayLike or None, default None
        Shape ``(N,)`` track ids, aligned with **boxes**. Pass the canonical numbering from
        :func:`annotation_fingerprint` rather than raw ids. Raw ids depend on the export tool.
    image_hw : tuple[int, int] or None, default None
        ``(height, width)``. When given, coordinates are normalized by it. A rescaled copy of
        the same annotation then agrees. Without it, coordinates are taken as already
        normalized.
    decimals : int, default 6
        Coordinates are rounded to this many decimals. This survives float round-trip wobble. It
        is not a similarity tolerance. That is :func:`annotation_divergence`'s job.

    Returns
    -------
    str
        16-character hex digest.

    Raises
    ------
    ValueError
        If boxes and labels do not have the same length, or if track_ids is provided but does not
        match. Also raised if image_hw is provided with non-positive dimensions.

    Notes
    -----
    Rows are sorted before hashing. Box order within a frame carries no meaning. Two annotations
    differing only in listing order are the same annotation. Scores are excluded. They change on
    re-annotation without the annotation changing.
    """
    import xxhash as xxh

    box_array, label_array, track_array = _validated_frame(boxes, labels, track_ids, image_hw)
    box_array = np.round(box_array, decimals)

    rows = [
        f"{int(label)}|{x0:.{decimals}f},{y0:.{decimals}f},{x1:.{decimals}f},{y1:.{decimals}f}|{int(track)}"
        # strict: `_validated_frame` has already made the three agree, so a mismatch here is a
        # broken invariant and must raise rather than quietly hash a truncated annotation.
        for label, (x0, y0, x1, y1), track in zip(label_array, box_array, track_array, strict=True)
    ]
    return xxh.xxh3_64_hexdigest("\x1f".join(sorted(rows)).encode())


def annotation_fingerprint(
    frames: Sequence[_AnnotatedFrame],
    *,
    image_hw: tuple[int, int] | None = None,
    decimals: int = 6,
) -> str:
    """
    Reduce a datum's whole annotation to a digest.

    An image is a sequence of one frame, so this serves both tasks.

    Parameters
    ----------
    frames : Sequence[tuple[ArrayLike, ArrayLike, ArrayLike | None]]
        ``(boxes, labels, track_ids)`` per frame, **in temporal order**.
    image_hw : tuple[int, int] or None, default None
        ``(height, width)`` used to normalize coordinates.
    decimals : int, default 6
        Coordinate rounding, as in :func:`frame_annotation_hash`.

    Returns
    -------
    str
        16-character hex digest. Two data sharing it carry the same annotation; one differing
        anywhere does not.

    Raises
    ------
    ValueError
        If any frame has mismatched boxes/labels/track_ids lengths, or if image_hw is provided
        with non-positive dimensions.

    Examples
    --------
    >>> import numpy as np
    >>> from dataeval.core import annotation_fingerprint
    >>> boxes = np.array([[10.0, 10.0, 20.0, 20.0]])
    >>> labels = np.array([1])
    >>> a = annotation_fingerprint([(boxes, labels, None)], image_hw=(100, 100))
    >>> b = annotation_fingerprint([(boxes * 2, labels, None)], image_hw=(200, 200))
    >>> a == b
    True
    """
    import xxhash as xxh

    canonical = _canonical_track_ids(frames)
    digests = [
        frame_annotation_hash(
            boxes, labels, track_ids=None if ids.size == 0 else ids, image_hw=image_hw, decimals=decimals
        )
        # strict: `_canonical_track_ids` returns one row per frame by construction, so a
        # mismatch would mean silently fingerprinting only part of a sequence.
        for (boxes, labels, _), ids in zip(frames, canonical, strict=True)
    ]
    return xxh.xxh3_64_hexdigest("\x00".join(digests).encode())


class AnnotationDivergence(TypedDict):
    """
    How two annotations of the same footage differ. Counts and ratios only.

    Attributes
    ----------
    frames_compared : int
        Number of frames compared, i.e. ``min(len(frames_a), len(frames_b))``. A length
        mismatch is a fact about the two annotations, not an error. The shorter sequence bounds
        how many frames can be compared.
    frames_differing : int
        Number of compared frames where anything below (an added or removed box, a relabel)
        was found.
    mean_iou : float or None
        Mean IoU over matched box pairs only. ``None`` when nothing matched. That is distinct
        from an actual IoU of ``0.0``, which means the boxes were compared and did not overlap
        at all. ``None`` means there was nothing to compare.
    boxes_added : int
        Boxes in **b** with no matching box in **a**, summed over frames.
    boxes_removed : int
        Boxes in **a** with no matching box in **b**, summed over frames.
    labels_changed : int
        Matched box pairs whose label differs.
    tracks_split : int
        Directional. Counts track ids in **a** whose matched boxes map to more than one track
        id in **b**. Swapping ``frames_a`` and ``frames_b`` gives a different number. Call it
        both ways to see the divergence in both directions.
    """

    frames_compared: int
    frames_differing: int
    mean_iou: float | None
    boxes_added: int
    boxes_removed: int
    labels_changed: int
    tracks_split: int


class _FrameMatch(NamedTuple):
    """One frame's contribution to :func:`annotation_divergence`."""

    ious: list[float]
    added: int
    removed: int
    changed: int
    differing: bool
    pairs: list[tuple[int, int]]


def _match_frame(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    la: NDArray[Any],
    lb: NDArray[Any],
    iou_threshold: float,
) -> _FrameMatch:
    """
    Match one frame's boxes in ``a`` against ``b`` with optimal assignment on IoU.

    Box counts are read from ``a``/``b`` (the box arrays), never from ``la``/``lb`` (the label
    arrays). Callers must have already validated ``len(a) == len(la)`` and ``len(b) ==
    len(lb)`` via :func:`_validated_frame`. A mismatch here would index ``la``/``lb`` out of
    bounds rather than silently miscount.
    """
    if len(a) == 0 or len(b) == 0:
        return _FrameMatch([], len(b), len(a), 0, len(a) != len(b), [])

    from scipy.optimize import linear_sum_assignment

    from dataeval.utils.preprocessing import compute_iou

    iou = compute_iou(a, b)
    rows, cols = linear_sum_assignment(1.0 - iou)
    keep = iou[rows, cols] >= iou_threshold
    rows, cols = rows[keep], cols[keep]

    changed = int(np.sum(la[rows] != lb[cols]))
    differing = bool(changed or len(a) != len(rows) or len(b) != len(cols))
    pairs = list(zip(rows.tolist(), cols.tolist(), strict=True))
    return _FrameMatch(iou[rows, cols].tolist(), len(b) - len(cols), len(a) - len(rows), changed, differing, pairs)


def annotation_divergence(
    frames_a: Sequence[_AnnotatedFrame],
    frames_b: Sequence[_AnnotatedFrame],
    *,
    image_hw_a: tuple[int, int] | None = None,
    image_hw_b: tuple[int, int] | None = None,
    iou_threshold: float = 0.5,
) -> AnnotationDivergence:
    """
    Measure how far two annotations of the same footage disagree.

    Boxes are matched frame by frame with optimal (not greedy) assignment on IoU. The reported
    counts do not depend on box listing order. It reports what differs and by how much. It does
    **not** say which annotation is right. That is not knowable from the pixels alone.

    Parameters
    ----------
    frames_a : Sequence[tuple[ArrayLike, ArrayLike, ArrayLike | None]]
        ``(boxes, labels, track_ids)`` per frame, in temporal order.
    frames_b : Sequence[tuple[ArrayLike, ArrayLike, ArrayLike | None]]
        Same shape as **frames_a**, for the annotation being compared against it.
    image_hw_a : tuple[int, int] or None, default None
        ``(height, width)`` used to normalize **frames_a** coordinates, as in
        :func:`frame_annotation_hash`.
    image_hw_b : tuple[int, int] or None, default None
        Same, for **frames_b**.
    iou_threshold : float, default 0.5
        Matched box pairs below this IoU are treated as unmatched: the box in **a** counts as
        removed and the box in **b** as added, rather than as a low-quality match.

    Returns
    -------
    AnnotationDivergence

    Raises
    ------
    ValueError
        If a frame's boxes, labels, and track_ids (when given) do not have matching row counts,
        or if image_hw_a/image_hw_b has a non-positive dimension.

    Notes
    -----
    Sequences of unequal length are compared over their shared prefix, the same as
    :func:`zip`; ``frames_compared`` reports that shorter length rather than either input's.
    """
    ious: list[float] = []
    added = removed = relabeled = differing = 0
    track_targets: dict[int, set[int]] = {}

    for (boxes_a, labels_a, tracks_a), (boxes_b, labels_b, tracks_b) in zip(frames_a, frames_b, strict=False):
        a, la, ta = _validated_frame(boxes_a, labels_a, tracks_a, image_hw_a)
        b, lb, tb = _validated_frame(boxes_b, labels_b, tracks_b, image_hw_b)

        match = _match_frame(a, b, la, lb, iou_threshold)
        ious.extend(match.ious)
        added += match.added
        removed += match.removed
        relabeled += match.changed
        differing += int(match.differing)

        if tracks_a is not None and tracks_b is not None:
            for r, c in match.pairs:
                track_targets.setdefault(int(ta[r]), set()).add(int(tb[c]))

    return AnnotationDivergence(
        frames_compared=min(len(frames_a), len(frames_b)),
        frames_differing=differing,
        mean_iou=float(np.mean(ious)) if ious else None,
        boxes_added=added,
        boxes_removed=removed,
        labels_changed=relabeled,
        tracks_split=sum(1 for targets in track_targets.values() if len(targets) > 1),
    )
