__all__ = []

from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval.core._calculators._base import Calculator, ContextNeed, Handler, ViewKind
from dataeval.core._calculators._cache import CalculatorCache
from dataeval.core._calculators._registry import CalculatorRegistry
from dataeval.flags import ImageStats
from dataeval.utils._annotations import AnnotationContext, Kind

_NOT_APPLICABLE = "not_applicable"
_TOO_MANY_OVERLAPS = "too_many_overlaps"
_MISSING_LABELS = "missing_labels"

# The most peers that may intersect one box before its covered fraction is NaN. The union is taken on a grid of
# O(k^2) cells for k intersecting peers, so this bounds memory and not time: at 1,000 peers the grid has up to about
# 4 million cells and one box peaked at about 100 MiB in a worker (tracemalloc, 1,000 random rectangles; a layout whose
# edges coincide needs about a quarter of that), and took about 10 ms. Provisional: it comes from one spike and one
# measurement, and the ImageStats docstring says so.
_MAX_OVERLAPS = 1_000

# The statistics whose details include how many peers a box was compared with.
_COUNTED = (
    ImageStats.SPATIAL_COVERED_FRACTION | ImageStats.SPATIAL_MAX_IOU_SAME_CLASS | ImageStats.SPATIAL_MAX_IOU_OTHER_CLASS
)


def _fraction_inside(row: NDArray[np.float64], extent: tuple[int, int]) -> float:
    """
    Return the share of one box that lies inside the image.

    Parameters
    ----------
    row : NDArray[np.float64]
        The box as given, ``(x0, y0, x1, y1)``. The caller has checked it is a ``VALID`` box, so its width and height
        are positive and finite.
    extent : tuple[int, int]
        The image's ``(H, W)``.

    Returns
    -------
    float
        ``area(box ∩ image) / area(box)``, as the product of the share of each axis. Exactly 1.0 for a box wholly
        inside.
    """
    height, width = extent
    x0, y0, x1, y1 = row
    inside_x = max(0.0, min(x1, width) - max(x0, 0.0))
    inside_y = max(0.0, min(y1, height) - max(y0, 0.0))
    return float((inside_x / (x1 - x0)) * (inside_y / (y1 - y0)))


def _nearest_peer(raw: NDArray[np.float64], peers: NDArray[np.intp], focal: int) -> tuple[int, float] | None:
    """
    Return the peer whose center is nearest to the focal box's, and the distance over the focal box's diagonal.

    Parameters
    ----------
    raw : NDArray[np.float64]
        Every box of the image as given, ``(N, 4)`` in ``x0, y0, x1, y1`` order.
    peers : NDArray[np.intp]
        Indices into `raw` of the boxes to search, in index order, so that on a tie the lowest index is nearest.
    focal : int
        Index into `raw` of the box the distance is measured from. It is a ``VALID`` box.

    Returns
    -------
    tuple[int, float] or None
        The nearest peer's index into `raw` and the distance, or None when there is no peer to search.
    """
    if peers.size == 0:
        return None
    x0, y0, x1, y1 = raw[focal]
    others = raw[peers]
    # ponytail: scans every peer for each box, O(N) per box and O(N^2) per image; a spatial index if one image holds
    # tens of thousands of boxes. Hypot rather than squares so that coordinates near 1e200 do not overflow to inf.
    distance = np.hypot(
        (others[:, 0] + others[:, 2]) / 2 - (x0 + x1) / 2, (others[:, 1] + others[:, 3]) / 2 - (y0 + y1) / 2
    )
    nearest = int(np.argmin(distance))  # the first minimum, so the lowest index among equal distances
    return int(peers[nearest]), float(distance[nearest] / np.hypot(x1 - x0, y1 - y0))


def _intersections(
    clipped: NDArray[np.float64], peers: NDArray[np.intp], focal: int
) -> tuple[NDArray[np.intp], NDArray[np.float64], NDArray[np.float64]]:
    """
    Return the peers that overlap the focal box, with each overlap and its intersection over union.

    Parameters
    ----------
    clipped : NDArray[np.float64]
        Every box of the image clipped to the image, ``(N, 4)`` in ``x0, y0, x1, y1`` order.
    peers : NDArray[np.intp]
        Indices into `clipped` of the boxes to compare, in index order. The caller has checked they are ``VALID``.
    focal : int
        Index into `clipped` of the box compared with. It is a ``VALID`` box.

    Returns
    -------
    tuple[NDArray[np.intp], NDArray[np.float64], NDArray[np.float64]]
        The indices into `clipped` of the peers whose intersection with the focal box has positive area, in index order,
        so a peer that only touches an edge is not one; each intersection as ``x0, y0, x1, y1``, ``(k, 4)``; and each
        peer's IoU with the focal box. One vectorized pass over the peers and no N x N array.
    """
    # ponytail: scans every peer for each box, O(N) per box and O(N^2) per image, as `_nearest_peer` does; a spatial
    # index if one image holds tens of thousands of boxes. An area is a product of sides, which underflows to 0 below
    # about 1e-154 pixels a side and reads NaN; scale by the focal box's longer side if boxes that small ever matter.
    fx0, fy0, fx1, fy1 = clipped[focal]
    others = clipped[peers]
    x0, y0 = np.maximum(others[:, 0], fx0), np.maximum(others[:, 1], fy0)
    x1, y1 = np.minimum(others[:, 2], fx1), np.minimum(others[:, 3], fy1)
    hit = (x1 > x0) & (y1 > y0)
    rects = np.stack([x0[hit], y0[hit], x1[hit], y1[hit]], axis=1)
    inter = (rects[:, 2] - rects[:, 0]) * (rects[:, 3] - rects[:, 1])
    area = (others[hit, 2] - others[hit, 0]) * (others[hit, 3] - others[hit, 1])
    return peers[hit], rects, inter / ((fx1 - fx0) * (fy1 - fy0) + area - inter)


def _union_fraction(box: NDArray[np.float64], rects: NDArray[np.float64]) -> float:
    """
    Return the share of `box` that its overlaps cover, counting overlapping overlaps once.

    Parameters
    ----------
    box : NDArray[np.float64]
        The box, ``(x0, y0, x1, y1)``.
    rects : NDArray[np.float64]
        Rectangles inside `box`, ``(k, 4)`` with ``k >= 1``, as `_intersections` returns them.

    Returns
    -------
    float
        ``area(⋃ rects) / area(box)``, exact up to floating-point rounding. The grid of the distinct edges of the
        rectangles and of the box tiles the box, and the denominator is the sum of its cells, so a covered box reads
        exactly 1.0 and the value never exceeds it. O(k^2) cells, so O(k^2) memory.
    """
    xs = np.unique(np.concatenate([rects[:, 0], rects[:, 2], box[[0, 2]]]))
    ys = np.unique(np.concatenate([rects[:, 1], rects[:, 3], box[[1, 3]]]))
    x0, x1 = np.searchsorted(xs, rects[:, 0]), np.searchsorted(xs, rects[:, 2])
    y0, y1 = np.searchsorted(ys, rects[:, 1]), np.searchsorted(ys, rects[:, 3])
    width = len(ys)
    # A 2-D difference array: one +1/-1 at each rectangle's four corners, whose running sums count the rectangles
    # covering each cell.
    corners = np.concatenate([x0 * width + y0, x1 * width + y0, x0 * width + y1, x1 * width + y1])
    signs = np.repeat([1.0, -1.0, -1.0, 1.0], len(rects))
    diff = np.bincount(corners, weights=signs, minlength=len(xs) * width).reshape(len(xs), width)
    np.cumsum(diff, axis=0, out=diff)
    np.cumsum(diff, axis=1, out=diff)
    count = diff[:-1, :-1]
    cells = np.diff(xs)[:, None] * np.diff(ys)[None, :]
    return float(np.where(count > 0.5, cells, 0.0).sum() / cells.sum())


def _largest(iou: NDArray[np.float64], mask: NDArray[np.bool_], unknown: float) -> tuple[float, int]:
    """
    Return the largest IoU among the peers `mask` selects, and which peer it is.

    Parameters
    ----------
    iou : NDArray[np.float64]
        The IoU of each overlapping peer, in index order.
    mask : NDArray[np.bool_]
        The peers whose relationship to the focal box is known and is the one asked for.
    unknown : float
        The largest IoU among the peers whose relationship is unknown (a label is missing), 0.0 if there are none.

    Returns
    -------
    tuple[float, int]
        The largest IoU, 0.0 when no peer is selected, and its position in `iou` (the lowest on a tie), or -1 when
        that is 0.0. NaN and -1 when an unknown peer overlaps more, since that peer may be the true maximum.
    """
    best = float(iou[mask].max()) if mask.any() else 0.0
    if unknown > best:
        return float("nan"), -1
    return best, int(np.flatnonzero(mask & (iou == best))[0]) if best > 0 else -1


@CalculatorRegistry.register(ImageStats)
class SpatialStatCalculator(Calculator[ImageStats]):
    """Calculator for statistics that compare a box with the other boxes of its image, or with the image's edges."""

    def __init__(self, datum: NDArray[Any], cache: "CalculatorCache") -> None:
        self.datum = datum
        self.cache = cache

    def get_applicable_flags(self) -> ImageStats:
        """Return which flags this calculator handles."""
        return ImageStats.SPATIAL

    def _focal(self) -> tuple[AnnotationContext, int] | None:
        """Return the context and the focal box, or None on a row with none: the image's own, or no spatial extent."""
        context, focal = self.cache.context, self.cache.focal
        if context is None or focal is None or len(self.cache.shape) < 2:
            return None
        return context, focal

    def _box_in_image_fraction(self) -> list[float]:
        at = self._focal()
        if at is None:
            return [np.nan]
        context, focal = at
        if context.kind[focal] != Kind.VALID:
            return [np.nan]
        return [_fraction_inside(context.raw[focal], context.extent)]

    @cached_property
    def _nearest(self) -> tuple[int, float] | None:
        """The focal box's nearest valid peer and the normalized distance to it, or None; read by value and details."""
        at = self._focal()
        if at is None:
            return None
        context, focal = at
        if context.kind[focal] != Kind.VALID:
            return None
        return _nearest_peer(context.raw, context.peers(focal), focal)

    def _nn_distance_norm(self) -> list[float]:
        found = self._nearest
        return [np.nan if found is None else found[1]]

    @cached_property
    def _overlap(self) -> tuple[NDArray[np.intp], NDArray[np.float64], NDArray[np.float64]] | None:
        """The focal box's overlapping valid peers, their intersections and IoUs, or None; read by value and details."""
        at = self._focal()
        if at is None:
            return None
        context, focal = at
        if context.kind[focal] != Kind.VALID:
            return None
        return _intersections(context.clipped, context.peers(focal), focal)

    @cached_property
    def _by_class(self) -> tuple[tuple[float, int], tuple[float, int]] | None:
        """The largest IoU with a peer of the same and of another class, each with its position among the overlaps."""
        at, found = self._focal(), self._overlap
        if at is None or found is None:
            return None
        context, focal = at
        hit, _, iou = found
        # A negative label is missing, and so is every label when none were given: such a peer's relationship is
        # unknown, which is not "another class" and, for two missing labels, not "the same".
        known = np.zeros(len(hit), dtype=np.bool_)
        same = other = known
        if context.labels is not None:
            mine, theirs = context.labels[focal], context.labels[hit]
            known = (theirs >= 0) & (mine >= 0)
            same, other = known & (theirs == mine), known & (theirs != mine)
        unknown = float(iou[~known].max()) if (~known).any() else 0.0
        return _largest(iou, same, unknown), _largest(iou, other, unknown)

    def _max_iou_same_class(self) -> list[float]:
        found = self._by_class
        return [np.nan if found is None else found[0][0]]

    def _max_iou_other_class(self) -> list[float]:
        found = self._by_class
        return [np.nan if found is None else found[1][0]]

    def _covered_fraction(self) -> list[float]:
        at, found = self._focal(), self._overlap
        if at is None or found is None:
            return [np.nan]
        hit, rects, _ = found
        if len(hit) == 0:
            return [0.0]
        if len(hit) > _MAX_OVERLAPS:
            return [np.nan]
        context, focal = at
        return [_union_fraction(context.clipped[focal], rects)]

    def details(self, flags: ImageStats, view: ViewKind = ViewKind.WHOLE) -> dict[str, list[Any]]:  # noqa: ARG002
        """Return the frame's status for this row, and the nearest neighbor's target index and reason for no value."""
        at = self._focal()
        status = _NOT_APPLICABLE if at is None else at[0].status(at[1])
        details: dict[str, list[Any]] = {"spatial_status": [status]}
        if flags & ImageStats.SPATIAL_NN_DISTANCE_NORM:
            found = self._nearest
            details["nn_distance_norm_target_index"] = [-1 if found is None else found[0]]
            # Only a frame that can be trusted says "no neighbor": beside a corrupt box the truth is unknown, and the
            # status already says so.
            details["nn_distance_norm_reason"] = ["no_neighbor" if found is None and status == "ok" else ""]
        if flags & (_COUNTED | ImageStats.SPATIAL_COVERED_FRACTION):
            self._overlap_details(flags, details)
        return details

    def _overlap_details(self, flags: ImageStats, details: dict[str, list[Any]]) -> None:
        """Add the peer counts, the covered fraction's reason and the class statistics' counterparts and reasons."""
        at, found = self._focal(), self._overlap
        if flags & _COUNTED:
            # How much of the frame was usable. Not a count for a box that is not measured, where 0 would read as a
            # measurement.
            counts = (-1, -1)
            if at is not None and found is not None:
                context, focal = at
                counts = (len(context.peers(focal)), sum(context.skipped(focal)))
            details["spatial_peer_count"], details["spatial_skipped_count"] = [counts[0]], [counts[1]]
        if flags & ImageStats.SPATIAL_COVERED_FRACTION:
            too_many = found is not None and len(found[0]) > _MAX_OVERLAPS
            details["covered_fraction_reason"] = [_TOO_MANY_OVERLAPS if too_many else ""]
        if flags & (ImageStats.SPATIAL_MAX_IOU_SAME_CLASS | ImageStats.SPATIAL_MAX_IOU_OTHER_CLASS):
            self._class_details(flags, details)

    def _class_details(self, flags: ImageStats, details: dict[str, list[Any]]) -> None:
        """Add the counterpart and the reason for each class statistic selected."""
        at, overlap, by_class = self._focal(), self._overlap, self._by_class
        # The counterpart of a maximum is the peer that attains it; none exists where the value is 0 or not known.
        for flag, name, side in (
            (ImageStats.SPATIAL_MAX_IOU_SAME_CLASS, "same", 0),
            (ImageStats.SPATIAL_MAX_IOU_OTHER_CLASS, "other", 1),
        ):
            if not flags & flag:
                continue
            value, position = (float("nan"), -1) if by_class is None else by_class[side]
            target = -1 if at is None or overlap is None or position < 0 else int(overlap[0][position])
            details[f"max_iou_{name}_class_target_index"] = [target]
            if name == "other":
                labels = None if at is None else at[0].labels
                details["max_iou_other_class_label"] = [-1 if labels is None or target < 0 else int(labels[target])]
            details[f"max_iou_{name}_class_reason"] = [
                _MISSING_LABELS if by_class is not None and np.isnan(value) else ""
            ]

    def get_empty_values(self) -> dict[str, Any]:
        """Return the values of the details on a row where no statistic is measured."""
        return {
            "spatial_status": _NOT_APPLICABLE,
            "nn_distance_norm_target_index": -1,
            "nn_distance_norm_reason": "",
            "spatial_peer_count": -1,
            "spatial_skipped_count": -1,
            "covered_fraction_reason": "",
            "max_iou_same_class_target_index": -1,
            "max_iou_same_class_reason": "",
            "max_iou_other_class_target_index": -1,
            "max_iou_other_class_label": -1,
            "max_iou_other_class_reason": "",
        }

    def get_handlers(self) -> dict[ImageStats, Handler]:
        """Return mapping of flags to the statistic each produces.

        Geometry takes `Handler`'s default of WHOLE alone: masking pixels or dropping bands moves no box. Every
        statistic declares that it reads the image's other boxes, so `image_stats` builds the annotation context.
        """
        return {
            ImageStats.SPATIAL_NN_DISTANCE_NORM: Handler(
                "nn_distance_norm", self._nn_distance_norm, needs=ContextNeed.CONTEXT
            ),
            ImageStats.SPATIAL_BOX_IN_IMAGE_FRACTION: Handler(
                "box_in_image_fraction", self._box_in_image_fraction, needs=ContextNeed.CONTEXT
            ),
            ImageStats.SPATIAL_COVERED_FRACTION: Handler(
                "covered_fraction", self._covered_fraction, needs=ContextNeed.CONTEXT
            ),
            # These two read the class labels as well, so a dataset's are read only for a request that includes one.
            ImageStats.SPATIAL_MAX_IOU_SAME_CLASS: Handler(
                "max_iou_same_class", self._max_iou_same_class, needs=ContextNeed.LABELS
            ),
            ImageStats.SPATIAL_MAX_IOU_OTHER_CLASS: Handler(
                "max_iou_other_class", self._max_iou_other_class, needs=ContextNeed.LABELS
            ),
        }
