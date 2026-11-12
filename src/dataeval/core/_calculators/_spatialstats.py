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
        return details

    def get_empty_values(self) -> dict[str, Any]:
        """Return the values of the details on a row where no statistic is measured."""
        return {
            "spatial_status": _NOT_APPLICABLE,
            "nn_distance_norm_target_index": -1,
            "nn_distance_norm_reason": "",
        }

    def get_handlers(self) -> dict[ImageStats, Handler]:
        """Return mapping of flags to the statistic each produces.

        Geometry takes `Handler`'s default of WHOLE alone: masking pixels or dropping bands moves no box. Both
        statistics declare that they read the image's other boxes, so `image_stats` builds the annotation context.
        """
        return {
            ImageStats.SPATIAL_NN_DISTANCE_NORM: Handler(
                "nn_distance_norm", self._nn_distance_norm, needs=ContextNeed.CONTEXT
            ),
            ImageStats.SPATIAL_BOX_IN_IMAGE_FRACTION: Handler(
                "box_in_image_fraction", self._box_in_image_fraction, needs=ContextNeed.CONTEXT
            ),
        }
