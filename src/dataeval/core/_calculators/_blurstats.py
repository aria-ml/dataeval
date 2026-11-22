__all__ = []

import math
from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval.core._blur import BlurValues, Edges, background_blur, blur_values, grow_region, measure_edges
from dataeval.core._calculators._base import Calculator, Handler, ViewKind
from dataeval.core._calculators._cache import CalculatorCache
from dataeval.core._calculators._registry import CalculatorRegistry
from dataeval.flags import ImageStats

# A named band group is a smaller cube that averages to a picture like the whole. Directionality is left out of the
# background view, and reads NaN under a mask, because nothing has tested it there.
_VIEWS = ViewKind.WHOLE | ViewKind.BAND
# `blur_edge_width` answers the background view too, but not by reading it: the view holds NaN where boxes were, and the
# smoothing would spread those holes over every edge. The edges are found on the unmasked picture and the ones outside
# the boxes, and a margin around them, are pooled.
_BLUR_VIEWS = _VIEWS | ViewKind.MASK

_NAN = BlurValues(np.nan, np.nan)


@CalculatorRegistry.register(ImageStats)
class BlurStatCalculator(Calculator[ImageStats]):
    """Calculator for the edge-width blur statistics.

    Kept in a class and a file of its own so that the calculator comes out by deleting this file and its lines in
    `_register.py`. The rest of the feature lives elsewhere: the flags in `flags.py`, the measurement in `_blur.py`, the
    warnings in `_image_stats.py` and `_quality_gate.py`, and the invalidation entries of the view operations. The
    registry would send the blur flags to `VisualStatCalculator` if that claimed them, so routing is not the reason.
    The two flags share one detection, run on first use, so asking for both costs one pass. Under `per_background`
    that pass is also shared with the background value, through `CalculatorCache.scratch`: the pass that measures the
    datum's edges first keeps them there, and the other reads them. Under the call's mask both passes find the same
    edges, since the threshold and the cut come from the mask alone and never from the boxes. Directionality then
    reads NaN, with the reason ``masked``, and the whole picture's values are never measured.
    """

    def __init__(self, datum: NDArray[Any], cache: "CalculatorCache") -> None:
        self.datum = datum
        self.cache = cache

    @staticmethod
    def _gray(cache: CalculatorCache) -> NDArray[Any]:
        """Return the picture blur reads: the average over bands of the display-range view, off the window if any.

        The same one sharpness reads.
        """
        gray = cache.display_grayscale
        if gray is None:
            perceptual = cache.perceptual
            gray = perceptual if cache.image.ndim == 2 else np.mean(perceptual, axis=0)
        return gray

    @cached_property
    def _values(self) -> BlurValues:
        """The two values over the view's gray picture. NaN where no reading is possible, or below 2D."""
        if self.cache.is_unmeasurable or self.cache.image.ndim < 2:
            return _NAN
        edges = measure_edges(self._gray(self.cache))
        if self.cache.scratch is not None:
            self.cache.scratch["edges"] = edges
        return blur_values(edges)

    def _edges(self, picture: CalculatorCache) -> Edges:
        """Measure the edges of `picture`'s gray picture, leaving the call's mask out where it reaches the window.

        Under the mask, the threshold is the mean gradient outside the mask grown by the pooling margin, and the widths
        stop at the mask grown by 1 px, so that no masked pixel sets which edges count or how wide they are. Both are
        read off the mask alone, in the window's coordinates.
        """
        gray = self._gray(picture)
        mask = self.cache.mask_window
        if mask is None or not mask.any():
            return measure_edges(gray)
        return measure_edges(gray, threshold_region=~grow_region(mask), cut=grow_region(mask, 1))

    @cached_property
    def _background(self) -> float:
        """The blur value over the edge points outside the excluded pixels grown by the margin, on the unmasked picture.

        The excluded pixels are the boxes on a background, the call's mask on a row under one, or both. The edges come
        from `scratch` when another pass of this view of the datum measured them first, otherwise from a sibling cache
        with nothing excluded, and are then kept in `scratch` for the other pass. NaN where the view cannot be
        measured, which includes boxes, or a mask, that cover all of it.
        """
        cache = self.cache
        mask = cache.window_mask
        if cache.is_unmeasurable or mask is None:
            return float("nan")
        edges = cache.scratch.get("edges") if cache.scratch is not None else None
        if edges is None:
            unmasked = CalculatorCache(
                cache.raw,
                cache.box if cache.has_box else None,
                normalize_pixel_values=cache.normalize_pixel_values,
                value_range=cache.value_range,
                bands=cache.bands,
            )
            edges = self._edges(unmasked)
            if cache.scratch is not None:
                cache.scratch["edges"] = edges
        return background_blur(edges, grow_region(mask))

    def _directionality(self) -> float:
        """Return the `blur_edge_directionality` column: NaN under the call's mask, the whole picture's otherwise."""
        return float("nan") if self.cache.mask is not None else self._values.directionality

    def _blur(self) -> float:
        """Return the `blur_edge_width` column: the background value in a masked view, the whole picture's otherwise."""
        return self._background if self.cache.exclude is not None else self._values.blur

    @cached_property
    def _status(self) -> str:
        """Name why the row has no edge width, or ``ok``, in the shared order of the statuses.

        A box that extends past the image reads `insufficient_support`: the fill outside the image is NaN, and a NaN
        pixel spreads through the smoothing so that no edge is found.
        """
        cache = self.cache
        if cache.invalid_box:
            return "invalid_box"
        if cache.lacks_bands:
            return "missing_channel"
        if not cache.value_range.is_known:
            return "unknown_range"
        return "ok" if math.isfinite(self._blur()) else "insufficient_support"

    def details(self, flags: ImageStats, view: ViewKind = ViewKind.WHOLE) -> dict[str, list[Any]]:  # noqa: ARG002
        """Return the row's status, and why directionality alone is NaN where the edge width has a value."""
        details: dict[str, list[Any]] = {"blur_status": [self._status]}
        if flags & ImageStats.VISUAL_BLUR_EDGE_DIRECTIONALITY:
            if self.cache.mask is not None:
                reason = "masked"
            else:
                alone = self._status == "ok" and not math.isfinite(self._values.directionality)
                reason = "insufficient_axis_support" if alone else ""
            details["blur_edge_directionality_reason"] = [reason]
        return details

    def get_empty_values(self) -> dict[str, Any]:
        """Return the values of the details on a row where no statistic is measured."""
        return {"blur_status": "not_applicable", "blur_edge_directionality_reason": ""}

    def get_applicable_flags(self) -> ImageStats:
        """Return which flags this calculator handles."""
        return ImageStats.BLUR

    def get_handlers(self) -> dict[ImageStats, Handler]:
        """Return mapping of flags to the statistic each produces."""
        return {
            ImageStats.VISUAL_BLUR_EDGE_WIDTH: Handler("blur_edge_width", lambda: [self._blur()], _BLUR_VIEWS),
            ImageStats.VISUAL_BLUR_EDGE_DIRECTIONALITY: Handler(
                "blur_edge_directionality", lambda: [self._directionality()], _VIEWS
            ),
        }
