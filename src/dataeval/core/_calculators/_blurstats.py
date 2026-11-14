__all__ = []

from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval.core._blur import BlurValues, background_blur, blur_values, grow_region, measure_edges
from dataeval.core._calculators._base import Calculator, Handler, ViewKind
from dataeval.core._calculators._cache import CalculatorCache
from dataeval.core._calculators._registry import CalculatorRegistry
from dataeval.flags import ImageStats

# A named band group is a smaller cube that averages to a picture like the whole. Directionality is left out of the
# background view because nothing has tested it there.
_VIEWS = ViewKind.WHOLE | ViewKind.BAND
# `blur` answers the background view too, but not by reading it: the view holds NaN where boxes were, and the smoothing
# would spread those holes over every edge. The edges are found on the unmasked picture and the ones outside the boxes,
# and a margin around them, are pooled.
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
    that pass is also shared with the background value, through `CalculatorCache.scratch`, when the whole-image pass
    ran first on the same datum.
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

    @cached_property
    def _background(self) -> float:
        """The blur value over the edge points outside the boxes grown by the margin, on the unmasked picture.

        The edges come from the whole-image pass when it ran first on this datum, otherwise from a sibling cache with
        nothing masked. NaN where the view cannot be measured, which includes boxes that cover all of it.
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
            edges = measure_edges(self._gray(unmasked))
        return background_blur(edges, grow_region(mask))

    def _blur(self) -> float:
        """Return the `blur` column: the background value in a masked view, the whole picture's otherwise."""
        return self._background if self.cache.exclude is not None else self._values.blur

    def get_applicable_flags(self) -> ImageStats:
        """Return which flags this calculator handles."""
        return ImageStats.BLUR

    def get_handlers(self) -> dict[ImageStats, Handler]:
        """Return mapping of flags to the statistic each produces."""
        return {
            ImageStats.VISUAL_BLUR: Handler("blur", lambda: [self._blur()], _BLUR_VIEWS),
            ImageStats.VISUAL_BLUR_DIRECTIONALITY: Handler(
                "blur_directionality", lambda: [self._values.directionality], _VIEWS
            ),
        }
