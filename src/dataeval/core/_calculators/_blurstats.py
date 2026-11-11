__all__ = []

from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval.core._blur import BlurValues, blur_stat
from dataeval.core._calculators._base import Calculator, Handler, ViewKind
from dataeval.core._calculators._cache import CalculatorCache
from dataeval.core._calculators._registry import CalculatorRegistry
from dataeval.flags import ImageStats

# A named band group is a smaller cube that averages to a picture like the whole. The background view holds NaN
# where boxes were, and the smoothing would spread those holes over every edge, so it is left out.
_VIEWS = ViewKind.WHOLE | ViewKind.BAND

_NAN = BlurValues(np.nan, np.nan)


@CalculatorRegistry.register(ImageStats)
class BlurStatCalculator(Calculator[ImageStats]):
    """Calculator for the edge-width blur statistics.

    Kept in a class and a file of its own so that the calculator comes out by deleting this file and its lines in
    `_register.py`. The rest of the feature lives elsewhere: the flags in `flags.py`, the measurement in `_blur.py`, the
    warnings in `_image_stats.py` and `_quality_gate.py`, and the invalidation entries of the view operations. The
    registry would send the blur flags to `VisualStatCalculator` if that claimed them, so routing is not the reason.
    The two flags share one detection, run on first use, so asking for both costs one pass.
    """

    def __init__(self, datum: NDArray[Any], cache: "CalculatorCache") -> None:
        self.datum = datum
        self.cache = cache

    @cached_property
    def _values(self) -> BlurValues:
        """The two values of `blur_stat` over the view's gray picture.

        The picture is the same one sharpness reads: the average over bands of the display-range view,
        taken off the window where there is one. NaN where no reading is possible, or below 2D.
        """
        if self.cache.is_unmeasurable or self.cache.image.ndim < 2:
            return _NAN
        gray = self.cache.display_grayscale
        if gray is None:
            perceptual = self.cache.perceptual
            gray = perceptual if self.cache.image.ndim == 2 else np.mean(perceptual, axis=0)
        # blur_stat casts to float32 itself, and returns NaN for anything that is not 2D by then
        return blur_stat(gray)

    def get_applicable_flags(self) -> ImageStats:
        """Return which flags this calculator handles."""
        return ImageStats.BLUR

    def get_handlers(self) -> dict[ImageStats, Handler]:
        """Return mapping of flags to the statistic each produces."""
        return {
            ImageStats.VISUAL_BLUR: Handler("blur", lambda: [self._values.blur], _VIEWS),
            ImageStats.VISUAL_BLUR_DIRECTIONALITY: Handler(
                "blur_directionality", lambda: [self._values.directionality], _VIEWS
            ),
        }
