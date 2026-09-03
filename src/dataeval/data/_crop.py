"""Crop a fixed region out of every image in a dataset."""

from __future__ import annotations

__all__ = []

from typing import Any

from dataeval.data._geometry import GeometryMap, rewrite_geometry
from dataeval.data._view import Operation, View
from dataeval.flags import ImageStats
from dataeval.utils._array import as_numpy
from dataeval.utils.preprocessing import normalize_image_shape

#: Cropping reframes the image, so every dimension statistic describes the region rather
#: than the source. Channel count and bit depth are untouched — a crop is a slice, not a
#: resample — and ``PIXEL``/``VISUAL`` are deliberately absent: cropping out an overlay
#: *improves* those statistics, which is the reason to do it.
_INVALIDATES = ImageStats.DIMENSION & ~(ImageStats.DIMENSION_CHANNELS | ImageStats.DIMENSION_DEPTH)


def _validate_params(region: Any) -> None:
    """Validate the crop region."""
    valid = isinstance(region, tuple) and len(region) == 4 and all(isinstance(value, int) for value in region)
    if not valid:
        raise ValueError(f"region must be a 4-tuple of ints (x0, y0, x1, y1); got {region!r}.")
    x0, y0, x1, y1 = region
    if x0 < 0 or y0 < 0:
        raise ValueError(f"region origin must be non-negative; got {region!r}.")
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"region must have positive width and height; got {region!r}.")


class Crop(Operation):
    """
    Crop a fixed pixel region out of every image, rewriting bounding boxes to match.

    Fixed regions of an image are often not data: a burned-in HUD or timestamp overlay, a
    scan border, a known-dead sensor region. They skew pixel and visual statistics and can
    dominate duplicate hashes. Cropping them away at the view level means every tool
    downstream sees only the region that is actually evidence.

    Every image is cropped to the *same* region, so this is for fixed artifacts of the
    capture pipeline, not for per-image content. To turn each detection into its own crop,
    use :class:`~dataeval.data.DetectionCrops` instead.

    Parameters
    ----------
    region : tuple[int, int, int, int]
        The region to keep, as absolute pixels ``(x0, y0, x1, y1)`` — left, top, right,
        bottom, with the right and bottom edges exclusive. Must be non-negative and have
        positive width and height.
    invalidates : ImageStats or None, default None
        Override the statistics this operation declares it invalidates. Leave as ``None``
        for the computed default: every dimension statistic except ``channels`` and
        ``depth``, which a crop preserves. ``PIXEL`` and ``VISUAL`` are deliberately not
        declared — cropping out an overlay makes those statistics *more* faithful.

    Raises
    ------
    ValueError
        At construction if ``region`` is malformed, negative, or empty. At first read if
        ``region`` extends past the image — image size is not known until then.

    See Also
    --------
    :doc:`/notebooks/h2_place_transforms` : choosing between a view operation and an extractor transform

    Examples
    --------
    Strip a 20-pixel status bar off the top of every 64x64 frame:

    >>> from dataeval.data import Crop, View
    >>> view = View(dataset, [Crop((0, 20, 64, 64))])
    >>> image, target, _ = view[0]
    >>> image.shape
    (3, 44, 64)
    """

    def __init__(self, region: tuple[int, int, int, int], *, invalidates: ImageStats | None = None) -> None:
        _validate_params(region)
        self.region = region
        self._invalidates = invalidates
        # The map is fully determined by the region, so build it once rather than per datum.
        x0, y0, x1, y1 = region
        self._mapping = GeometryMap(size=(y1 - y0, x1 - x0), offset=(-x0, -y0))

    def _repr_overrides(self) -> dict[str, str]:
        # Render the constructor's override, not the computed property the name resolves to.
        return {"invalidates": repr(self._invalidates)}

    @property
    def invalidates(self) -> ImageStats:
        """Statistics this crop makes describe the region rather than the source image."""
        return _INVALIDATES if self._invalidates is None else self._invalidates

    def apply(self, view: View[Any]) -> None:
        view.map(self._transform)

    def _transform(self, datum: Any) -> Any:
        image = normalize_image_shape(as_numpy(datum[0] if isinstance(datum, tuple) else datum))
        x0, y0, x1, y1 = self.region
        height, width = image.shape[-2], image.shape[-1]
        if x1 > width or y1 > height:
            raise ValueError(f"region {self.region} extends past the {width}x{height} (WxH) image.")

        return rewrite_geometry(datum, image[..., y0:y1, x0:x1], self._mapping)
