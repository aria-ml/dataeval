"""Resize a dataset's images — and their boxes — at the view level."""

from __future__ import annotations

__all__ = []

from typing import Any, Literal, TypeAlias

from numpy.typing import NDArray

from dataeval.data._crops import FillType, resolve_fill
from dataeval.data._geometry import GeometryMap, rewrite_geometry
from dataeval.data._view import Operation, View
from dataeval.flags import ImageStats
from dataeval.utils._array import as_numpy, resize_chw
from dataeval.utils.preprocessing import crop_with_fill, normalize_image_shape

ResizeMode: TypeAlias = Literal["stretch", "pad", "crop"]

#: Every dimension statistic reports the resize target rather than the source, and
#: ``sharpness`` measures the interpolation kernel. Channel count survives; ``HASH`` is
#: deliberately absent, since resize-then-phash is a *better* near-duplicate check across
#: heterogeneous source resolutions.
_BASE_INVALIDATES = (ImageStats.DIMENSION & ~ImageStats.DIMENSION_CHANNELS) | ImageStats.VISUAL_SHARPNESS

#: Letterbox padding adds a block of flat fill pixels, which moves every percentile-derived
#: visual statistic, the pixel histogram and the entropy read off it, and every spread and
#: shape moment of the pixel distribution.
_PAD_INVALIDATES = (
    ImageStats.PIXEL_ZEROS
    | ImageStats.PIXEL_HISTOGRAM
    | ImageStats.PIXEL_ENTROPY
    | ImageStats.PIXEL_STD
    | ImageStats.PIXEL_VAR
    | ImageStats.PIXEL_SKEW
    | ImageStats.PIXEL_KURTOSIS
    | ImageStats.VISUAL_BRIGHTNESS
    | ImageStats.VISUAL_DARKNESS
    | ImageStats.VISUAL_CONTRAST
    | ImageStats.VISUAL_PERCENTILES
)

#: ``fill="mean"`` pads with the content's own per-channel mean, so the mean is preserved by
#: construction. ``fill="zero"`` is not a second-order effect: letterboxing a 2:1 source into
#: a square canvas makes half the pixels zero and halves ``mean``.
_ZERO_FILL_INVALIDATES = ImageStats.PIXEL_MEAN


def _validate_size(size: Any) -> None:
    """Validate ``size`` is a positive int, or a positive ``(height, width)`` pair of ints."""
    values = (size,) if isinstance(size, int) else size
    # A tuple must be a *pair*; a 1-tuple would pass construction and then fail on the
    # `height, width = target` unpack at first read.
    if not isinstance(values, tuple) or len(values) != (1 if isinstance(size, int) else 2):
        raise ValueError(f"size must be an int or a (height, width) tuple of ints; got {size!r}.")
    if not all(isinstance(value, int) and value > 0 for value in values):
        raise ValueError(f"size must be a positive int or a pair of positive ints; got {size!r}.")


def _validate_params(size: Any, mode: str, fill: str) -> None:
    """Validate the constructor's policy parameters."""
    if mode not in ("stretch", "pad", "crop"):
        raise ValueError(f"mode must be 'stretch', 'pad', or 'crop'; got {mode!r}.")
    if fill not in ("mean", "zero"):
        raise ValueError(f"fill must be 'mean' or 'zero'; got {fill!r}.")
    _validate_size(size)


class Resize(Operation):
    """
    Resize every image in a dataset, rewriting bounding boxes to match.

    When a deployed system runs at a fixed resolution, the source-resolution imagery is not
    what ships. This applies the resize at the *view* level, so every tool downstream --
    statistics, duplicates, outliers, embeddings -- sees the resized data, not just the
    embedding extractor.

    Parameters
    ----------
    size : int or tuple[int, int]
        Target size. A ``(height, width)`` tuple is exact. A single ``int`` sets the
        *shortest* side and preserves the source aspect ratio, so the result's other
        dimension varies with the source.
    mode : "stretch" or "pad" or "crop", default "stretch"
        How to reconcile the source aspect ratio with the target.

        - ``"stretch"`` -- scale each axis independently to hit ``size`` exactly. Distorts
          the aspect ratio.
        - ``"pad"`` -- letterbox: scale uniformly to fit inside ``size``, center the result,
          and fill the remainder with ``fill``. What essentially every detection pipeline
          actually does.
        - ``"crop"`` -- scale uniformly to cover ``size``, then center-crop the excess.
          Detections in the cropped-away margin are dropped; ones straddling the new edge
          are clipped.
    fill : "mean" or "zero", default "mean"
        Value for the letterbox bars under ``mode="pad"``; ignored otherwise. ``"mean"``
        uses the per-channel mean of the resized image, ``"zero"`` uses zeros. Shares the
        vocabulary of :class:`~dataeval.data.DetectionCrops`. The choice feeds
        ``invalidates``: mean fill leaves ``PIXEL_MEAN`` where it was by construction,
        zero fill does not -- see Notes.
    invalidates : ImageStats or None, default None
        Override the statistics this operation declares it invalidates. Leave as ``None``
        for the computed default, which is the reason this parameter exists at all -- see
        Notes.

    See Also
    --------
    :doc:`/notebooks/h2_place_transforms` : choosing between a view operation and an extractor transform

    Notes
    -----
    **Resizing at the view level asserts that source resolution is not part of your data.**
    That is a real claim about the dataset, and it is often wrong. Resolution heterogeneity
    -- a subset of images captured by a different sensor, a batch that was upscaled before
    it reached you -- is a genuine :class:`~dataeval.quality.Outliers` finding through the
    ``width``, ``height``, and ``aspect_ratio`` statistics. Resize erases it before anyone
    can see it, and every dimension statistic computed afterward describes this operation
    rather than the data. Run :class:`~dataeval.quality.Outliers` on the *unresized* view
    first if you have not already.

    ``sharpness`` is affected just as sharply and much less visibly: bilinear downsampling
    smooths, so sharpness after a resize measures the interpolation kernel. Both are
    declared in ``invalidates``, so the quality evaluators warn rather than report them as
    data findings.

    ``PIXEL_MEAN``/``PIXEL_STD`` are deliberately *not* declared for ``mode="stretch"``.
    Aggressive downsampling does shrink pixel variance, but it is a second-order effect and
    warning on every resize would make the mechanism cry wolf.

    ``mode="pad"`` is a different matter, because a letterbox replaces a large block of the
    canvas with a single flat value. Every spread and shape moment of the pixel
    distribution, the histogram, and the entropy read off it are declared there. ``mean``
    fill pads with the content's own per-channel mean, so ``PIXEL_MEAN`` survives exactly;
    ``fill="zero"`` does not get that reprieve -- letterboxing a 2:1 source into a square
    canvas makes half the pixels zero and halves the mean -- so it declares ``PIXEL_MEAN``
    as well. Boxes are clipped to the real content rather than to the whole canvas, so an
    annotation that overhangs the source frame cannot end up inside the bars.

    Images are resized bilinearly and returned as floats, so bit depth changes too.

    Examples
    --------
    Letterbox to a detector's input size, keeping boxes registered to the pixels:

    >>> from dataeval.data import Resize, View
    >>> view = View(dataset, [Resize((64, 64), mode="pad")])
    >>> image, target, _ = view[0]
    >>> image.shape
    (3, 64, 64)
    """

    def __init__(
        self,
        size: int | tuple[int, int],
        *,
        mode: ResizeMode = "stretch",
        fill: FillType = "mean",
        invalidates: ImageStats | None = None,
    ) -> None:
        _validate_params(size, mode, fill)
        self.size = size
        self.mode: ResizeMode = mode
        self.fill: FillType = fill
        self._invalidates = invalidates

    def _repr_overrides(self) -> dict[str, str]:
        # Render the constructor's override, not the computed property the name resolves to.
        return {"invalidates": repr(self._invalidates)}

    @property
    def invalidates(self) -> ImageStats:
        """Statistics this resize makes describe the transform rather than the data."""
        if self._invalidates is not None:
            return self._invalidates
        if self.mode != "pad":
            return _BASE_INVALIDATES
        zero_fill = _ZERO_FILL_INVALIDATES if self.fill == "zero" else ImageStats.NONE
        return _BASE_INVALIDATES | _PAD_INVALIDATES | zero_fill

    def apply(self, view: View[Any]) -> None:
        view.map(self._transform)

    def _target_size(self, source: tuple[int, int]) -> tuple[int, int]:
        """Resolve ``size`` against a source ``(height, width)``."""
        if not isinstance(self.size, int):
            return self.size
        height, width = source
        scale = self.size / min(height, width)
        return (self.size, round(width * scale)) if height <= width else (round(height * scale), self.size)

    def _transform(self, datum: Any) -> Any:
        image = normalize_image_shape(as_numpy(datum[0] if isinstance(datum, tuple) else datum))
        source = (image.shape[-2], image.shape[-1])
        resized, mapping = self._resize(image, self._target_size(source))
        return rewrite_geometry(datum, resized, mapping)

    def _resize(self, image: NDArray[Any], target: tuple[int, int]) -> tuple[NDArray[Any], GeometryMap]:
        """Resize ``image`` to ``target`` and return it with the map its boxes must follow."""
        height, width = target
        source_h, source_w = image.shape[-2], image.shape[-1]

        if self.mode == "stretch":
            resized = resize_chw(image, target)
            return resized, GeometryMap(size=target, scale=(width / source_w, height / source_h))

        # Uniform scale: fit inside the target and pad the remainder, or cover it and crop
        # the excess. Scale from the *rounded* content size so boxes track the real pixels.
        ratio = (
            min(width / source_w, height / source_h) if self.mode == "pad" else max(width / source_w, height / source_h)
        )
        content = (max(1, round(source_h * ratio)), max(1, round(source_w * ratio)))
        resized = resize_chw(image, content)
        scale = (content[1] / source_w, content[0] / source_h)

        if self.mode == "pad":
            dx, dy = ((width - content[1]) // 2, (height - content[0]) // 2)
            # Only the centered content is real pixels; clip boxes to it, not to the bars,
            # so an annotation overhanging the source edge cannot land in the padding.
            clip = (dx, dy, dx + content[1], dy + content[0])
            mapping = GeometryMap(size=target, scale=scale, offset=(dx, dy), clip=clip)
            return self._paste(resized, target, (dx, dy)), mapping

        offset = ((content[1] - width) // 2, (content[0] - height) // 2)
        cropped = resized[..., offset[1] : offset[1] + height, offset[0] : offset[0] + width]
        return cropped, GeometryMap(size=target, scale=scale, offset=(-offset[0], -offset[1]))

    def _paste(self, content: NDArray[Any], target: tuple[int, int], offset: tuple[int, int]) -> NDArray[Any]:
        """Center ``content`` in a ``target``-sized canvas of the fill value."""
        height, width = target
        dx, dy = offset
        # The canvas is the content window shifted by -offset, so the shared crop helper
        # builds it: real pixels pasted at the offset, everything outside taking the fill.
        window = (-dx, -dy, -dx + width, -dy + height)
        canvas, _ = crop_with_fill(
            content,
            window,
            fill=lambda pixels: resolve_fill(pixels, content.shape[0], self.fill),
            dtype=content.dtype,
        )
        return canvas
