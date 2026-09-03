"""Shared geometry rewrite for view operations that change an image's frame.

An operation that resizes or crops an image must carry its targets along: bounding boxes
scale and shift with the frame, detections that fall outside the new frame disappear, and
every per-detection array in the datum's metadata has to shrink in step or it silently
misaligns with the boxes it describes. That work is identical for ``Crop`` and for
``Resize`` in its letterbox and center-crop modes, so it lives here once.
"""

__all__ = []

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval.protocols import ObjectDetectionTarget, SegmentationTarget
from dataeval.utils._array import as_numpy
from dataeval.utils._mask import MaskedTarget, mask_metadata


@dataclass(frozen=True)
class GeometryMap:
    """An axis-aligned map from a source frame into a destination frame.

    Every geometric view operation this covers is the same two steps: scale about the
    origin, then translate, into a destination canvas that the result is clipped to. A
    stretch resize is scale alone; a letterbox is scale plus a positive offset for the pad;
    a fixed-region crop is a negative offset alone; a center-crop resize is both.

    Parameters
    ----------
    size : tuple[int, int]
        Destination frame as ``(height, width)``, in pixels. Boxes are clipped to it.
    scale : tuple[float, float], default (1.0, 1.0)
        Per-axis scale factor as ``(x, y)``. Must be positive — a non-positive scale would
        reflect the frame, which reorders a box's corners and is not something any
        operation here performs.
    offset : tuple[float, float], default (0.0, 0.0)
        Per-axis translation as ``(x, y)``, in *destination* pixels, applied after the
        scale. Positive pads (letterbox), negative crops.
    clip : tuple[float, float, float, float] or None, default None
        Destination rectangle ``(x0, y0, x1, y1)`` that mapped boxes are clipped to.
        ``None`` clips to the whole ``size`` canvas, which is right whenever every
        destination pixel came from the source. A letterbox does not satisfy that: only
        the centered content is real, the bars are synthetic fill, so ``Resize`` passes
        the content rectangle here. Without it a source annotation that overhangs the
        image edge would be clipped into the padding instead of to the real pixels.

    Examples
    --------
    Halve a 20x20 image, so a box halves with it:

    >>> import numpy as np
    >>> from dataeval.data._geometry import GeometryMap
    >>> boxes, mask = GeometryMap(size=(10, 10), scale=(0.5, 0.5)).apply_boxes(np.array([[2.0, 2.0, 6.0, 6.0]]))
    >>> boxes.tolist()
    [[1.0, 1.0, 3.0, 3.0]]
    """

    size: tuple[int, int]
    scale: tuple[float, float] = (1.0, 1.0)
    offset: tuple[float, float] = (0.0, 0.0)
    clip: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if any(s <= 0 for s in self.scale):
            raise ValueError(f"scale must be positive on both axes; got {self.scale}.")
        if any(s <= 0 for s in self.size):
            raise ValueError(f"size must be positive on both axes; got {self.size}.")
        if self.clip is not None and (self.clip[2] <= self.clip[0] or self.clip[3] <= self.clip[1]):
            raise ValueError(f"clip must have positive width and height; got {self.clip}.")

    def apply_boxes(self, boxes: NDArray[Any]) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """Map XYXY boxes into the destination frame, clipping to it and dropping empties.

        Parameters
        ----------
        boxes : NDArray
            Source boxes, ``(N, 4)`` in absolute-pixel ``[x0, y0, x1, y1]``.

        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.bool_]]
            The surviving boxes, ``(M, 4)``, and the length-``N`` keep mask that selected
            them. A detection is dropped when the clipped box has no area — it fell wholly
            outside the new frame, or the frame left only its edge.
        """
        height, width = self.size
        x0, y0, x1, y1 = self.clip if self.clip is not None else (0.0, 0.0, float(width), float(height))
        scaled = boxes.reshape(-1, 4).astype(np.float64) * (*self.scale, *self.scale) + (*self.offset, *self.offset)
        clipped = np.clip(scaled, (x0, y0, x0, y0), (x1, y1, x1, y1))
        keep = (clipped[:, 2] > clipped[:, 0]) & (clipped[:, 3] > clipped[:, 1])
        return clipped[keep], keep


def rewrite_geometry(datum: Any, image: Any, mapping: GeometryMap) -> Any:
    """Return ``datum`` with ``image`` substituted and its target and metadata remapped.

    The caller owns the pixel transform — it produces ``image`` however it likes — and this
    carries the annotations across the same ``mapping``, so the two cannot drift apart.

    Parameters
    ----------
    datum : Any
        The source datum: a MAITE ``(image, target, metadata)`` 3-tuple, or a bare image.
    image : Any
        The already-transformed image to put in the datum's place.
    mapping : GeometryMap
        The same geometric map that produced ``image`` from ``datum``'s image.

    Returns
    -------
    Any
        A datum of the same shape as the input. For an object-detection target, the boxes
        are transformed and clipped, out-of-frame detections are dropped, and every
        per-detection array in the metadata is masked to match. An image-classification
        target and its metadata pass through unchanged — a class label survives any
        reframing. A bare-image datum yields ``image`` itself.

    Raises
    ------
    NotImplementedError
        For a segmentation target. Carrying a mask across the frame needs a pixel-space
        resample of the mask itself, which is the caller's transform to make, not this
        helper's — rather than silently returning a mask that no longer registers with its
        image, this refuses.

    Notes
    -----
    Dropped detections route through ``MaskedTarget`` and ``mask_metadata`` in
    ``dataeval.utils._internal``, with the transformed boxes supplied as an ``overrides``
    entry — the same pattern :class:`~dataeval.data.Relabel` uses for remapped labels.
    """
    if not isinstance(datum, tuple) or len(datum) != 3:
        return image

    _, target, metadata = datum
    if isinstance(target, SegmentationTarget):
        raise NotImplementedError(
            "Geometric view operations do not support segmentation targets yet: the mask "
            "would have to be resampled alongside the image to stay registered with it."
        )
    if not isinstance(target, ObjectDetectionTarget):
        return image, target, metadata

    boxes, mask = mapping.apply_boxes(as_numpy(target.boxes))
    # mask_metadata walks and rebuilds the whole metadata mapping; when the frame kept
    # every detection there is nothing to drop, so pass the datum's own mapping through.
    remapped = metadata if mask.all() else mask_metadata(metadata, mask)
    return image, MaskedTarget(target, mask, {"boxes": boxes}), remapped
