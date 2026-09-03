"""Narrow or combine an image's channels at the view level."""

from __future__ import annotations

__all__ = []

from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from dataeval.data._view import Operation, View
from dataeval.flags import ImageStats
from dataeval.utils._array import as_numpy
from dataeval.utils.preprocessing import _validate_index_selection, normalize_image_shape

ChannelSelection: TypeAlias = "Sequence[int] | Literal['gray', 'rgb']"

#: Rec. 601 luma coefficients, matching
#: :func:`~dataeval.utils.preprocessing.to_canonical_grayscale`.
_REC601 = np.array([0.299, 0.587, 0.114], dtype=np.float64)


def _validate_indices(channels: Any) -> None:
    """Validate an explicit channel-index selection.

    Defers to ``_validate_index_selection``, which asks what `ChannelGroup` asks of the
    same value — a non-empty selection of non-negative ints, with ``bool`` rejected because
    numpy reads a list of bools as a mask. Only that half is shared: a selection here may
    repeat and reorder bands, which a group reduced over jointly may not.
    """
    try:
        _validate_index_selection(channels)
    except ValueError as e:
        raise ValueError(
            f"channels must be a non-empty sequence of non-negative ints, 'gray', or 'rgb'; got {channels!r}."
        ) from e


def _validate_params(channels: Any) -> None:
    """Validate the channel selection."""
    if isinstance(channels, str):
        if channels not in ("gray", "rgb"):
            raise ValueError(f"channels must be a sequence of indices, 'gray', or 'rgb'; got {channels!r}.")
        return
    _validate_indices(channels)


class SelectChannels(Operation):
    """
    Narrow each image to a chosen set of channels, or combine them into one.

    Two cases this exists for. A monochrome sensor whose output was stored as 3-channel
    RGB triples the work of every statistic and distorts any channel-wise analysis, when
    the three channels carry one channel's worth of information. And in multispectral
    imagery, only some bands are usually the data under evaluation -- the ones the model
    consumes.

    Geometry is untouched, so targets pass through byte-identical: a channel selection
    never moves, drops, or reshapes a bounding box.

    Parameters
    ----------
    channels : Sequence[int] or "gray" or "rgb"
        Which channels to keep.

        - A sequence of indices selects those channels, **in the order given**, so
          ``[2, 1, 0]`` reverses BGR to RGB. Indices may repeat.
        - ``"gray"`` mixes a 3-channel image down to one via Rec. 601 luma weights, and
          passes a 1-channel image through.
        - ``"rgb"`` broadcasts a 1-channel image up to three identical channels, and
          passes a 3-channel image through. For a mono-stored-as-RGB source this is the
          no-op half of the pair; reach for ``"gray"`` to actually collapse it.
    invalidates : ImageStats or None, default None
        Override the statistics this operation declares it invalidates. Leave as ``None``
        for the computed default -- see Notes.

    Raises
    ------
    ValueError
        At construction if ``channels`` is malformed. At first read if ``"gray"`` or
        ``"rgb"`` meets a source that is neither 1- nor 3-channel, since the intended
        mapping is undefined there.
    IndexError
        At first read if an index exceeds the source's channel count.

    See Also
    --------
    :doc:`/notebooks/h2_place_transforms` : choosing between a view operation and an extractor transform

    Notes
    -----
    Selecting channels invalidates ``channels`` and nothing else: the surviving channels'
    pixels are untouched, so pixel and visual statistics still describe the data.

    ``"gray"`` is different, and is the clearest illustration of why ``invalidates`` is
    computed per-instance rather than fixed per-class. A luminance mix is a new pixel
    value at every position, so it moves ``PIXEL_MEAN``, ``PIXEL_STD``, and every visual
    statistic -- all of which the ``"gray"`` form therefore declares.

    Examples
    --------
    Collapse a mono sensor that was stored as RGB:

    >>> from dataeval.data import SelectChannels, View
    >>> view = View(dataset, [SelectChannels("gray")])
    >>> view[0][0].shape
    (1, 64, 64)

    Keep three bands of a multispectral cube, in model order:

    >>> view = View(dataset, [SelectChannels([2, 1, 0])])
    >>> view[0][0].shape
    (3, 64, 64)
    """

    def __init__(self, channels: ChannelSelection, *, invalidates: ImageStats | None = None) -> None:
        _validate_params(channels)
        self.channels: ChannelSelection = channels if isinstance(channels, str) else list(channels)
        self._invalidates = invalidates

    def _repr_overrides(self) -> dict[str, str]:
        # Render the constructor's override, not the computed property the name resolves to.
        return {"invalidates": repr(self._invalidates)}

    @property
    def invalidates(self) -> ImageStats:
        """Statistics this selection makes describe the transform rather than the data."""
        if self._invalidates is not None:
            return self._invalidates
        if self.channels == "gray":
            return ImageStats.DIMENSION_CHANNELS | ImageStats.PIXEL | ImageStats.VISUAL
        return ImageStats.DIMENSION_CHANNELS

    def apply(self, view: View[Any]) -> None:
        view.map(self._transform)

    def _transform(self, datum: Any) -> Any:
        is_tuple = isinstance(datum, tuple) and len(datum) == 3
        image = self._select(normalize_image_shape(as_numpy(datum[0] if is_tuple else datum)))
        return (image, datum[1], datum[2]) if is_tuple else image

    def _select(self, image: NDArray[Any]) -> NDArray[Any]:
        """Apply the channel selection to one CHW image."""
        channels = image.shape[-3]
        if self.channels == "gray":
            return image if channels == 1 else self._to_gray(self._require_rgb(image, channels, "gray"))
        if self.channels == "rgb":
            return np.repeat(image, 3, axis=0) if channels == 1 else self._require_rgb(image, channels, "rgb")

        indices = list(self.channels)
        if max(indices) >= channels:
            raise IndexError(f"channel index {max(indices)} is out of range for a {channels}-channel image.")
        return image[indices]

    @staticmethod
    def _require_rgb(image: NDArray[Any], channels: int, mode: str) -> NDArray[Any]:
        """Reject a source whose channel count leaves ``mode`` undefined."""
        if channels != 3:
            raise ValueError(
                f"channels={mode!r} needs a 1- or 3-channel image; got {channels} channels. "
                "Pass explicit indices to choose from a source with a different channel count."
            )
        return image

    @staticmethod
    def _to_gray(image: NDArray[Any]) -> NDArray[Any]:
        """Mix RGB down to one channel, keeping the source dtype."""
        mixed = np.tensordot(_REC601, image.astype(np.float64), axes=1)[np.newaxis, ...]
        return (np.rint(mixed) if np.issubdtype(image.dtype, np.integer) else mixed).astype(image.dtype)
