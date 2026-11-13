"""
Per-image annotations that keep what box conversion destroys.

`to_bounding_box` swaps reversed edges, turns NaN into a finite zero-area box and turns input that is not four
numbers into ``(0, 0, 0, 0)``. A statistic that has to decide whether a box can be trusted needs the coordinates as
given, so conversion here carries them beside the converted box, in :class:`Annotations`, and
:class:`AnnotationContext` classifies them once per image.

Kinds. Each box gets exactly one :class:`Kind`, and the first match wins in this order: ``NONFINITE``, ``REVERSED``
(finite, with ``x1 < x0`` or ``y1 < y0``), ``ZERO_AREA`` (finite, ordered, ``x1 == x0`` or ``y1 == y0``), ``OUTSIDE``
(positive area, ``x1 <= 0``, ``y1 <= 0``, ``x0 >= W`` or ``y0 >= H``, the rule `BoundingBox.is_outside` uses, so a box
that only touches the edge is outside), else ``VALID``.

Status. ``AnnotationContext.status`` is ``"invalid_box"`` when the focal box's own kind is not ``VALID``, else
``"incomplete_geometry"`` when any other box in the frame is ``NONFINITE`` or ``REVERSED``, else ``"ok"``. The focal's
own defect wins over a peer's. Zero-area and outside peers are skipped and counted, and never change the status.

Three meanings of "invalid" differ and are not interchangeable. `BoundingBox.is_valid` is what
``DIMENSION_INVALID_BOX`` reads, and `BoundingBox.is_clippable` is what ``invalid_box_count`` counts; both read the
converted box. The status ``invalid_box`` reads the raw coordinates. An XYXY box with reversed edges is swapped by its
constructor, so it is valid for the first two and ``REVERSED``, hence ``invalid_box``, for the third.

An image with no annotation gets an empty context (``N = 0``) rather than None, so a consumer never branches on it. A
datum with no spatial extent has extent ``(0, 0)``, which makes every box ``OUTSIDE``. A ``None`` entry inside a box
list is the full-image box, as `to_bounding_box` has always made it.
"""

__all__ = []

from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum
from functools import cached_property
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dataeval.utils.preprocessing import BoundingBox, BoxLike, boxes_to_mask, to_bounding_box


class Kind(IntEnum):
    """What a box's raw coordinates make it. Stored as `int8` codes in :attr:`AnnotationContext.kind`."""

    VALID = 0
    ZERO_AREA = 1
    OUTSIDE = 2
    REVERSED = 3
    NONFINITE = 4


@dataclass(frozen=True, eq=False)
class Annotations:
    """
    One image's boxes, as converted and as given.

    ``eq=False`` because ``==`` on the array fields is ambiguous.

    Attributes
    ----------
    boxes : list[BoundingBox]
        The converted boxes, which every existing statistic reads.
    raw : NDArray[np.float64]
        The coordinates as given, ``(N, 4)`` in x0, y0, x1, y1 order. NaN, infinity and reversed edges survive here.
    labels : NDArray[np.int64] or None
        One class id per box, or None where no labels were given.
    """

    boxes: list[BoundingBox]
    raw: NDArray[np.float64]
    labels: NDArray[np.int64] | None = None


_ROW_FIX = "each box is one row of four numbers, so drop any extra column and pass class ids through labels="
_LIST_FIX = "boxes= takes one list of boxes per image, each box four numbers: [[(x0, y0, x1, y1), ...], ...]"


def _bad_row(item: int, box: int, got: str, fix: str = _ROW_FIX) -> ValueError:
    return ValueError(f"image {item}, box {box}: expected 4 values (x0, y0, x1, y1), got {got}; {fix}.")


def _four_numbers(entry: Any, item: int, box: int) -> tuple[float, float, float, float]:
    # `BoxLike` includes one-shot iterables, so the entry is read once here and both the raw row and the converted box
    # are made from what this returns. Reading it twice would leave the second reader an empty iterator.
    try:
        values = tuple(entry)
    except TypeError:
        # A number where a box belongs is a list of boxes missing a level, not a box with an extra column.
        raise _bad_row(item, box, f"a single {type(entry).__name__} ({entry!r})", _LIST_FIX) from None
    if len(values) != 4:
        raise _bad_row(item, box, f"{len(values)} values")
    try:
        return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))
    except (TypeError, ValueError):
        raise _bad_row(item, box, f"non-numeric values {values!r}") from None


def _refuse_labels(array: NDArray[Any], count: int, item: int) -> None:
    """Raise unless `array` is one integer class id per box, saying which part is wrong and how to fix it."""
    if array.ndim != 1:
        raise ValueError(
            f"image {item}: labels must be a 1-D sequence with one class id per box, got shape {array.shape}."
        )
    if len(array) != count:
        raise ValueError(f"image {item}: labels holds {len(array)} class ids for {count} boxes; pass one per box.")
    if array.dtype.kind not in "iu":
        raise ValueError(
            f"image {item}: labels must be integer class ids, got dtype {array.dtype}; "
            "cast whole-number labels with .astype(int)."
        )


def _label_array(labels: ArrayLike | None, count: int, item: int) -> NDArray[np.int64] | None:
    if labels is None:
        return None
    array = np.asarray(labels)
    if array.size == 0 and count == 0:
        return np.empty(0, dtype=np.int64)
    _refuse_labels(array, count, item)
    return array.astype(np.int64)


def build_annotations(
    entries: Iterable[BoxLike] | None,
    image_shape: tuple[int, ...],
    item: int,
    labels: ArrayLike | None = None,
) -> Annotations:
    """
    Convert one image's box entries, keeping the coordinates as given.

    Parameters
    ----------
    entries : Iterable[BoxLike] or None
        The image's boxes. None, an empty iterable and an empty array all mean no boxes. An entry that is None is the
        full-image box.
    image_shape : tuple[int, ...]
        Shape of the image the boxes belong to.
    item : int
        The image's index, named in every error.
    labels : ArrayLike or None, default None
        One integer class id per entry, in the same order.

    Returns
    -------
    Annotations
        The converted boxes, their raw coordinates and the labels.

    Raises
    ------
    ValueError
        If an entry is not four numbers, or the labels are not one integer per box.
    """
    rows = [] if entries is None else list(entries)
    height, width = (image_shape[-2], image_shape[-1]) if len(image_shape) >= 2 else (0, 0)
    boxes: list[BoundingBox] = []
    raw = np.empty((len(rows), 4), dtype=np.float64)
    for j, entry in enumerate(rows):
        if isinstance(entry, BoundingBox):
            # Already converted by its own constructor, which swapped reversed XYXY edges before this module sees it:
            # the raw row is what the box now holds.
            box, raw[j] = entry, entry.xyxy
        elif entry is None:
            box, raw[j] = to_bounding_box(None, image_shape), (0, 0, width, height)
        else:
            values = _four_numbers(entry, item, j)
            box, raw[j] = to_bounding_box(values, image_shape), values
        if not np.isfinite(box.xyxy).all():
            # A nonfinite box cannot be cropped: `xyxy_int` raises on NaN and on infinity. The zero box is what a row
            # that is not four numbers became before, so legacy statistics see an invalid box; `raw` keeps the truth.
            box = BoundingBox(0, 0, 0, 0, image_shape=image_shape)
        boxes.append(box)
    return Annotations(boxes, raw, _label_array(labels, len(rows), item))


class AnnotationContext:
    """
    One image's boxes classified once, for statistics that compare a box with its peers.

    Parameters
    ----------
    annotations : Annotations or None
        The image's annotations, or None for an image with none.
    extent : tuple[int, int]
        The image's ``(H, W)``, or ``(0, 0)`` for a datum with no spatial extent.

    Attributes
    ----------
    raw : NDArray[np.float64]
        The coordinates as given, ``(N, 4)``.
    kind : NDArray[np.int8]
        The :class:`Kind` code of each box, ``(N,)``.
    clipped : NDArray[np.float64]
        The boxes clipped to ``[0, W] x [0, H]``, ``(N, 4)``. Not rounded, so fractional edges survive. Meaningful for
        ``VALID`` boxes; the other kinds are clipped coordinate by coordinate.
    labels : NDArray[np.int64] or None
        One class id per box, or None where none were given.
    coverage : NDArray[np.bool_]
        An ``(H, W)`` mask that is True where any ``VALID`` box covers the pixel, each box rounded outward and clipped
        as :func:`~dataeval.utils.preprocessing.boxes_to_mask` does. Built once per image, so a statistic slices it per
        box.

    Notes
    -----
    The arrays are computed on first read, so a statistic that reads only ``raw`` pays for nothing else. The methods
    take a focal box index; on an image's own row there is no focal box and a statistic answers NaN without asking.
    """

    def __init__(self, annotations: Annotations | None, extent: tuple[int, int]) -> None:
        self._annotations = annotations
        self.extent = extent
        self.labels: NDArray[np.int64] | None = None if annotations is None else annotations.labels

    @cached_property
    def raw(self) -> NDArray[np.float64]:
        return np.empty((0, 4), dtype=np.float64) if self._annotations is None else self._annotations.raw

    @cached_property
    def kind(self) -> NDArray[np.int8]:
        height, width = self.extent
        x0, y0, x1, y1 = self.raw.T
        kind = np.full(len(self.raw), Kind.VALID, dtype=np.int8)
        # Assigned from the lowest precedence up, so the first match in the documented order is what remains. An extent
        # with no pixels has nothing a box can be inside: the edge rules alone miss a box that spans the origin.
        kind[(x1 <= 0) | (y1 <= 0) | (x0 >= width) | (y0 >= height) | (width <= 0) | (height <= 0)] = Kind.OUTSIDE
        kind[(x1 == x0) | (y1 == y0)] = Kind.ZERO_AREA
        kind[(x1 < x0) | (y1 < y0)] = Kind.REVERSED
        kind[~np.isfinite(self.raw).all(axis=1)] = Kind.NONFINITE
        return kind

    @cached_property
    def clipped(self) -> NDArray[np.float64]:
        height, width = self.extent
        clipped = self.raw.copy()
        clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], 0, width)
        clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], 0, height)
        return clipped

    @cached_property
    def coverage(self) -> NDArray[np.bool_]:
        return boxes_to_mask(self.extent, self.raw[self.kind == Kind.VALID])

    def _check(self, focal: int) -> None:
        """Refuse anything that is not the index of one of the boxes, so a missing focal cannot read as a status."""
        count = len(self.kind)
        # `isinstance` and not a None test: the image row's focal is None, which a consumer must handle first, and a
        # negative index would wrap to the last box and answer for the wrong one.
        if not isinstance(focal, int | np.integer) or not 0 <= focal < count:
            raise ValueError(
                f"focal must be the index of one of this image's {count} boxes, got {focal!r}; "
                "the image row has no focal box."
            )

    def status(self, focal: int) -> Literal["ok", "invalid_box", "incomplete_geometry"]:
        """
        Whether the focal box can be trusted, and whether the frame around it is complete.

        Parameters
        ----------
        focal : int
            Index of the focal box.

        Returns
        -------
        {"ok", "invalid_box", "incomplete_geometry"}
            ``invalid_box`` when the focal box is not ``VALID``, else ``incomplete_geometry`` when another box in the
            frame is ``NONFINITE`` or ``REVERSED``, else ``ok``.
        """
        self._check(focal)
        kind = self.kind
        if kind[focal] != Kind.VALID:
            return "invalid_box"
        return "incomplete_geometry" if (kind >= Kind.REVERSED).any() else "ok"

    def peers(self, focal: int) -> NDArray[np.intp]:
        """Return the indices of the ``VALID`` boxes other than `focal`, in index order."""
        self._check(focal)
        valid = np.flatnonzero(self.kind == Kind.VALID)
        return valid[valid != focal]

    def skipped(self, focal: int) -> tuple[int, int]:
        """Return the numbers of ``ZERO_AREA`` and ``OUTSIDE`` boxes other than `focal`, which are not its peers."""
        self._check(focal)
        others = np.arange(len(self.kind)) != focal
        return int(((self.kind == Kind.ZERO_AREA) & others).sum()), int(((self.kind == Kind.OUTSIDE) & others).sum())
