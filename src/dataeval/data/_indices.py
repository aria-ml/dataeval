__all__ = []

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dataeval.data._video_proxies import _FrameTracksTarget
from dataeval.data._view import Operation, Transform, View
from dataeval.types import SourceIndex
from dataeval.types._removal import canonical_address
from dataeval.types._target import detection_count, track_ids_of
from dataeval.utils._mask import MaskedTarget, mask_metadata
from dataeval.utils.data import DatasetKind, validate_dataset


@dataclass
class _Within:
    """The rows one item loses below the item level: detections by ``target_index``, tracks by ``track_id``."""

    instances: set[int] = field(default_factory=set)
    tracks: set[int] = field(default_factory=set)

    def add(self, address: SourceIndex, key: int) -> None:
        """File a keyed address under the rows it removes, refusing rows this operation cannot remove."""
        if address.kind == "instance":
            self.instances.add(key)
        elif address.kind == "track" and key >= 0:
            self.tracks.add(key)
        else:
            raise ValueError(_unsupported(address))


def _unsupported(address: SourceIndex) -> str:
    """Say why `address` names a row Indices cannot remove."""
    if address.kind == "track":
        return f"{address!r} names track -1, which marks detections no tracker linked rather than a track."
    if address.kind == "unit":
        return (
            f"{address!r} names a frame, and Indices cannot remove frames yet. Leave frame addresses out of a plan "
            'with RemovalPlan(a for a in plan if a.kind != "unit").'
        )
    return f"{address!r} names a {address.kind} row inside an item, which Indices cannot remove."


def _split(indices: Iterable[int | SourceIndex]) -> tuple[list[int], dict[int, _Within]]:
    """Separate whole items, in the order given, from the rows named inside items."""
    items: list[int] = []
    within: dict[int, _Within] = {}
    for index in indices:
        address = canonical_address(index)
        if address.key is None:
            items.append(address.item)
        else:
            within.setdefault(address.item, _Within()).add(address, address.key)
    return items, within


def _required_kind(within: Mapping[int, _Within]) -> DatasetKind | None:
    """Name the dataset kind the named rows need: tracks need tracking data, detections need any annotated data."""
    if any(rows.tracks for rows in within.values()):
        return "multiobject_tracking"
    return "any_target" if within else None


class _DropDetections:
    """The transform that drops the named detections from one image's target and its metadata.

    A class rather than a closure so that a view holding it pickles, which a multi-worker DataLoader needs.
    """

    def __init__(self, rows: _Within) -> None:
        self.named = np.fromiter(rows.instances, dtype=np.intp)

    def __call__(self, datum: Any) -> Any:
        """Return `datum` without the named detections."""
        image, target, metadata = datum
        keep = ~np.isin(np.arange(detection_count(target)), self.named)
        return image, MaskedTarget(target, keep), mask_metadata(metadata, keep)


class _DropTracked:
    """The transform that drops the named detections and tracks from one sequence, frame by frame.

    A detection's ``target_index`` counts the sequence's detections in frame order, so each frame's share starts
    where the previous frame's ended. Each frame's target is masked on its own, never the sequence's target, whose
    ``frame_tracks`` a mask of matching length would otherwise cut. A class rather than a closure so that a view
    holding it pickles, which a multi-worker DataLoader needs.
    """

    def __init__(self, rows: _Within) -> None:
        self.named = np.fromiter(rows.instances, dtype=np.intp)
        self.tracks = np.fromiter(rows.tracks, dtype=np.intp)

    def __call__(self, datum: Any) -> Any:
        """Return `datum` without the named detections and tracks, keeping every frame."""
        stream, target, metadata = datum
        frames: list[Any] = []
        start = 0
        for frame in target.frame_tracks:
            count = detection_count(frame)
            keep = ~np.isin(np.arange(start, start + count), self.named)
            keep &= ~np.isin(track_ids_of(frame, count), self.tracks)
            frames.append(frame if keep.all() else MaskedTarget(frame, keep))
            start += count
        return stream, _FrameTracksTarget(frames), metadata


_DROPPERS: dict[DatasetKind, Callable[[_Within], Transform]] = {
    "object_detection": _DropDetections,
    "segmentation": _DropDetections,
    "multiobject_tracking": _DropTracked,
}


def _dropper(kind: DatasetKind) -> Callable[[_Within], Transform]:
    """Choose how the named rows are dropped from one datum of a dataset of `kind`."""
    if kind not in _DROPPERS:
        raise ValueError(
            f"Indices removes rows inside items from detection, segmentation and tracking datasets; this is a "
            f"{kind} dataset, whose items hold no detections to remove."
        )
    return _DROPPERS[kind]


class Indices(Operation):
    """
    Keep the given items, or remove the given items and the rows named inside items when `exclude` is set.

    Parameters
    ----------
    indices : Iterable[int | SourceIndex]
        What to keep or remove. An int, or an unkeyed :class:`~dataeval.types.SourceIndex`, names a whole item
        by its index in the dataset this view wraps. With `exclude`, a keyed address names a row inside an item:
        a detection by its ``target_index``, or, on a tracking dataset, a track by its ``track_id``. A
        :class:`~dataeval.types.RemovalPlan` is accepted as it is.
    exclude : bool, default False
        If True, remove what `indices` names, keeping the remaining items in order. If False, keep only the named
        items, in the order given.

    Attributes
    ----------
    indices : Iterable[int | SourceIndex]
        What was passed, kept for the representation. A one-shot iterator is read into a list.
    exclude : bool
        Whether the named rows are removed rather than kept.
    requires : DatasetKind or None
        ``None`` when only whole items are named, ``"any_target"`` when detections are, and
        ``"multiobject_tracking"`` when tracks are.

    Raises
    ------
    ValueError
        If `exclude` is False and `indices` names a row inside an item, or if an address names a frame or track
        ``-1``. When the view is built, if detections are named on a dataset that holds none.
    TypeError
        If an entry of `indices` is neither an int nor a SourceIndex.
    MaiteShapeError
        When the view is built, if a track is named on a dataset that is not a tracking dataset.

    Notes
    -----
    **Apply a plan to the dataset it was computed on.** An evaluator reports positions in the dataset it read,
    and those positions are what this operation compares against, so build the view on that same dataset:
    ``View(evaluated, Indices(plan, exclude=True))``. Applied to the dataset underneath a filtered view, the same
    numbers name different items.

    **Apply several plans as one.** Combine them with ``|`` into one ``Indices``, and place it before any
    operation that removes or relabels detections. A detection's ``target_index`` counts the detections its item
    holds when ``Indices`` reads it, so once an earlier operation, or an earlier ``Indices``, has removed a
    detection, the same number names a different one. Item addresses are not affected, since an item keeps its
    index in the dataset the view wraps.

    **Removing an item removes every row in it.** A detection named inside a removed item has no further effect.
    Removing every detection in an image or a frame leaves the image or frame, since an unlabeled one is valid
    data.

    **A segmentation label map is kept whole.** A mask with one plane per detection, shape ``(N, H, W)``, loses
    the removed detections' planes. A label map, shape ``(H, W)``, keeps every pixel, since its pixels are not
    divided by detection.

    **Detections are renumbered.** :class:`~dataeval.Metadata` built over the view numbers the remaining
    detections from 0, within each image, and across each whole sequence on a tracking dataset, so an address
    from before the removal does not name the same detection afterwards.

    **Rows an item does not hold are ignored**, as items outside the dataset are.

    **Keep a plan's items to see them.** Without `exclude`, a plan of whole items keeps exactly the items it
    names, which shows what would be removed: ``View(dataset, Indices(plan))``.

    Examples
    --------
    >>> from dataeval.data import Indices, View
    >>> from dataeval.types import SourceIndex

    Keep items 3, 1 and 0, in that order:

    >>> View(dataset, Indices([3, 1, 0])).selection
    [3, 1, 0]

    Remove item 0, and the first detection of item 1:

    >>> view = View(dataset, Indices([0, SourceIndex(1, 0)], exclude=True))
    >>> len(view) == len(dataset) - 1
    True
    """

    requires: DatasetKind | None = None

    def __init__(self, indices: Iterable[int | SourceIndex], exclude: bool = False) -> None:
        self.indices: Iterable[int | SourceIndex] = list(indices) if isinstance(indices, Iterator) else indices
        self.exclude: bool = bool(exclude)
        self._items, self._within = _split(self.indices)
        if self._within and not self.exclude:
            raise ValueError(
                "Indices keeps whole items only, in the order given, which has no meaning for rows inside an item. "
                "Pass exclude=True to remove the rows the keyed addresses name."
            )
        self.requires = _required_kind(self._within)

    def apply(self, view: View[Any]) -> None:
        if not self.exclude:
            current = set(view.selection)
            view.selection = [index for index in self._items if index in current]
            return
        removed = set(self._items)
        view.selection = [index for index in view.selection if index not in removed]
        within = {item: rows for item, rows in self._within.items() if item not in removed}
        if within and len(view.source):
            drop = _dropper(validate_dataset(view.source, expected="any_target", caller="Indices"))
            view.map_each({item: drop(rows) for item, rows in within.items()})
