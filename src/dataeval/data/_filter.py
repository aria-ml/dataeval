"""Keep a dataset's items, and the detections and tracks inside them, by class and by metadata factor."""

__all__ = []

import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray
from polars.exceptions import PolarsError

from dataeval._log import get_logger
from dataeval._metadata._filters import evaluate
from dataeval._metadata._store import LevelStore
from dataeval._metadata._structurers import select_structurer
from dataeval.data._indices import _dropper, _Within
from dataeval.data._reads import Reads
from dataeval.data._video_proxies import _FrameTracksTarget
from dataeval.data._view import Operation, Transform, View
from dataeval.protocols import Array, MultiobjectTrackingTarget, ObjectDetectionTarget, SegmentationTarget
from dataeval.types import FactorLevel
from dataeval.utils._array import argmax_label, as_numpy
from dataeval.utils._mask import MaskedTarget, mask_metadata
from dataeval.utils.data import DatasetKind

_logger = get_logger(__name__)

Drop = Literal["both", "items", "detections"]
_DROPS: tuple[Drop, ...] = ("both", "items", "detections")
_NO_DETECTIONS = (
    "drop='detections' removes detections inside items, and classification items hold none. "
    "Use drop='both' or drop='items'."
)


def _predicate(  # noqa: C901
    where: str | pl.Expr | None, classes: Sequence[int | np.integer[Any]] | NDArray[np.integer[Any]] | None
) -> pl.Expr:
    """`where` and the class constraint as one polars expression."""
    parts: list[pl.Expr] = []
    if isinstance(where, str):
        try:
            parts.append(pl.sql_expr(where))
        except PolarsError as error:
            raise ValueError(f"Could not read where={where!r} as a SQL predicate: {error}") from error
    elif isinstance(where, pl.Expr):
        parts.append(where)
    elif where is not None:
        raise TypeError(f"`where` is a SQL string or a polars expression, not {type(where).__name__}.")
    if classes is not None:
        parts.append(pl.col("class_label").is_in([int(label) for label in classes]))
    return parts[0] if len(parts) == 1 else parts[0] & parts[1]


_KINDS: dict[str, DatasetKind] = {"IC": "classification", "OD": "object_detection", "MOT": "multiobject_tracking"}


def _resolve_level(  # noqa: C901
    store: LevelStore,
    predicate: pl.Expr,
    unusable: Mapping[FactorLevel, Mapping[str, Any]],
    where: str | pl.Expr | None,
) -> FactorLevel:
    """Return the coarsest level whose rows answer every column `predicate` reads.

    For a chain of levels that is the finest level any of them is defined at; for columns on sibling branches of the
    tracking diamond (a track factor and a frame factor) it is their common descendant, the detection.
    """
    names = list(dict.fromkeys(predicate.meta.root_names()))
    levels = cast("list[FactorLevel]", [level for level in store.schema if level in store.frames])
    for level in levels:
        if all(store.source_of(level, name) is not None for name in names):
            return level
    reasons: list[str] = []
    for name in names:
        readable = [level for level in levels if store.source_of(level, name) is not None]
        if readable:
            reasons.append(f"{name!r} is readable at {readable}")
        elif any(name in columns for columns in unusable.values()):
            reasons.append(f"{name!r} mixes numbers with text, so structuring held it back (see Metadata.unusable)")
        else:
            reason = (
                f"no item in the view at this point declares {name!r}: check its spelling, whether an earlier "
                "operation removed the items carrying it, and, for a per-detection factor, whether any item still "
                "holds detections"
            )
            if isinstance(where, str) and f'"{name}"' in where:
                reason += " (double quotes name a column in SQL; use single quotes for a value, e.g. tod = 'night')"
            reasons.append(reason)
    raise ValueError(
        "Filter found no level whose rows answer every column of this predicate: " + "; ".join(reasons) + "."
    )


def _rows_to_drop(
    store: LevelStore, level: FactorLevel, positions: NDArray[Any], matched: NDArray[np.bool_], kept: set[int]
) -> dict[int, _Within]:
    """Return the rows at `level` that did not match inside the kept items, filed by the item's view position."""
    keys = store.column(level, "track_id" if level == "track" else "target_index").to_numpy()
    rows: dict[int, _Within] = {}
    for position, key, match in zip(positions.tolist(), keys.tolist(), matched.tolist(), strict=True):
        if not match and position in kept:
            within = rows.setdefault(position, _Within())
            (within.tracks if level == "track" else within.instances).add(int(key))
    return rows


class _MaskToClasses:
    """The transform that drops the detections outside `classes` from one datum's target and its metadata.

    A class rather than a closure so that a view holding it pickles, which a multi-worker DataLoader needs.
    """

    def __init__(self, classes: Sequence[int]) -> None:
        self.keep = list(classes)

    def __call__(self, datum: Any) -> Any:
        """Return `datum` with only the detections whose label is kept."""
        image, target, metadata = datum
        detection_mask = np.isin(as_numpy(target.labels), self.keep)
        return image, MaskedTarget(target, detection_mask), mask_metadata(metadata, detection_mask)


class _MaskTracksToClasses:
    """The transform that drops the detections outside `classes` from every frame of one sequence, keeping the frames.

    Each frame's target is masked on its own, as ``Indices`` does, never the sequence's. A class rather than a closure
    so that a view holding it pickles, which a multi-worker DataLoader needs.
    """

    def __init__(self, classes: Sequence[int]) -> None:
        self.keep = list(classes)

    def __call__(self, datum: Any) -> Any:
        """Return `datum` with only the detections whose label is kept, frame by frame."""
        stream, target, metadata = datum
        frames: list[Any] = []
        for frame in target.frame_tracks:
            keep = np.isin(as_numpy(frame.labels).reshape(-1), self.keep)
            frames.append(frame if keep.all() else MaskedTarget(frame, keep))
        return stream, _FrameTracksTarget(frames), metadata


class Filter(Operation):
    """
    Keep the items matching a predicate over class labels and metadata factors, and the matching detections in them.

    A predicate is about the rows of one level, and the level decides what is kept: a predicate about whole items
    (an image's ``time_of_day``, a video's ``weather``) keeps or drops items, while one about rows inside items (a
    detection's ``class_label`` or ``occlusion``, a track's ``track_length``) also chooses which of those rows stay, as
    `drop` says.

    Parameters
    ----------
    where : str or pl.Expr or None, default None
        A predicate over metadata factors: a SQL ``WHERE`` clause, the form a config file holds
        (``"time_of_day = 'night' AND altitude > 100"``), or a polars boolean expression. Columns are named as
        :attr:`~dataeval.Metadata.dataframe` names them, and hold raw values, never bin codes.
    classes : Sequence[int] or NDArray[int] or None, default None
        Class labels to keep (a sequence of integers or a numpy integer array), combined (AND) with `where`:
        shorthand for ``class_label IN (...)``. Given alone, each target
        is read directly, as :class:`ClassFilter` always has, and no metadata is structured at all.
    drop : {"both", "items", "detections"}, default "both"
        What a predicate about rows inside items removes. ``"both"`` keeps the items holding a matching row and removes
        their other rows; ``"items"`` keeps the same items whole; ``"detections"`` keeps every item and removes the
        rows that do not match. The rows are detections, or whole tracks for a predicate about tracks.

    Raises
    ------
    ValueError
        If neither `where` nor `classes` is given, if `classes` is empty, if `drop` is not one of the three, or if
        `where` is not a SQL predicate polars can read. When the view is built: if no level answers every column the
        predicate reads, if the predicate cannot be evaluated on those columns (text compared with a number), if
        ``drop="detections"`` meets a predicate about whole items or a classification dataset, if a tracking predicate
        about frames is asked to remove anything, or if `where` meets a dataset whose metadata cannot be structured,
        such as segmentation.
    TypeError
        If `where` is neither a string nor a polars expression, or `classes` holds anything but integers. When the view
        is built, if `classes` alone meets a target that is not a class array, detections, segmentation or tracks.

    Notes
    -----
    **The level is the coarsest that answers every column the predicate reads.** ``time_of_day`` alone is about
    images; with ``class_label`` beside it the predicate is about detections, and an image with no detections has no
    row to match — even in ``"time_of_day = 'night' OR class_label = 0"``. ``item_index`` names an item's position in
    the view at this point, not its index in the source dataset.

    **Missing values never match.** An item that does not declare a factor holds null for it, and a NaN is read as
    null too; ``altitude IS NULL`` keeps them. Booleans are stored as integers, so write ``flag = true``, not ``flag``.
    A column name holding spaces is double-quoted, as SQL does: ``"time of day" = 'night'``.

    **Multi-object tracking.** A predicate about frames can keep the videos holding a matching frame
    (``drop="items"``) but cannot remove frames from a video: present the frames as items with :class:`SequenceFrames`
    first. A detection no tracker linked has no track, so a predicate about tracks leaves it in place; one that also
    reads a detection column removes it, and ``"track_id >= 0 AND target_index >= 0"`` removes exactly those.

    **The vocabulary is unchanged**, so a filtered view keeps every class index; :class:`Relabel` compacts it. The
    detections left behind are renumbered from 0, as after :class:`Indices`.

    **An empty result warns**, naming the predicate; a non-empty one logs what it kept on the ``dataeval`` logger at
    INFO.

    Examples
    --------
    >>> from dataeval.data import Filter, View

    Keep the images holding a car, and only their cars:

    >>> len(View(dataset, Filter(classes=[1])))
    21

    Keep the night-time images:

    >>> len(View(dataset, Filter("time_of_day = 'night'")))
    16
    """

    requires: DatasetKind | None = "any_target"

    def __init__(  # noqa: C901
        self,
        where: str | pl.Expr | None = None,
        *,
        classes: Sequence[int | np.integer[Any]] | NDArray[np.integer[Any]] | None = None,
        drop: Drop = "both",
    ) -> None:
        if where is None and classes is None:
            raise ValueError("Filter needs `where`, `classes`, or both.")
        if classes is not None:
            if len(classes) == 0:
                raise ValueError(
                    "`classes` is empty, which would keep nothing; pass classes=None for no class constraint."
                )
            wrong = [
                label
                for label in classes
                if isinstance(label, bool | np.bool_) or not isinstance(label, int | np.integer)
            ]
            if wrong:
                raise TypeError(f"`classes` holds integer class labels, not {wrong!r}.")
        if drop not in _DROPS:
            raise ValueError(f"`drop` is one of {_DROPS}, not {drop!r}.")
        self.where = where
        self.classes = classes
        self.drop: Drop = drop
        self._expr = _predicate(where, classes)

    def _repr_overrides(self) -> dict[str, str]:
        # The default repr of a polars expression carries a memory address, which would make repr(view) differ run to
        # run; its str is the expression itself.
        return {"where": repr(str(self.where))} if isinstance(self.where, pl.Expr) else {}

    def apply(self, view: View[Any]) -> None:
        # Only the deprecated ClassFilter([]) gets here with an empty `classes`, and it has always kept everything.
        if not view.selection or (self.classes is not None and not len(self.classes)):
            return
        before = len(view.selection)
        removed = self._by_target(view) if self.where is None else self._by_factors(view)
        if not view.selection:
            # Filter.apply <- View.__init__ <- the caller's View(...)
            warnings.warn(f"{self!r} kept none of the {before} items it was given.", UserWarning, stacklevel=3)
        else:
            _logger.info(
                "%r kept %d of %d items and removed %d rows inside them.", self, len(view.selection), before, removed
            )

    def _by_target(self, view: View[Any]) -> int:  # noqa: C901
        """Filter by reading each target, as ClassFilter always has: no structuring, so any data ClassFilter took.

        A read cache an earlier filter started is read and kept current, but none is started here: it would hold every
        kept item's full target (a segmentation mask, a class vector) on a view that may never build metadata.
        """
        keep = {int(label) for label in self.classes} if self.classes is not None else set()
        selection: list[int] = []
        where: set[int] = set()
        removed = 0
        mask: Transform | None = None
        old = list(view.selection)
        cache = view._reads  # held before a transform of this filter clears it
        datums = view._skeletons() if cache is not None else (view.read(index) for index in old)
        for index, datum in zip(old, datums, strict=True):
            target = datum[1]
            if isinstance(target, Array):
                if self.drop == "detections":
                    raise ValueError(_NO_DETECTIONS)
                if argmax_label(target) in keep:
                    selection.append(index)
                continue
            if isinstance(target, ObjectDetectionTarget | SegmentationTarget):
                labels = np.atleast_1d(as_numpy(target.labels)).tolist()
                mask = mask or _MaskToClasses(sorted(keep))
            elif isinstance(target, MultiobjectTrackingTarget):
                labels = [
                    label for frame in target.frame_tracks for label in np.atleast_1d(as_numpy(frame.labels)).tolist()
                ]
                mask = mask or _MaskTracksToClasses(sorted(keep))
            else:
                raise TypeError(f"{type(self).__name__} does not support targets of type {type(target)}.")
            matched = sum(label in keep for label in labels)
            if matched or self.drop == "detections":
                selection.append(index)
                if self.drop != "items" and matched < len(labels):
                    where.add(index)
                    removed += len(labels) - matched
        view.selection = selection
        transforms: dict[int, Transform] = {}
        if where and mask is not None:
            view.map(mask, where=where)
            transforms = dict.fromkeys(where, mask)
        view._keep_reads(cache, transforms)
        return removed

    def _by_factors(self, view: View[Any]) -> int:  # noqa: C901
        """Filter by structuring the view so far and evaluating the predicate at the level that answers it."""
        reads = Reads(view._skeletons(), view.metadata)
        cache = view._reads  # every selected item, held before a transform of this filter clears it
        try:
            structurer = select_structurer(reads, partial_factors=True)
        except TypeError as error:
            raise ValueError(
                f"{type(self).__name__}(where=...) filters on metadata factors, which needs a dataset "
                "dataeval.Metadata can structure: classification, object detection or multi-object tracking. "
                f"{error}"
            ) from error
        data = structurer.build(reads)
        store = LevelStore.of(structurer.levels, data)
        level = _resolve_level(store, self._expr, data.unusable, self.where)
        below = level != structurer.item_level and structurer.task != "IC"
        self._check_drop(structurer.task, level, below)
        try:
            matched = evaluate(store, level, self._expr, nan_as_null=True)
        except PolarsError as error:
            raise ValueError(f"Filter could not evaluate where={self.where!r}: {error}") from error
        positions = store.column(level, "item_index").to_numpy()
        old = list(view.selection)
        hit = set(positions[matched].tolist())
        if self.drop != "detections":
            view.selection = [old[p] for p in range(len(old)) if p in hit]
        transforms: dict[int, Transform] = {}
        removed = 0
        if below and self.drop != "items":
            kept = set(range(len(old))) if self.drop == "detections" else hit
            rows = _rows_to_drop(store, level, positions, matched, kept)
            if rows:
                drop = _dropper(_KINDS[structurer.task])
                transforms = {old[position]: drop(within) for position, within in rows.items()}
                view.map_each(transforms)
                removed = sum(len(within.instances) + len(within.tracks) for within in rows.values())
        view._keep_reads(cache, transforms)
        return removed

    def _check_drop(self, task: str, level: FactorLevel, below: bool) -> None:
        """Refuse a `drop` the predicate's level cannot honor, before anything is evaluated."""
        if task == "MOT" and level == "unit" and self.drop != "items":
            raise ValueError(
                f"This predicate is about frames, and a frame cannot be removed from a video in place, so "
                f"drop={self.drop!r} has nothing to remove. Pass drop='items' to keep the videos holding a matching "
                "frame, or present the frames as items with SequenceFrames first and filter those."
            )
        if self.drop == "detections" and not below:
            raise ValueError(
                _NO_DETECTIONS
                if task == "IC"
                else "drop='detections' keeps every item and removes the non-matching rows inside them, but this "
                "predicate is about whole items. Use drop='both' or drop='items' to filter the items."
            )
