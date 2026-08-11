"""Pluggable structuring strategies that turn a dataset into levelled metadata rows.

The core :class:`~dataeval.Metadata` engine is task agnostic: it consumes a
:class:`StructuredData` bundle and never inspects the dataset itself. Everything
that depends on what a dataset item is (e.g. image, or video sequence) and
where the labels sit (e.g. image, or instance) lives in a :class:`Structurer`.
"""

__all__ = [
    "RESERVED_COLUMNS",
    "TASK",
    "DatasetStructurer",
    "FactorsStructurer",
    "ICStructurer",
    "MOTStructurer",
    "ODImageStructurer",
    "RowBlock",
    "RowLayout",
    "StructuredData",
    "Structurer",
    "TaskOverride",
    "reserved_block_columns",
    "safe_column_name",
    "select_structurer",
]

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Container, Iterable, Iterator, Mapping, Sequence, Sized
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray

from dataeval.protocols import (
    AnnotatedDataset,
    Array,
    DatumMetadata,
    ObjectDetectionTarget,
    ProgressCallback,
    SingleFrameObjectTrackingTarget,
    is_multiobject_tracking_target,
)
from dataeval.types import FactorLevel, FactorLevelSchema
from dataeval.utils._internal import as_numpy, merge_metadata

_logger = logging.getLogger(__name__)

# Task identifier of a structuring strategy.
TASK = Literal["IC", "OD", "MOT", "factors", "unknown"]


# Columns the metadata dataframe has always carried. Retained verbatim because
# they are public surface: downstream code reads ``dataframe["item_index"]``.
LEGACY_COLUMNS: tuple[str, ...] = ("item_index", "target_index", "class_label", "score", "box")

# Columns introduced alongside the expanded level schema. ``level`` tags each row
# with the level it belongs to and replaces the old "target_index is null" test;
# the remaining columns carry the components of a level's compound key. Only
# columns a structurer actually writes belong here — a name reserved for a level
# that does not exist yet costs a user their metadata key for nothing.
#
# ``image_index``, ``track_index`` and ``sequence_index`` exist for multi-object tracking,
# the only task whose item level is not ``image``: there a frame's position within its
# sequence is not ``item_index`` (that identifies the video) and has nowhere else to go.
# ``image_index`` and ``track_index`` are written on the tracking task's instance rows
# too, naming the frame a detection was observed in and the track it belongs to — without
# ``image_index``, ``(item_index, instance_index)`` repeats across the frames of one
# sequence. ``track_index`` is ``-1`` on a detection no tracker linked. An image-item task
# has no use for any of them, and writes none.
LEVEL_COLUMNS: tuple[str, ...] = (
    "level",
    "instance_index",
    "image_index",
    "track_index",
    "sequence_index",
)

# Identifiers only some tasks produce. ``track_id`` names the track a tracking detection
# belongs to, ``-1`` when the detection is untracked.
#
# A reserved column rather than a factor, because it is an identifier: as a factor it
# would be binned and handed to bias and diversity analysis as though a track number
# were an observed property of the data, which is exactly what ``item_index`` and
# ``target_index`` are reserved to prevent. It stays fully queryable —
# ``rows_at("instance")["track_id"]`` — and this is also the column a future ``track``
# level would key its rows on, so nothing has to move when tracks become rows.
IDENTIFIER_COLUMNS: tuple[str, ...] = ("track_id",)

# Factor names colliding with any of these are prefixed with ``metadata_``. This is
# wider than the historical five-column set: the level key columns are just as
# load-bearing, so a metadata key named ``level`` or ``instance_index`` is renamed
# rather than allowed to clobber the column. :data:`LEGACY_COLUMNS` still holds the
# original tuple for callers that need it.
RESERVED_COLUMNS: tuple[str, ...] = LEGACY_COLUMNS + LEVEL_COLUMNS + IDENTIFIER_COLUMNS

# Reserved columns a block emits only when it actually has a value for them: the
# level-key columns, i.e. everything in LEVEL_COLUMNS that is not the level tag itself,
# plus the optional identifiers. A block carries the key components of its own level and
# of any ancestor its ``item_index`` does not already identify, and no others.
_OPTIONAL_COLUMNS: tuple[str, ...] = (
    *(name for name in LEVEL_COLUMNS if name != "level"),
    *IDENTIFIER_COLUMNS,
)


def reserved_block_columns(level: FactorLevel, size: int, **values: Any) -> dict[str, list[Any]]:
    """Build the reserved (non-factor) columns of a single row block.

    Sole producer of the reserved column layout: every structurer and
    :meth:`~dataeval.Metadata.from_factors` routes through here, so the schema
    cannot drift between the dataset path and the raw-factor path.

    Parameters
    ----------
    level : str
        Level tag written to every row of the block.
    size : int
        Number of rows in the block.
    **values : Any
        Values for individual reserved columns, as a sequence of length ``size``.
        Every column in :data:`LEGACY_COLUMNS` is emitted whether supplied or not,
        filled with null when omitted; the level-key columns of
        :data:`LEVEL_COLUMNS` and the identifiers of :data:`IDENTIFIER_COLUMNS` are
        emitted only when supplied, since a block carries its own level's key
        components and only the identifiers its task produces.

    Returns
    -------
    dict[str, list[Any]]
        Column name to values, ready to hand to :class:`RowBlock`.

    Raises
    ------
    ValueError
        When a supplied name is not a reserved column, or when a supplied value
        sequence is not ``size`` long.
    """
    unknown = sorted(set(values) - set(RESERVED_COLUMNS))
    if unknown:
        raise ValueError(f"Column(s) {unknown} are not reserved columns {list(RESERVED_COLUMNS)}.")

    columns: dict[str, list[Any]] = {"level": [level] * size}
    for name in LEGACY_COLUMNS:
        supplied = values.get(name)
        columns[name] = [None] * size if supplied is None else _as_column(supplied)
    columns.update({name: _as_column(values[name]) for name in _OPTIONAL_COLUMNS if values.get(name) is not None})
    _reject_ragged(level, size, columns)
    return columns


def _reject_ragged(level: FactorLevel, size: int, columns: Mapping[str, Sequence[Any]]) -> None:
    """Reject a block whose columns disagree on how many rows it has.

    Ragged columns would otherwise surface as an opaque polars ``ShapeError`` at
    DataFrame construction, long after the structurer that produced them.
    """
    ragged = {name: len(column) for name, column in columns.items() if len(column) != size}
    if ragged:
        raise ValueError(
            f"Every column of a {level!r} block must have {size} values, one per row; got {ragged}.",
        )


def _as_column(values: Any) -> list[Any]:
    """Normalize a column's values to a plain list of Python scalars."""
    return values.tolist() if isinstance(values, np.ndarray) else list(values)


def safe_column_name(name: str) -> str:
    """Prefix a factor name that would clobber a reserved dataframe column.

    Parameters
    ----------
    name : str
        Factor name as supplied by the dataset or the caller.

    Returns
    -------
    str
        ``name`` unchanged, or ``metadata_<name>`` when it is in
        :data:`RESERVED_COLUMNS`.
    """
    return f"metadata_{name}" if name in RESERVED_COLUMNS else name


def _take(values: Any, positions: NDArray[np.intp]) -> list[Any]:
    """Gather ``values`` at ``positions``, tolerating arrays, lists and other sequences.

    A **negative position** means the row has no ancestor at that level and yields None.
    That is not a defensive nicety: it is the only representation of partial ancestry the
    layout has, and the diamond in the level graph makes partial ancestry real — a
    detection no tracker linked has a frame but no track, so a per-track factor has no
    value for it. Gathering such a row naively would index from the end of the array and
    silently attribute another track's value to it.
    """
    missing = positions < 0
    if missing.all():
        return [None] * len(positions)

    # Clamped rather than filtered so the gather stays one vectorized operation; every
    # clamped slot is overwritten with None below.
    safe = np.where(missing, 0, positions)
    if isinstance(values, np.ndarray):
        gathered = values[safe].tolist()
    else:
        sequence = values if isinstance(values, (list, tuple)) else list(values)
        gathered = [sequence[position] for position in safe]

    for index in np.flatnonzero(missing):
        gathered[index] = None
    return gathered


def _log_items_without_targets(without: Sequence[int], level: FactorLevel, items: int) -> None:
    """Note dataset items that carried no target.

    These items keep their item-level row and every factor on it; they contribute no
    row at ``level``, so label-aware analysis covers fewer items than the dataset has.
    Informational rather than a warning: a partially labelled dataset is a legitimate
    shape, and it costs no data now that the item level is separate from the target
    level.
    """
    if not without:
        return
    _logger.info(
        "%d of %d dataset item(s) %s carried no target and contribute no %r rows. Their item-level "
        "rows and factors are unaffected; Metadata.item_indices lists the items that do have targets.",
        len(without),
        items,
        list(without) if len(without) <= 10 else [*without[:10], "..."],
        level,
    )


# Sentinel for "the iterator had nothing left", distinct from any value it could yield.
_EXHAUSTED: Any = object()


def _running_index(parents: NDArray[np.intp]) -> NDArray[np.intp]:
    """Index each row within its parent group, assuming rows are grouped by parent.

    Parameters
    ----------
    parents : NDArray[np.intp]
        Parent position of each row, in row order and grouped by parent.

    Returns
    -------
    NDArray[np.intp]
        0, 1, 2, ... restarting at each new parent.
    """
    count = len(parents)
    if count == 0:
        return np.empty(0, dtype=np.intp)
    starts = np.concatenate(([0], np.flatnonzero(parents[1:] != parents[:-1]) + 1))
    group_sizes = np.diff(np.append(starts, count))
    return np.arange(count, dtype=np.intp) - np.repeat(starts, group_sizes)


@dataclass(frozen=True)
class RowBlock:
    """A contiguous run of dataframe rows belonging to a single level.

    Attributes
    ----------
    level : str
        Level every row in this block belongs to.
    size : int
        Number of rows in the block.
    columns : Mapping[str, Sequence[Any]]
        Reserved (non-factor) column values for the block.
    ancestor_pos : Mapping[str, NDArray[np.intp]]
        For the block's own level and each ancestor level, the position of the
        corresponding row within *that* level's block. This is what makes
        downward factor propagation a gather rather than a join.

        A **negative** position marks a row with no ancestor at that level, and
        propagates as None. A level absent from the mapping entirely is the different,
        block-wide statement that no row here has such an ancestor. Both arise from the
        diamond in the level graph: an untracked detection has a frame but no track, so
        its ``track`` position is negative, while a frame row has no ``track`` key at all
        because ``image`` and ``track`` are siblings.
    """

    level: FactorLevel
    size: int
    columns: Mapping[str, Sequence[Any]]
    ancestor_pos: Mapping[FactorLevel, NDArray[np.intp]]


@dataclass(frozen=True)
class RowLayout:
    """Positional map from dataframe rows back to the level hierarchy.

    Retained by :class:`~dataeval.Metadata` after structuring so that
    factors added later can be propagated using exactly the same rules that were
    applied during the initial build.

    The per-block ancestor maps are plain dicts rather than ``MappingProxyType``:
    a layout travels inside every :class:`~dataeval.Metadata` instance, and a
    mappingproxy cannot be pickled, which would make the whole instance
    un-deep-copyable. The dataclass is frozen and the fields are typed as
    ``Mapping``, which is the read-only contract.
    """

    blocks: tuple[tuple[FactorLevel, int, Mapping[FactorLevel, NDArray[np.intp]]], ...]

    @classmethod
    def from_blocks(cls, blocks: Sequence[RowBlock]) -> "RowLayout":
        """Build a layout from the row blocks a structurer produced."""
        return cls(tuple((block.level, block.size, dict(block.ancestor_pos)) for block in blocks))

    @property
    def counts(self) -> Mapping[FactorLevel, int]:
        """Number of rows at each level, in row order."""
        return MappingProxyType({level: size for level, size, _ in self.blocks})

    def partial_ancestry(self, level: FactorLevel, at: FactorLevel) -> bool:
        """Whether some row at ``at`` has no ancestor at ``level``.

        True only for the in-between case: ``level`` does reach ``at``, but not from every
        row. A detection no tracker linked is the instance of it — it has a frame and no
        track, so a per-track factor is null on that one row while being present on its
        neighbours. Callers that need a total column have to exclude such a factor, which
        is a property of the layout rather than of the values, so it is answered here.

        Parameters
        ----------
        level : str
            Level the values are defined at.
        at : str
            Level whose rows would read them.

        Returns
        -------
        bool
            True when at least one row at ``at`` records no ancestor position at ``level``.
            False when every row has one, and False when ``at`` has no rows at all.
        """
        for block_level, _, ancestor_pos in self.blocks:
            if block_level != at:
                continue
            positions = ancestor_pos.get(level)
            return positions is not None and bool(np.any(positions < 0))
        return False

    def expand(self, values: Any, level: FactorLevel) -> list[Any]:
        """Spread values defined at ``level`` across every dataframe row.

        Rows at ``level`` receive their own value, rows at descendant levels
        receive their ancestor's value, and every other row receives None — as does a
        descendant row that has no ancestor at ``level``, such as a detection no tracker
        linked when ``level`` is ``track``.

        Parameters
        ----------
        values : Any
            One value per row at ``level``, in that level's row order.
        level : str
            Level the values are defined at.

        Returns
        -------
        list[Any]
            A full-length column ready to hand to polars.
        """
        column: list[Any] = []
        for _, size, ancestor_pos in self.blocks:
            positions = ancestor_pos.get(level)
            column.extend([None] * size if positions is None else _take(values, positions))
        return column


@dataclass(frozen=True)
class StructuredData:
    """Everything a structurer extracts from a dataset, before any binning.

    Attributes
    ----------
    blocks : Sequence[RowBlock]
        Row blocks ordered coarsest level first.
    factors : Mapping[str, Mapping[str, Any]]
        Factor values keyed by the level they are defined at.
    dropped_factors : Mapping[str, Sequence[str]]
        Factors discarded during metadata merging, with reasons.
    raw : Sequence[Mapping[str, Any]]
        Untouched per-item metadata dictionaries.
    class_labels : NDArray[np.intp]
        One label per target-level row.
    item_indices : NDArray[np.intp]
        Source item index for each target-level row.
    """

    blocks: Sequence[RowBlock]
    factors: Mapping[FactorLevel, Mapping[str, Any]]
    dropped_factors: Mapping[str, Sequence[str]] = field(default_factory=dict)
    raw: Sequence[Mapping[str, Any]] = field(default_factory=list)
    class_labels: NDArray[np.intp] = field(default_factory=lambda: np.empty(0, dtype=np.intp))
    item_indices: NDArray[np.intp] = field(default_factory=lambda: np.empty(0, dtype=np.intp))

    def __post_init__(self) -> None:
        """Reject a factor name declared at more than one level.

        A factor becomes one dataframe column, and a column holds values for exactly
        one level: :meth:`RowLayout.expand` fills that level's rows and its
        descendants' and nulls everything else, so a second declaration of the same
        name does not merge with the first, it replaces it — and the losing level's
        rows are left holding nulls in a column that still counts as its factor.

        Checked here rather than left to a convention because the two existing
        structurers only avoid it by explicitly subtracting the overlap their two
        metadata merges produce. Nothing about that is visible to the next structurer,
        and the failure it prevents is silent: :meth:`Metadata._factor_level` resolves
        the name to one level and bins whatever that level's rows hold, which is a
        column of nulls. Qualify the names instead — ``frame_timestamp`` and
        ``instance_timestamp`` — the way ``add_factors`` does when a source index
        spans levels.
        """
        seen: dict[str, FactorLevel] = {}
        for level, factors in self.factors.items():
            for name in factors:
                if (first := seen.get(name)) is not None:
                    raise ValueError(
                        f"Factor {name!r} is declared at both the {first!r} and {level!r} levels. "
                        "A factor is one column and a column belongs to one level, so the second "
                        "declaration would null out the first level's rows. Give each level's "
                        f"values their own name, e.g. {f'{first}_{name}'!r} and {f'{level}_{name}'!r}.",
                    )
                seen[name] = level

    @property
    def layout(self) -> RowLayout:
        """Positional map for the rows this bundle describes."""
        return RowLayout.from_blocks(self.blocks)

    def to_rows(self) -> dict[str, list[Any]]:
        """Flatten blocks and factors into a single column-oriented mapping.

        Returns
        -------
        dict[str, list[Any]]
            Mapping of column name to values across every row, with factors
            propagated down to descendant levels and nulled elsewhere.
        """
        # Reserved columns are block-local: each block supplies its own, and a
        # block that omits one carries null for it.
        rows: dict[str, list[Any]] = {name: [] for block in self.blocks for name in block.columns}
        for block in self.blocks:
            for name in rows:
                values = block.columns.get(name)
                rows[name].extend([None] * block.size if values is None else list(values))

        # Factors are level-local and propagate downwards, which is exactly what
        # RowLayout.expand does — so the gather lives in one place. Assignment rather
        # than merge is safe because __post_init__ has already rejected a name declared
        # at two levels, so no two iterations write the same key.
        layout = self.layout
        for level, factors in self.factors.items():
            for name, values in factors.items():
                rows[name] = layout.expand(values, level)
        return rows


class Structurer:
    """Level model for a task: which levels exist, and what the items and labels are.

    Subclasses declare which levels they produce, which level a dataset *item*
    corresponds to, and which level the labels sit at. The core engine consumes
    the resulting :class:`StructuredData` identically regardless of task.

    Declaring the level model is deliberately separate from producing rows.
    Most structurers read a dataset and so derive from :class:`DatasetStructurer`,
    but :class:`FactorsStructurer` is fed raw arrays and has no dataset to
    iterate; it declares a level model without acquiring an obligation to
    implement :meth:`DatasetStructurer.build`.

    Attributes
    ----------
    task : str
        Short task identifier, e.g. ``"IC"`` or ``"OD"``.
    levels : FactorLevelSchema
        Levels this structurer emits rows for.
    item_level : str
        Level corresponding to one dataset item.
    label_level : str
        Level whose rows carry ``class_label``.
    multi_target : bool
        Whether one dataset item can yield more than one labelled row.
    """

    task: TASK = "unknown"
    levels: FactorLevelSchema = FactorLevelSchema.of("image")
    item_level: FactorLevel = "image"
    label_level: FactorLevel = "image"
    multi_target: bool = False
    legacy_level_aliases: Mapping[str, FactorLevel] = MappingProxyType({})

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Reject a subclass whose item or label level sits outside its own schema.

        The three declarations are interdependent, and a mismatch surfaces far
        from its cause: propagation and every level filter would quietly select
        no rows. Checking at class creation puts the error on the declaration.
        """
        super().__init_subclass__(**kwargs)
        for attribute in ("item_level", "label_level"):
            level = getattr(cls, attribute)
            if level not in cls.levels:
                raise TypeError(
                    f"{cls.__name__}.{attribute} is {level!r}, which is not one of its "
                    f"declared levels {list(cls.levels)}.",
                )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(task={self.task!r}, levels={list(self.levels)})"


class DatasetStructurer(Structurer, ABC):
    """Strategy for turning a dataset into levelled metadata rows.

    Parameters
    ----------
    first_datum : tuple or None, default None
        The dataset's first ``(item, target, metadata)`` triple, when the caller has
        already read it. :func:`select_structurer` has to read it to detect the task,
        and a dataset that decodes on ``__getitem__`` would otherwise pay for that
        item twice; handing it back here makes :meth:`build` reuse it.
    """

    def __init__(self, first_datum: tuple[Any, Any, DatumMetadata] | None = None) -> None:
        self._first_datum = first_datum

    def _datum(
        self,
        dataset: AnnotatedDataset[tuple[Any, Any, DatumMetadata]],
        index: int,
    ) -> tuple[Any, Any, DatumMetadata]:
        """Read one datum, reusing — and then releasing — the one the task probe read.

        The cached datum holds a decoded item, which for an image or a video frame is
        the largest object in this class by orders of magnitude. A structurer outlives
        :meth:`build` (:class:`~dataeval.Metadata` keeps it for its level model), so
        handing the datum out has to drop the reference with it, or every long-lived
        Metadata pins item 0's pixels for its whole lifetime.
        """
        if index == 0 and self._first_datum is not None:
            datum, self._first_datum = self._first_datum, None
            return datum
        return dataset[index]

    @abstractmethod
    def build(
        self,
        dataset: AnnotatedDataset[tuple[Any, Any, DatumMetadata]],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> StructuredData:
        """Extract rows, factors and labels from ``dataset``."""
        ...

    def _merge_factors(
        self,
        raw: Sequence[Mapping[str, Any]],
        *,
        ignore_lists: bool,
        targets_per_item: Sequence[int] | None = None,
    ) -> tuple[Mapping[str, Any], Mapping[str, Sequence[str]]]:
        """Merge per-item metadata dictionaries into flat factor arrays.

        Parameters
        ----------
        raw : Sequence[Mapping[str, Any]]
            Per-item metadata as provided by the dataset.
        ignore_lists : bool
            When True, list-valued metadata is discarded and one value per item
            is produced. When False, list-valued metadata is expanded across the
            targets of each item.
        targets_per_item : Sequence[int] or None, default None
            Number of target-level rows contributed by each item; required to
            expand list-valued metadata.

        Returns
        -------
        tuple[Mapping[str, Any], Mapping[str, Sequence[str]]]
            The merged factors and the factors that were dropped.
        """
        merged, dropped = merge_metadata(
            raw,
            return_dropped=True,
            ignore_lists=ignore_lists,
            targets_per_image=targets_per_item,
        )
        factors = {safe_column_name(k): v for k, v in merged.items() if k != "_image_index"}
        return factors, dropped


class PropagationMixin:
    """Downward propagation of factors along the level hierarchy.

    Propagation is expressed positionally: every row block records, for its own
    level and each ancestor level, the index of the row it inherits from. A
    factor defined at any ancestor is then a single gather away, and factors
    never travel upwards or get aggregated — rows above a factor's level simply
    hold nulls.
    """

    @staticmethod
    def _own_positions(size: int) -> NDArray[np.intp]:
        """Identity position map for a block's own level."""
        return np.arange(size, dtype=np.intp)

    @staticmethod
    def _inherit(
        parent_positions: Mapping[FactorLevel, NDArray[np.intp]],
        selector: NDArray[np.intp],
    ) -> dict[FactorLevel, NDArray[np.intp]]:
        """Lift a parent block's ancestor map down onto a child block.

        Parameters
        ----------
        parent_positions : Mapping[str, NDArray[np.intp]]
            The parent block's ``ancestor_pos`` mapping.
        selector : NDArray[np.intp]
            For each child row, the position of its parent row.

        Returns
        -------
        dict[Level, NDArray[np.intp]]
            Ancestor positions for the child block, covering the parent level
            and everything above it.
        """
        return {level: np.asarray(positions, dtype=np.intp)[selector] for level, positions in parent_positions.items()}


class InstanceBuildingMixin:
    """Box/label extraction shared by instance-producing structurers.

    Used by object detection and multi-object tracking strategies. A
    :obj:`~dataeval.protocols.SingleFrameObjectTrackingTarget` is an
    :obj:`~dataeval.protocols.ObjectDetectionTarget` plus ``track_ids``, so both tasks
    read boxes and labels the same way and tracking adds one call on top.
    """

    @staticmethod
    def _instance_arrays(
        target: ObjectDetectionTarget | SingleFrameObjectTrackingTarget,
    ) -> tuple[NDArray[np.intp], NDArray[np.float32], NDArray[np.float32]]:
        """Extract per-detection labels, boxes and scores from a detection target.

        Returns
        -------
        tuple
            ``(labels, boxes, scores)`` with one entry per detection. ``scores``
            keeps its original shape, which may be per-detection or
            per-detection-per-class.
        """
        labels = as_numpy(target.labels).reshape(-1).astype(np.intp)
        count = len(labels)
        boxes = (
            as_numpy(target.boxes).astype(np.float32).reshape(count, 4) if count else np.empty((0, 4), dtype=np.float32)
        )
        scores = as_numpy(target.scores).astype(np.float32) if count else np.empty(0, dtype=np.float32)
        return labels, boxes, scores

    @staticmethod
    def _track_ids(target: SingleFrameObjectTrackingTarget, count: int) -> NDArray[np.intp]:
        """Extract per-detection track ids from one frame's tracking target.

        Parameters
        ----------
        target : SingleFrameObjectTrackingTarget
            Frame target to read ``track_ids`` from.
        count : int
            Detections in this frame, as already established by :meth:`_instance_arrays`.

        Returns
        -------
        NDArray[np.intp]
            One track id per detection, ``-1`` where a detection belongs to no track.
        """
        if not count:
            return np.empty(0, dtype=np.intp)
        return as_numpy(target.track_ids).reshape(-1).astype(np.intp)


class ICStructurer(PropagationMixin, DatasetStructurer):
    """Image classification: items are images, targets are the images themselves.

    The instance level is separate from the image level even though a classification
    instance *is* the whole image, because the two answer different questions: an
    image row exists for every dataset item, an instance row only where there is a
    label to attach. Collapsing them would delete an unlabeled item — and all of its
    metadata — from the dataframe entirely.

    Sharing the level with object detection is what keeps one object one thing: a
    detection is an instance in an object detection dataset, and the same detection
    seen through :class:`~dataeval.data.DetectionCrops` is an instance here too.
    """

    task = "IC"
    levels = FactorLevelSchema.of("image", "instance")
    item_level = "image"
    label_level = "instance"

    def build(
        self,
        dataset: AnnotatedDataset[tuple[Any, Any, DatumMetadata]],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> StructuredData:
        raw: list[Mapping[str, Any]] = []
        labels: list[int] = []
        scores: list[NDArray[Any]] = []
        srcidx: list[int] = []

        count = len(dataset)
        unlabeled: list[int] = []
        for i in range(count):
            _, target, metadata = self._datum(dataset, i)
            raw.append(metadata)
            if not isinstance(target, Array):
                raise TypeError(
                    f"Encountered unsupported target type {type(target).__name__} for task {self.task}.",
                )
            values = as_numpy(target)
            if len(values):
                labels.append(int(np.argmax(values)))
                scores.append(values)
                srcidx.append(i)
            else:
                unlabeled.append(i)
            if progress_callback:
                progress_callback(i + 1, total=count)

        image_of_instance = np.asarray(srcidx, dtype=np.intp)
        class_labels = np.asarray(labels, dtype=np.intp)
        score_values = np.asarray(scores, dtype=np.float32) if scores else np.empty(0, dtype=np.float32)
        instance_index = _running_index(image_of_instance)
        instance_count = len(image_of_instance)

        instances_per_item = np.bincount(image_of_instance, minlength=count).astype(int).tolist()
        instance_factors, dropped = self._merge_factors(
            raw,
            ignore_lists=False,
            targets_per_item=instances_per_item,
        )
        image_factors, _ = self._merge_factors(raw, ignore_lists=True)
        # Same rule as object detection: a name both merges produced is item metadata
        # replicated onto the target rows, so keep it once at the image level and let
        # propagation do the replicating.
        instance_factors = {name: values for name, values in instance_factors.items() if name not in image_factors}

        image_block = RowBlock(
            "image",
            count,
            reserved_block_columns("image", count, item_index=list(range(count))),
            {"image": self._own_positions(count)},
        )
        instance_block = RowBlock(
            "instance",
            instance_count,
            reserved_block_columns(
                "instance",
                instance_count,
                item_index=image_of_instance,
                # One instance per image at most, so the index within the image is
                # always 0 — but derive it rather than assume, as object detection does.
                # ``instance_index`` is the instance level's own key column and is
                # written by every structurer that declares the level, so that caller
                # code reading rows_at(level)[f"{level}_index"] does not branch on task.
                target_index=instance_index,
                class_label=class_labels,
                score=score_values,
                instance_index=instance_index,
            ),
            {
                **self._inherit(image_block.ancestor_pos, image_of_instance),
                "instance": self._own_positions(instance_count),
            },
        )

        _log_items_without_targets(unlabeled, "instance", count)
        _logger.info("%s dataset: %d items, %d classes", self.task, count, len(np.unique(class_labels)))
        return StructuredData(
            [image_block, instance_block],
            {"image": image_factors, "instance": instance_factors},
            dropped,
            raw,
            class_labels,
            image_of_instance,
        )


class ODImageStructurer(InstanceBuildingMixin, PropagationMixin, DatasetStructurer):
    """Object detection over images: items are images, targets are instances."""

    task = "OD"
    levels = FactorLevelSchema.of("image", "instance")
    item_level = "image"
    label_level = "instance"
    multi_target = True

    # Object detection called its target rows ``"target"`` through v1.1.0. It is the
    # only task that ever did, so it is the only one that translates the name.
    legacy_level_aliases = MappingProxyType({"target": "instance"})

    def build(
        self,
        dataset: AnnotatedDataset[tuple[Any, Any, DatumMetadata]],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> StructuredData:
        raw: list[Mapping[str, Any]] = []
        labels: list[NDArray[Any]] = []
        boxes: list[NDArray[Any]] = []
        scores: list[NDArray[Any]] = []
        srcidx: list[int] = []

        count = len(dataset)
        # An image with no detections contributes no instance rows at all. That is far
        # more common here than an unlabeled item is for classification, so it is
        # tracked and reported for the same reason: without it, label-aware analysis
        # silently covers a subset of the dataset and the first sign is a row-count
        # mismatch from add_factors much later.
        undetected: list[int] = []
        for i in range(count):
            _, target, metadata = self._datum(dataset, i)
            raw.append(metadata)
            if not isinstance(target, ObjectDetectionTarget):
                raise TypeError(
                    f"Encountered unsupported target type {type(target).__name__} for task {self.task}.",
                )
            instance_labels, instance_boxes, instance_scores = self._instance_arrays(target)
            if len(instance_labels):
                labels.append(instance_labels)
                boxes.append(instance_boxes)
                scores.append(instance_scores)
                srcidx.extend([i] * len(instance_labels))
            else:
                undetected.append(i)
            if progress_callback:
                progress_callback(i + 1, total=count)

        image_of_instance = np.asarray(srcidx, dtype=np.intp)
        class_labels = np.concatenate(labels).astype(np.intp) if labels else np.empty(0, dtype=np.intp)
        box_values = np.concatenate(boxes).astype(np.float32) if boxes else np.empty((0, 4), dtype=np.float32)
        score_values = np.concatenate(scores).astype(np.float32) if scores else np.empty(0, dtype=np.float32)
        instance_index = _running_index(image_of_instance)
        instances = len(image_of_instance)

        instances_per_item = np.bincount(image_of_instance, minlength=count).astype(int).tolist()
        instance_factors, dropped = self._merge_factors(
            raw,
            ignore_lists=False,
            targets_per_item=instances_per_item,
        )
        image_factors, _ = self._merge_factors(raw, ignore_lists=True)
        # Anything the target-level merge produced that the item-level merge also
        # produced is item metadata replicated across instances; keep it at the image
        # level and let propagation replicate it instead of storing it twice.
        instance_factors = {name: values for name, values in instance_factors.items() if name not in image_factors}

        image_block = RowBlock(
            "image",
            count,
            reserved_block_columns("image", count, item_index=list(range(count))),
            {"image": self._own_positions(count)},
        )
        # ``instance_index`` is the instance level's own key component; ``target_index`` is the
        # legacy public spelling of "index within the item at whatever level the labels
        # sit". For this task they are the same number, written from the same array so
        # they cannot drift; a task whose targets are not instances would fill them apart.
        instance_block = RowBlock(
            "instance",
            instances,
            reserved_block_columns(
                "instance",
                instances,
                item_index=image_of_instance,
                target_index=instance_index,
                class_label=class_labels,
                score=score_values,
                box=box_values,
                instance_index=instance_index,
            ),
            {**self._inherit(image_block.ancestor_pos, image_of_instance), "instance": self._own_positions(instances)},
        )

        _log_items_without_targets(undetected, "instance", count)
        _logger.info(
            "Object Detection dataset: %d images, %d classes, %d detections",
            count,
            len(np.unique(class_labels)),
            instances,
        )
        return StructuredData(
            [image_block, instance_block],
            {"image": image_factors, "instance": instance_factors},
            dropped,
            raw,
            class_labels,
            image_of_instance,
        )


class _FrameRows(NamedTuple):
    """One frame's contribution to a tracking dataset: its own keys, plus its detections.

    A named tuple rather than a bare one because the walk needs seven fields per frame,
    and ``rows.track_ids`` at the call site says what ``rows[6]`` cannot.
    """

    frame_index: int
    time_s: float | None
    pts: int | None
    labels: NDArray[np.intp]
    boxes: NDArray[np.float32]
    scores: NDArray[np.float32]
    track_ids: NDArray[np.intp]


@dataclass
class _MOTAccumulator:
    """Row accumulators for one pass over a tracking dataset.

    A class rather than a wall of locals in ``build``, because the walk fills three levels
    at once — frames, tracks and instances — and has to keep a per-sequence track registry
    alive while it does. Threading that many parallel lists through helpers is what makes
    the alternative unreadable.

    Tracks are discovered rather than declared: a sequence's track rows are created on each
    ``track_id``'s first appearance, so they end up densely numbered in order of first
    observation whatever ids the dataset used. The registry is per sequence, which is what
    keeps the same id in two videos two separate tracks.
    """

    frame_sequence: list[int] = field(default_factory=list)
    frame_index: list[int] = field(default_factory=list)
    frame_time_s: list[float | None] = field(default_factory=list)
    frame_pts: list[int | None] = field(default_factory=list)

    track_sequence: list[int] = field(default_factory=list)
    track_id: list[int] = field(default_factory=list)
    track_length: list[int] = field(default_factory=list)
    track_first_frame: list[int] = field(default_factory=list)
    track_last_frame: list[int] = field(default_factory=list)
    track_first_time: list[float | None] = field(default_factory=list)
    track_last_time: list[float | None] = field(default_factory=list)

    instance_labels: list[NDArray[Any]] = field(default_factory=list)
    instance_boxes: list[NDArray[Any]] = field(default_factory=list)
    instance_scores: list[NDArray[Any]] = field(default_factory=list)
    instance_track_ids: list[NDArray[Any]] = field(default_factory=list)
    instance_sequence: list[int] = field(default_factory=list)
    instance_image_pos: list[int] = field(default_factory=list)
    instance_track_pos: list[int] = field(default_factory=list)

    def add_item(self, item: int, frames: Iterable[_FrameRows]) -> None:
        """Absorb one dataset item — one video — and everything inside it."""
        registry: dict[int, int] = {}
        for rows in frames:
            position = len(self.frame_sequence)
            self.frame_sequence.append(item)
            self.frame_index.append(rows.frame_index)
            self.frame_time_s.append(rows.time_s)
            self.frame_pts.append(rows.pts)

            if len(rows.labels):
                self.instance_labels.append(rows.labels)
                self.instance_boxes.append(rows.boxes)
                self.instance_scores.append(rows.scores)
                self.instance_track_ids.append(rows.track_ids)
                self.instance_sequence.extend([item] * len(rows.labels))
                self.instance_image_pos.extend([position] * len(rows.labels))
                self._add_tracks(item, rows, registry)

    def _add_tracks(self, item: int, rows: _FrameRows, registry: dict[int, int]) -> None:
        """Attach one frame's detections to their tracks, opening any not yet seen.

        A detection with a negative id belongs to no track, and records ``-1`` as its track
        position: the layout's marker for "no ancestor at that level". Nothing is invented
        for it — a singleton track would be a track the data says does not exist, and would
        skew every per-track statistic toward length one.
        """
        for track_id in rows.track_ids.tolist():
            if track_id < 0:
                self.instance_track_pos.append(-1)
                continue

            position = registry.get(track_id)
            if position is None:
                position = registry[track_id] = len(self.track_sequence)
                self.track_sequence.append(item)
                self.track_id.append(track_id)
                self.track_length.append(0)
                self.track_first_frame.append(rows.frame_index)
                self.track_last_frame.append(rows.frame_index)
                self.track_first_time.append(rows.time_s)
                self.track_last_time.append(rows.time_s)
            else:
                # min/max rather than "the latest wins": frame_index comes off the stream
                # and a duck-typed frame is not obliged to number its frames in order.
                self.track_first_frame[position] = min(self.track_first_frame[position], rows.frame_index)
                self.track_last_frame[position] = max(self.track_last_frame[position], rows.frame_index)
                self.track_first_time[position] = _min_or_none(self.track_first_time[position], rows.time_s)
                self.track_last_time[position] = _max_or_none(self.track_last_time[position], rows.time_s)

            self.track_length[position] += 1
            self.instance_track_pos.append(position)


def _without(factors: Mapping[str, Any], displaced: Container[str], level: FactorLevel) -> dict[str, Any]:
    """Drop factor names a structurer derives itself, logging each one it removes.

    A factor belongs to exactly one level, so a metadata key spelling the same name as a
    derived factor cannot simply coexist with it. The derived value wins — it is read off
    the dataset's own frames and targets — and the displacement is logged rather than
    silent, because the value a caller sees is not the one their metadata supplied.
    """
    kept = {name: values for name, values in factors.items() if name not in displaced}
    for name in factors:
        if name not in kept:
            _logger.info(
                "Metadata key %r at the %r level is displaced by the derived factor of the same "
                "name, which is read from the dataset's own frames and targets.",
                name,
                level,
            )
    return kept


def _min_or_none(current: float | None, candidate: float | None) -> float | None:
    """Smaller of two optional times, None when either is missing."""
    return None if current is None or candidate is None else min(current, candidate)


def _max_or_none(current: float | None, candidate: float | None) -> float | None:
    """Larger of two optional times, None when either is missing."""
    return None if current is None or candidate is None else max(current, candidate)


class MOTStructurer(InstanceBuildingMixin, PropagationMixin, DatasetStructurer):
    """Multi-object tracking over video: items are sequences, targets are instances.

    Four levels, and the only task whose graph is a diamond rather than a chain:
    ``sequence`` is the item level (one dataset item is one video), ``image`` is a frame
    and ``track`` is one tracked object — siblings under the sequence — and ``instance``
    is the label level, one row per detection, which sits under *both*. A detection is one
    observation: of a track, in a frame.

    Because ``image`` sits between the item level and the label level, an instance row
    needs its frame's key as well as its own to be uniquely identified: ``instance_index``
    counts within the frame, so ``(item_index, image_index, instance_index)`` is the
    compound key, and ``(item_index, image_index)`` joins instance rows to their frame's
    row. ``target_index`` keeps counting within the whole item, as it does for every task.

    A track is a level rather than a column so that metadata can be organized *by track*:
    a factor added at ``track`` is stored once per track and propagates down to every
    detection in it, and ``rows_at("track")`` reads it once per track rather than once per
    detection. Tracks are scoped to their sequence — the same ``track_id`` in two videos is
    two tracks — and ``track_index`` numbers them densely within each, in order of first
    appearance, because a dataset's own ids may be sparse or arbitrary.

    A detection that no tracker linked (``track_id == -1``) has a frame but **no track**.
    Its ``track_index`` is ``-1``, which propagates every track-level factor to it as null
    rather than inventing a singleton track for it. :class:`~dataeval.Metadata` keeps such
    a factor out of factor analysis at any view where some row is untracked, and
    ``md.at("track")`` still reads it in full — see :attr:`~dataeval.Metadata.factor_data`.

    Per-frame metadata is merged at the ``image`` level, not the instance level. A video's
    list-valued metadata is per frame — one timestamp per frame, not one per detection —
    so expanding it across detections would be wrong even where the counts happen to
    match. Instance-level factors therefore come only from the target data itself.
    """

    task = "MOT"
    levels = FactorLevelSchema.of("sequence", "image", "track", "instance")
    item_level = "sequence"
    label_level = "instance"
    multi_target = True

    @classmethod
    def _frames_of(
        cls,
        video_stream: Any,
        frame_tracks: Sequence[SingleFrameObjectTrackingTarget],
    ) -> Iterator[_FrameRows]:
        """Pair each decoded frame with its target, yielding one frame's rows at a time.

        Streams rather than materializing the frames: only each frame's keys and timings
        are retained, while a decoded :obj:`~dataeval.protocols.VideoFrame` holds a full
        pixel array.

        A frame count that disagrees with the target count is a dataset bug and is raised
        rather than absorbed. Pairing the two up to the shorter of them would either drop
        real detections or annotate frames with another frame's boxes, and would give no
        signal that either had happened.

        Yields
        ------
        _FrameRows
            One frame's index, timings and detection arrays. ``time_s`` and ``pts`` are
            None for a frame that does not declare them.

        Raises
        ------
        ValueError
            When the stream and the target disagree on how many frames the item has.
        """
        frames = iter(video_stream)
        for position, frame_target in enumerate(frame_tracks):
            frame = next(frames, _EXHAUSTED)
            if frame is _EXHAUSTED:
                raise ValueError(
                    f"Tracking target declares {len(frame_tracks)} frame target(s) but the item's "
                    f"video stream yielded only {position}; frame_tracks must hold exactly one "
                    "target per frame.",
                )
            labels, boxes, scores = cls._instance_arrays(frame_target)
            # MAITE's VideoFrame declares frame_index, time_s and pts, but dispatch here
            # duck-types the target rather than requiring the full protocol, so each is
            # optional. frame_index falls back to decode order, which is what it means for
            # a conforming stream anyway; a timing has no such stand-in and stays None.
            yield _FrameRows(
                getattr(frame, "frame_index", position),
                getattr(frame, "time_s", None),
                getattr(frame, "pts", None),
                labels,
                boxes,
                scores,
                cls._track_ids(frame_target, len(labels)),
            )

        if next(frames, _EXHAUSTED) is not _EXHAUSTED:
            raise ValueError(
                f"Item's video stream yields more frames than the tracking target's "
                f"{len(frame_tracks)} frame target(s); frame_tracks must hold exactly one "
                "target per frame.",
            )

    @staticmethod
    def _frame_factors(rows: _MOTAccumulator) -> dict[str, NDArray[Any]]:
        """Per-frame timings, as image-level factors, when every frame supplies them.

        All-or-nothing rather than null-padded. A partially null numeric factor cannot be
        binned — sorting it compares None against a float — so a factor present for only
        some frames would break factor analysis for the whole dataset rather than degrade
        gracefully. A conforming :obj:`~dataeval.protocols.VideoStream` declares both, so
        the all-or-nothing case is the normal one; a duck-typed stream that omits them gets
        no timing factors and a log line saying so.
        """
        factors: dict[str, NDArray[Any]] = {}
        for name, values, dtype in (
            ("time_s", rows.frame_time_s, np.float64),
            ("pts", rows.frame_pts, np.intp),
        ):
            missing = sum(value is None for value in values)
            if missing:
                _logger.info(
                    "%d of %d frame(s) do not declare %r, so no %r factor is produced; a "
                    "partially populated factor cannot be binned.",
                    missing,
                    len(values),
                    name,
                    name,
                )
                continue
            factors[name] = np.asarray(values, dtype=dtype)
        return factors

    @staticmethod
    def _track_factors(rows: _MOTAccumulator) -> dict[str, NDArray[Any]]:
        """Derive per-track factors: how long each track is, and how far it spans.

        ``track_length`` counts observations; ``frame_span`` counts frames from first to
        last inclusive. They differ exactly when a track has gaps, which makes the pair
        more informative than either alone. ``duration_s`` is the elapsed time over the
        same span, and follows the same all-or-nothing rule as the frame timings.
        """
        factors: dict[str, NDArray[Any]] = {
            "track_length": np.asarray(rows.track_length, dtype=np.intp),
            "frame_span": np.asarray(rows.track_last_frame, dtype=np.intp)
            - np.asarray(rows.track_first_frame, dtype=np.intp)
            + 1,
        }
        if all(value is not None for value in (*rows.track_first_time, *rows.track_last_time)):
            factors["duration_s"] = np.asarray(rows.track_last_time, dtype=np.float64) - np.asarray(
                rows.track_first_time, dtype=np.float64
            )
        return factors

    @staticmethod
    def _stacked(
        labels: Sequence[NDArray[Any]],
        boxes: Sequence[NDArray[Any]],
        scores: Sequence[NDArray[Any]],
        track_ids: Sequence[NDArray[Any]],
    ) -> tuple[NDArray[np.intp], NDArray[np.float32], NDArray[np.float32], NDArray[np.intp]]:
        """Concatenate the per-frame arrays into one array each, coarsest dtype preserved.

        Each is built explicitly when there is nothing to concatenate, because
        :func:`numpy.concatenate` rejects an empty sequence and because an empty result
        still has to carry the right dtype and, for boxes, the right width — a dataset
        with no detections at all must still produce a well-typed empty block.
        """
        return (
            np.concatenate(labels).astype(np.intp) if labels else np.empty(0, dtype=np.intp),
            np.concatenate(boxes).astype(np.float32) if boxes else np.empty((0, 4), dtype=np.float32),
            np.concatenate(scores).astype(np.float32) if scores else np.empty(0, dtype=np.float32),
            np.concatenate(track_ids).astype(np.intp) if track_ids else np.empty(0, dtype=np.intp),
        )

    def build(
        self,
        dataset: AnnotatedDataset[tuple[Any, Any, DatumMetadata]],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> StructuredData:
        raw: list[Mapping[str, Any]] = []
        count = len(dataset)
        rows = _MOTAccumulator()

        for i in range(count):
            video_stream, target, metadata = self._datum(dataset, i)
            raw.append(metadata)
            rows.add_item(i, self._frames_of(video_stream, target.frame_tracks))
            if progress_callback:
                progress_callback(i + 1, total=count)

        sequence_of_frame = np.asarray(rows.frame_sequence, dtype=np.intp)
        frame_own_index_arr = np.asarray(rows.frame_index, dtype=np.intp)
        n_frames = len(sequence_of_frame)

        sequence_of_track = np.asarray(rows.track_sequence, dtype=np.intp)
        n_tracks = len(sequence_of_track)

        class_labels, box_values, score_values, track_id_values = self._stacked(
            rows.instance_labels,
            rows.instance_boxes,
            rows.instance_scores,
            rows.instance_track_ids,
        )
        sequence_of_instance = np.asarray(rows.instance_sequence, dtype=np.intp)
        image_pos_of_instance = np.asarray(rows.instance_image_pos, dtype=np.intp)
        track_pos_of_instance = np.asarray(rows.instance_track_pos, dtype=np.intp)
        n_instances = len(sequence_of_instance)

        # Two distinct running indices, because an instance's direct parent (image) and
        # its item (sequence) are no longer the same level, unlike object detection:
        # - instance_index: this level's own key, index within its own frame.
        # - target_index: the legacy public spelling, index within the whole item.
        # Instances were appended frame-by-frame within sequence-by-sequence order, so
        # both grouping arrays are already contiguous and _running_index applies directly.
        instance_index = _running_index(image_pos_of_instance)
        instance_target_index = _running_index(sequence_of_instance)
        # Derived from the finished counts rather than tracked during the walk: a sequence
        # contributes no instance rows exactly when none of its frames held a detection.
        instances_per_item = np.bincount(sequence_of_instance, minlength=count)
        undetected = np.flatnonzero(instances_per_item == 0).tolist()
        # The parent frame's own key, carried onto the instance row so that the compound
        # key is unique: instance_index alone repeats across the frames of one sequence.
        image_index_of_instance = frame_own_index_arr[image_pos_of_instance]
        # Dense within the sequence, in order of first observation. Tracks were opened
        # sequence-by-sequence, so the grouping array is already contiguous.
        track_index = _running_index(sequence_of_track)
        # Gathered with the -1 markers preserved — clamped first, since a marker would
        # otherwise index from the end and report another track's number. Guarded for a
        # dataset in which nothing is tracked at all: there is then no index to gather from.
        track_index_of_instance = (
            np.where(track_pos_of_instance < 0, -1, track_index[np.maximum(track_pos_of_instance, 0)])
            if n_tracks
            else np.full(n_instances, -1, dtype=np.intp)
        )

        # Per-frame, not per-instance: a video's list-valued metadata has one value per
        # frame. Expanding it across detections would be wrong even when the counts
        # coincide, and dropping it — which is what expanding across detections does
        # whenever they disagree — loses the per-frame factors entirely.
        frames_per_item = np.bincount(sequence_of_frame, minlength=count).astype(int).tolist()
        image_factors, dropped = self._merge_factors(raw, ignore_lists=False, targets_per_item=frames_per_item)
        sequence_factors, _ = self._merge_factors(raw, ignore_lists=True)
        # Same rule as the image-based tasks: a name both merges produced is item metadata
        # replicated onto the frame rows, so keep it once at the sequence level and let
        # propagation do the replicating.
        image_factors = {name: values for name, values in image_factors.items() if name not in sequence_factors}

        track_factors = self._track_factors(rows)
        frame_factors = self._frame_factors(rows)
        # A factor can only be declared at one level, so the structurer's own derived
        # values displace a metadata key of the same name rather than colliding with it:
        # these are read off the frames and the targets themselves, which outranks a
        # per-item dictionary that happens to reuse the spelling.
        derived = {*track_factors, *frame_factors}
        sequence_factors = _without(sequence_factors, derived, "sequence")
        image_factors = {**_without(image_factors, derived, "image"), **frame_factors}

        sequence_block = RowBlock(
            "sequence",
            count,
            reserved_block_columns("sequence", count, item_index=list(range(count)), sequence_index=list(range(count))),
            {"sequence": self._own_positions(count)},
        )
        image_block = RowBlock(
            "image",
            n_frames,
            reserved_block_columns("image", n_frames, item_index=sequence_of_frame, image_index=frame_own_index_arr),
            {**self._inherit(sequence_block.ancestor_pos, sequence_of_frame), "image": self._own_positions(n_frames)},
        )
        track_block = RowBlock(
            "track",
            n_tracks,
            reserved_block_columns(
                "track",
                n_tracks,
                item_index=sequence_of_track,
                track_index=track_index,
                track_id=np.asarray(rows.track_id, dtype=np.intp),
            ),
            {**self._inherit(sequence_block.ancestor_pos, sequence_of_track), "track": self._own_positions(n_tracks)},
        )
        instance_block = RowBlock(
            "instance",
            n_instances,
            reserved_block_columns(
                "instance",
                n_instances,
                item_index=sequence_of_instance,
                target_index=instance_target_index,
                class_label=class_labels,
                score=score_values,
                box=box_values,
                instance_index=instance_index,
                image_index=image_index_of_instance,
                track_index=track_index_of_instance,
                track_id=track_id_values,
            ),
            # The diamond: two parents, so two inherited maps. The image branch supplies
            # ``sequence`` and is spread last, because the track branch would supply it too
            # and an untracked row's track position is a null marker rather than an index.
            # ``track`` is taken from the accumulator directly, negatives intact — the track
            # block's own positions are the identity, so gathering through them would only
            # destroy the markers.
            {
                "track": track_pos_of_instance,
                **self._inherit(image_block.ancestor_pos, image_pos_of_instance),
                "instance": self._own_positions(n_instances),
            },
        )

        _log_items_without_targets(undetected, "instance", count)
        untracked = int(np.count_nonzero(track_pos_of_instance < 0))
        if untracked:
            _logger.info(
                "%d of %d detection(s) carry no track id and contribute no %r rows. Track-level "
                "factors read as null on them, and are excluded from factor analysis at any view "
                "where that happens; Metadata.at('track') reads them in full.",
                untracked,
                n_instances,
                "track",
            )
        _logger.info(
            "MOT dataset: %d sequences, %d frames, %d tracks, %d classes, %d detections",
            count,
            n_frames,
            n_tracks,
            len(np.unique(class_labels)),
            n_instances,
        )
        return StructuredData(
            [sequence_block, image_block, track_block, instance_block],
            {
                "sequence": sequence_factors,
                "image": image_factors,
                "track": track_factors,
                "instance": {},
            },
            dropped,
            raw,
            class_labels,
            sequence_of_instance,
        )


class FactorsStructurer(Structurer):
    """Single-level structuring for instances built from raw factor arrays.

    :meth:`~dataeval.Metadata.from_factors` has no dataset to iterate, so this
    derives from :class:`Structurer` rather than :class:`DatasetStructurer` and
    is driven by :meth:`build_from_arrays`. It still produces a
    :class:`StructuredData`, which is what keeps the reserved column schema
    identical to the dataset path.
    """

    task = "factors"

    def __init__(self, level: FactorLevel = "image") -> None:
        # One level, so it is both the item level and the target level; there is
        # no distinct target level here and hence no ``"target"`` alias, matching
        # image classification.
        self.levels = FactorLevelSchema.of(level)
        self.item_level = level
        self.label_level = level

    def build_from_arrays(
        self,
        factors: Mapping[str, Any],
        class_labels: NDArray[np.intp],
        item_indices: NDArray[np.intp],
    ) -> StructuredData:
        """Bundle pre-built factor arrays into a single-level :class:`StructuredData`.

        Parameters
        ----------
        factors : Mapping[str, Any]
            Factor name to a sequence of values, one per row.
        class_labels : NDArray[np.intp]
            Class label per row; its length defines the block size.
        item_indices : NDArray[np.intp]
            Source item index per row.

        Returns
        -------
        StructuredData
            One block at this structurer's level, with the same reserved columns
            the dataset path produces.
        """
        level = self.label_level
        size = len(class_labels)
        columns = reserved_block_columns(
            level,
            size,
            item_index=item_indices,
            target_index=np.zeros(size, dtype=np.intp),
            class_label=class_labels,
        )
        block = RowBlock(level, size, columns, {level: np.arange(size, dtype=np.intp)})
        named = {safe_column_name(str(name)): values for name, values in factors.items()}
        return StructuredData([block], {level: named}, {}, [], class_labels, item_indices)


# ========== STRUCTURER SELECTION ==========
#
# Selection inspects the *target* alone. MAITE constrains what a target is but
# places no constraint at all on what an item is — a path string, a PIL handle
# and a lazy loader are all valid — so predicating on the item type rejects
# perfectly good datasets. Several of these target protocols do not exist in
# MAITE yet, so the predicates duck-type on the attributes they should carry.


# Target predicates in priority order; the first that matches wins. Entries are
# ordered most specific first, so the tracking predicate — a *positive* check for its
# own target type — sits above the detection entry rather than being carved out of it.
DISPATCH: tuple[tuple[Callable[[Any], bool], type[DatasetStructurer]], ...] = (
    (is_multiobject_tracking_target, MOTStructurer),
    (lambda x: isinstance(x, ObjectDetectionTarget), ODImageStructurer),
    (lambda x: isinstance(x, Array), ICStructurer),
)

# Task names accepted as an explicit override. Narrower than :data:`TASK`, which
# also names the strategies no caller can ask for by name: ``"factors"`` is reached
# through :meth:`~dataeval.Metadata.from_factors` and ``"unknown"`` is only the
# unspecialized base class's default.
TaskOverride = Literal["IC", "OD", "MOT"]

# Explicit task overrides, for datasets whose protocols MAITE has not defined yet.
TASK_STRUCTURERS: Mapping[str, type[DatasetStructurer]] = MappingProxyType(
    {
        "IC": ICStructurer,
        "OD": ODImageStructurer,
        "MOT": MOTStructurer,
    },
)


def select_structurer(  # noqa: C901
    dataset: AnnotatedDataset[tuple[Any, Any, DatumMetadata]],
    task: TaskOverride | None = None,
) -> DatasetStructurer:
    """Choose a structuring strategy for a dataset.

    Parameters
    ----------
    dataset : AnnotatedDataset
        Dataset to inspect. Only the first datum is read.
    task : {"IC", "OD"} or None, default None
        Explicit task override. Matched case-insensitively, so an untyped caller
        may pass ``"od"``. When None the target of the first datum is matched
        against :data:`DISPATCH`.

    Returns
    -------
    DatasetStructurer
        Strategy instance for the detected or requested task, carrying the datum this
        function read so that :meth:`DatasetStructurer.build` does not read it again.

    Raises
    ------
    ValueError
        When ``task`` is unrecognized.
    TypeError
        When no registered predicate matches the dataset's target.

    Notes
    -----
    An empty dataset carries no datum to inspect, so it falls back to image
    classification. This keeps the historical behavior, where an empty dataset
    structured into an empty image-level dataframe rather than failing; pass an
    explicit ``task`` to structure an empty dataset any other way.

    The fallback is silent here. :class:`~dataeval.Metadata` warns about it instead,
    from ``__init__``/``bind``, because that is the only frame that can point a
    ``stacklevel`` at the user's line: structuring is lazy, so by the time this
    function runs the triggering call is an arbitrary attribute access.
    """
    if task is not None:
        key = str(task).upper()
        if key not in TASK_STRUCTURERS:
            raise ValueError(f"Unknown task {task!r}. Supported tasks are {sorted(TASK_STRUCTURERS)}.")
        return TASK_STRUCTURERS[key]()

    if not isinstance(dataset, Sized) or len(dataset) == 0:
        _logger.debug("Cannot infer a task from an empty dataset; assuming image classification.")
        return ICStructurer()

    # Handed to the chosen structurer so a dataset that decodes on __getitem__ pays
    # for item 0 once, not once here and again on the first iteration of build().
    first_datum = dataset[0]
    _, target, _ = first_datum
    for target_check, structurer in DISPATCH:
        if target_check(target):
            _logger.debug("Selected %s for target %s", structurer.__name__, type(target))
            return structurer(first_datum)

    raise TypeError(
        f"Unable to infer a task from target type {type(target).__name__}. "
        f"Pass an explicit task, one of {sorted(TASK_STRUCTURERS)}.",
    )
