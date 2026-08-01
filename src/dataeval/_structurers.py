"""Pluggable structuring strategies that turn a dataset into levelled metadata rows.

The core :class:`~dataeval.Metadata` engine is task agnostic: it consumes a
:class:`StructuredData` bundle and never inspects the dataset itself. Everything
that depends on what a dataset item is (e.g. image) and
where the labels sit (e.g. image, or instance) lives in a :class:`Structurer`.
"""

__all__ = [
    "RESERVED_COLUMNS",
    "TASK",
    "DatasetStructurer",
    "FactorsStructurer",
    "ICStructurer",
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
from collections.abc import Callable, Mapping, Sequence, Sized
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from dataeval.protocols import (
    AnnotatedDataset,
    Array,
    DatumMetadata,
    ObjectDetectionTarget,
    ProgressCallback,
)
from dataeval.types import FactorLevel, FactorLevelSchema
from dataeval.utils._internal import as_numpy, merge_metadata

_logger = logging.getLogger(__name__)

# Task identifier of a structuring strategy.
TASK = Literal["IC", "OD", "factors", "unknown"]


# Columns the metadata dataframe has always carried. Retained verbatim because
# they are public surface: downstream code reads ``dataframe["item_index"]``.
LEGACY_COLUMNS: tuple[str, ...] = ("item_index", "target_index", "class_label", "score", "box")

# Columns introduced alongside the expanded level schema. ``level`` tags each row
# with the level it belongs to and replaces the old "target_index is null" test;
# the remaining columns carry the components of a level's compound key. Only
# columns a structurer actually writes belong here — a name reserved for a level
# that does not exist yet costs a user their metadata key for nothing.
LEVEL_COLUMNS: tuple[str, ...] = (
    "level",
    "instance_index",
)

# Factor names colliding with any of these are prefixed with ``metadata_``. This is
# wider than the historical five-column set: the level key columns are just as
# load-bearing, so a metadata key named ``level`` or ``instance_index`` is renamed
# rather than allowed to clobber the column. :data:`LEGACY_COLUMNS` still holds the
# original tuple for callers that need it.
RESERVED_COLUMNS: tuple[str, ...] = LEGACY_COLUMNS + LEVEL_COLUMNS

# The level-key columns, i.e. everything in LEVEL_COLUMNS that is not the level tag
# itself. A block carries the key components of its own level and no others, so
# these are emitted only when supplied.
_LEVEL_KEY_COLUMNS: tuple[str, ...] = tuple(name for name in LEVEL_COLUMNS if name != "level")


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
        :data:`LEVEL_COLUMNS` are emitted only when supplied, since a block
        carries the key components of its own level and no others.

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
    columns.update({name: _as_column(values[name]) for name in _LEVEL_KEY_COLUMNS if values.get(name) is not None})
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
    """Gather ``values`` at ``positions``, tolerating arrays, lists and other sequences."""
    if isinstance(values, np.ndarray):
        return values[positions].tolist()
    sequence = values if isinstance(values, (list, tuple)) else list(values)
    return [sequence[position] for position in positions]


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

    def expand(self, values: Any, level: FactorLevel) -> list[Any]:
        """Spread values defined at ``level`` across every dataframe row.

        Rows at ``level`` receive their own value, rows at descendant levels
        receive their ancestor's value, and every other row receives None.

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

    Used by object detection strategies.
    """

    @staticmethod
    def _instance_arrays(
        target: ObjectDetectionTarget,
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
# ordered most specific first, so a future tracking predicate — a *positive*
# check for its own target type — sits above the detection entry rather than
# being carved out of it.
DISPATCH: tuple[tuple[Callable[[Any], bool], type[DatasetStructurer]], ...] = (
    (lambda x: isinstance(x, ObjectDetectionTarget), ODImageStructurer),
    (lambda x: isinstance(x, Array), ICStructurer),
)

# Task names accepted as an explicit override. Narrower than :data:`TASK`, which
# also names the strategies no caller can ask for by name: ``"factors"`` is reached
# through :meth:`~dataeval.Metadata.from_factors` and ``"unknown"`` is only the
# unspecialized base class's default.
TaskOverride = Literal["IC", "OD"]

# Explicit task overrides, for datasets whose protocols MAITE has not defined yet.
TASK_STRUCTURERS: Mapping[str, type[DatasetStructurer]] = MappingProxyType(
    {
        "IC": ICStructurer,
        "OD": ODImageStructurer,
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
