"""Structuring for instances built from raw factor arrays rather than a dataset."""

__all__ = []

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval._metadata._structurers._base import Structurer
from dataeval._metadata._structurers._block import RowBlock
from dataeval._metadata._structurers._data import StructuredData
from dataeval._metadata._structurers._detection import ODImageStructurer
from dataeval._metadata._structurers._ordering import index_within_parent
from dataeval._metadata._structurers._propagation import PropagationMixin
from dataeval._metadata._structurers._reserved import reserved_block_columns, safe_column_name
from dataeval._metadata._structurers._source_index import SourceIndexRows
from dataeval.exceptions import ShapeMismatchError
from dataeval.types import FactorLevel, FactorLevelSchema


class FactorsStructurer(PropagationMixin, Structurer):
    """Structuring for instances built from raw factor arrays rather than a dataset.

    :meth:`~dataeval.Metadata.from_factors` has no dataset to iterate, so this
    derives from :class:`Structurer` rather than :class:`DatasetStructurer` and
    is driven by :meth:`build_from_arrays` or :meth:`build_from_source_index`. It still
    produces a :class:`StructuredData`, which is what keeps the reserved column schema
    identical to the dataset path.

    One level or two, depending on what the caller can say about the rows. Bare factor
    arrays describe a single level, since nothing in them distinguishes an item from a
    label. A source index does distinguish them, and when it carries both kinds this
    builds the same two-block ``unit``/``instance`` shape :class:`ODImageStructurer`
    produces — which is what lets :func:`~dataeval.core.compute_stats` output over an
    object detection dataset be imported without the dataset itself.

    Parameters
    ----------
    level : str, default "unit"
        Level the rows sit at when there is only one.
    rows : SourceIndexRows or None, default None
        Parsed source index, when the caller has one. Two levels are declared only when
        it carries both kinds of entry.
    """

    task = "factors"

    def __init__(self, level: FactorLevel = "unit", rows: "SourceIndexRows | None" = None) -> None:
        self._rows = rows
        if rows is not None and rows.spans_two_levels:
            self._declare_two_levels()
            return
        # One level, so it is both the item level and the target level; there is
        # no distinct target level here and hence no ``"target"`` alias, matching
        # image classification.
        resolved = level if rows is None or len(rows.label_positions) == 0 else "instance"
        self.levels = FactorLevelSchema.of(resolved)
        self.item_level = resolved
        self.label_level = resolved
        self.multi_target = False
        # The ``"image"`` alias is unconditional on the base class because every *task*
        # has a unit level for it to resolve to. A factors-only instance need not: its
        # single level is whatever the caller asked for, and below the unit level the
        # alias is not merely unused but actively wrong — it would announce that
        # ``"image"`` is now spelled ``"unit"`` on an instance holding no unit rows,
        # advice that can never apply and that turns ``rows_at("image")`` into a warning
        # followed by a failure to resolve.
        self.legacy_level_aliases = Structurer.legacy_level_aliases if resolved == "unit" else MappingProxyType({})

    def _declare_two_levels(self) -> None:
        """Declare the two-block ``unit``/``instance`` shape a source index can address.

        Items are units and labels are instances, the same pairing every task whose item
        *is* one unit declares; a source index cannot address any other, since it names a
        row by item and target alone and has no way to say which frame or which track.
        Copied from the declaration rather than restated so the two cannot drift —
        including the ``"target"`` alias, which a caller reaching this through imported OD
        stats has every reason to still spell. Borrowing a class-level declaration also
        borrows the ``__init_subclass__`` check that already validated it. ``unit_type``
        is deliberately left at the base default: a bare source index says nothing about
        the medium its items came from.
        """
        self.levels = ODImageStructurer.levels
        self.item_level = ODImageStructurer.item_level
        self.label_level = ODImageStructurer.label_level
        self.multi_target = ODImageStructurer.multi_target
        self.legacy_level_aliases = ODImageStructurer.legacy_level_aliases

    @classmethod
    def for_shape(cls, item_level: FactorLevel, label_level: FactorLevel) -> "FactorsStructurer":
        """Rebuild the declaration for rows that have already been laid out.

        :meth:`~dataeval.Metadata.load` restores rows some earlier run produced, so it
        needs the level declaration without the source index that originally implied it —
        a source index describes values waiting to be placed, and by then they are placed.

        The two shapes :meth:`__init__` can reach are told apart by whether the item and
        label levels differ, which is the only thing the two branches disagree about that
        survives into the rows. Both are produced by the same code paths ``__init__`` uses,
        so a restored declaration cannot drift from a freshly built one.

        Parameters
        ----------
        item_level : str
            Level one source item corresponds to.
        label_level : str
            Level whose rows carry ``class_label``.

        Returns
        -------
        FactorsStructurer
            A structurer declaring that shape, holding no source index. It can describe
            the rows but not build any: :meth:`build_from_source_index` raises on it.

        Raises
        ------
        ValueError
            When the two levels name a shape this structurer never produces.
        """
        if item_level == label_level:
            return cls(item_level)
        if (item_level, label_level) != (ODImageStructurer.item_level, ODImageStructurer.label_level):
            raise ValueError(
                f"A factors-only instance is laid out at one level, or at "
                f"{ODImageStructurer.item_level!r} and {ODImageStructurer.label_level!r}; "
                f"{item_level!r} and {label_level!r} is neither.",
            )
        structurer = cls()
        structurer._declare_two_levels()
        return structurer

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
        return self._single_block(
            factors,
            class_labels,
            item_indices,
            # Derived rather than assumed to be 0, as the dataset structurers derive it.
            # ``item_indices`` may name one item more than once — several detections
            # sharing a unit is what it exists for — and a constant 0 would give every
            # such row the same ``(item_index, target_index)`` key, which is the identity
            # rows are matched on when a source index is placed against them later. Order-
            # independent, unlike the dataset paths': nothing obliges a caller to hand over
            # rows grouped by item. One row per item, the default, still indexes to 0
            # throughout.
            target_index=index_within_parent(item_indices),
        )

    def _single_block(
        self,
        factors: Mapping[str, Any],
        class_labels: NDArray[np.intp],
        item_indices: NDArray[np.intp],
        **keyed: Any,
    ) -> StructuredData:
        """Bundle one block's worth of already-gathered values into a :class:`StructuredData`.

        Sole producer of this structurer's single-level shape, so the two ways of reaching
        it — bare arrays and a single-kind source index — cannot disagree about the
        reserved columns. Only the level-key columns differ between them, and those are
        the caller's to supply.
        """
        level = self.label_level
        size = len(class_labels)
        columns = reserved_block_columns(level, size, item_index=item_indices, class_label=class_labels, **keyed)
        block = RowBlock(level, size, columns, {level: self._own_positions(size)})
        named = {safe_column_name(str(name)): values for name, values in factors.items()}
        return StructuredData([block], {level: named}, {}, [], class_labels, item_indices)

    def build_from_source_index(
        self,
        factors: Mapping[str, NDArray[Any]],
        class_labels: NDArray[np.intp] | None,
    ) -> StructuredData:
        """Lay out rows from the source index this structurer was built with.

        Every factor array holds one value per source-index entry, and each value is
        placed on the row its entry names. When the index carries both kinds of entry the
        result has two blocks, and each factor is split into one column per level named
        ``<level>_<name>`` — a single column cannot hold both a value per unit and a
        value per instance, and each half is a distinct measurement anyway. The naming
        matches :meth:`~dataeval.Metadata.add_factors`, so the same statistics carry the
        same names whether or not a dataset is bound.

        Parameters
        ----------
        factors : Mapping[str, NDArray[Any]]
            Factor name to one value per source-index entry.
        class_labels : NDArray[np.intp] or None
            Class label per label-level row, or None for a single class.

        Returns
        -------
        StructuredData
            One block per level the source index covers.

        Raises
        ------
        ShapeMismatchError
            When ``class_labels`` does not have one entry per label-level row.
        """
        rows = self._rows
        if rows is None:
            raise ValueError("build_from_source_index requires a structurer built with a source index.")
        if not rows.spans_two_levels:
            return self._build_single_level(factors, class_labels, rows)
        return self._build_two_levels(factors, class_labels, rows)

    def _build_single_level(
        self,
        factors: Mapping[str, NDArray[Any]],
        class_labels: NDArray[np.intp] | None,
        rows: SourceIndexRows,
    ) -> StructuredData:
        """Lay out a source index that describes only items, or only labels.

        Item-level rows leave ``target_index`` null rather than zero, matching
        :meth:`_build_two_levels` and every dataset structurer. Downstream code still
        reads that nullness as "this row is not a target" — see
        :class:`~dataeval.quality.Outliers` — so a zero there would misreport a
        per-item result as a per-target one.
        """
        labelled = len(rows.label_positions) > 0
        positions = rows.label_positions if labelled else rows.item_positions
        item_indices = rows.label_items if labelled else rows.item_ids
        keyed: dict[str, Any] = (
            {"target_index": rows.label_targets, "instance_index": rows.label_targets} if labelled else {}
        )

        labels = self._checked_labels(class_labels, len(positions), self.label_level)
        gathered = {name: values[positions] for name, values in factors.items()}
        return self._single_block(gathered, labels, item_indices, **keyed)

    def _build_two_levels(
        self,
        factors: Mapping[str, NDArray[Any]],
        class_labels: NDArray[np.intp] | None,
        rows: SourceIndexRows,
    ) -> StructuredData:
        """Lay out a source index that describes both items and labels."""
        unit_count, instance_count = len(rows.item_positions), len(rows.label_positions)
        labels = self._checked_labels(class_labels, instance_count, "instance")
        parents = rows.parent_positions()

        unit_block = RowBlock(
            "unit",
            unit_count,
            reserved_block_columns("unit", unit_count, item_index=rows.item_ids),
            {"unit": self._own_positions(unit_count)},
        )
        instance_block = RowBlock(
            "instance",
            instance_count,
            reserved_block_columns(
                "instance",
                instance_count,
                item_index=rows.label_items,
                target_index=rows.label_targets,
                class_label=labels,
                instance_index=rows.label_targets,
            ),
            {
                **self._inherit(unit_block.ancestor_pos, parents),
                "instance": self._own_positions(instance_count),
            },
        )
        levelled: dict[FactorLevel, Mapping[str, Any]] = {
            "unit": {safe_column_name(f"unit_{name}"): values[rows.item_positions] for name, values in factors.items()},
            "instance": {
                safe_column_name(f"instance_{name}"): values[rows.label_positions] for name, values in factors.items()
            },
        }
        return StructuredData([unit_block, instance_block], levelled, {}, [], labels, rows.label_items)

    @staticmethod
    def _checked_labels(class_labels: NDArray[np.intp] | None, size: int, level: FactorLevel) -> NDArray[np.intp]:
        """Validate class labels against the level they describe, defaulting to one class."""
        if class_labels is None:
            return np.zeros(size, dtype=np.intp)
        if len(class_labels) != size:
            raise ShapeMismatchError(
                f"class_labels length {len(class_labels)} does not match the {size} {level}-level "
                "rows the source index describes.",
            )
        return class_labels
