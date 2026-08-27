"""Reading row placement out of a source index, when there is no dataset to walk."""

__all__ = []

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from dataeval.types import FactorLevel, SourceIndex
from dataeval.types._factors import _FACTOR_LEVEL_HIERARCHY

# Stands in for the ``None`` of an address that carries no key. It is both the flag that
# tells an item's own row from a row within that item and, sorting below every real key,
# what puts an item's value ahead of its labels' — the order ``compute_stats`` emits.
_UNKEYED = -1

# Levels as small integers, coarsest first, so that grouping and ordering are numpy work
# rather than string comparisons. ``_FACTOR_LEVEL_HIERARCHY`` declares the canonical order
# and is the single place it lives; taking the codes from it means ``np.unique`` below
# yields levels coarsest-first for free.
_LEVEL_CODES: Mapping[FactorLevel | None, int] = MappingProxyType({
    None: -1,
    **{level: rank for rank, level in enumerate(_FACTOR_LEVEL_HIERARCHY)},
})
_CODE_LEVELS: tuple[FactorLevel, ...] = tuple(_FACTOR_LEVEL_HIERARCHY)


@dataclass(frozen=True)
class LevelRows:
    """The rows one level's addresses describe, ordered as those rows sit.

    Attributes
    ----------
    positions : NDArray[np.intp]
        Positions within the source index of the entries at this level, ordered as the
        rows they describe. This is the gather that puts incoming values in row order.
    items : NDArray[np.intp]
        Source item index of each row.
    keys : NDArray[np.intp]
        Which row within that item, as the level's own key column names it, or
        :data:`_UNKEYED` where the item alone names the row.
    """

    positions: NDArray[np.intp]
    items: NDArray[np.intp]
    keys: NDArray[np.intp]

    def __len__(self) -> int:
        return len(self.positions)

    @property
    def is_keyed(self) -> bool:
        """Whether these rows are named by a key rather than by their item alone."""
        return bool(len(self.keys)) and bool(np.any(self.keys >= 0))


_NO_ROWS = LevelRows(
    positions=np.empty(0, dtype=np.intp),
    items=np.empty(0, dtype=np.intp),
    keys=np.empty(0, dtype=np.intp),
)


@dataclass(frozen=True)
class SourceIndexRows:
    """The rows a :class:`~dataeval.types.SourceIndex` sequence describes.

    A source index labels each value with what it measures rather than relying on a
    positional convention, which is exactly the information needed to lay rows out when
    there is no dataset to read them from. Parsing it once, here, keeps the placement
    rule in a single implementation shared by both of
    :class:`~dataeval.Metadata`'s constructors.

    This is also where an address is **canonicalized**. An unstated level is the
    task-generic one, so ``SourceIndex(3)`` and ``SourceIndex(3, None, "sequence")`` name
    one row on a video and have to land in one group — otherwise a source index naming a
    row twice, once each way, would pass the duplicate check and place two values on it.

    Attributes
    ----------
    item_level : FactorLevel
        The level one dataset item sits at, which an unkeyed address with no stated level
        resolves to.
    label_level : FactorLevel
        The level one labelled thing sits at, which a keyed address with no stated level
        resolves to.
    by_level : Mapping[FactorLevel, LevelRows]
        The rows at each level the source index names, coarsest level first. Levels the
        source index says nothing about are absent rather than empty.
    """

    item_level: FactorLevel
    label_level: FactorLevel
    by_level: Mapping[FactorLevel, LevelRows]

    @classmethod
    def parse(
        cls,
        source_index: Sequence[SourceIndex],
        *,
        item_level: FactorLevel = "unit",
        label_level: FactorLevel = "instance",
    ) -> "SourceIndexRows":
        """Group a source index by the level each entry addresses.

        Parameters
        ----------
        source_index : Sequence[SourceIndex]
            The addresses to group.
        item_level : FactorLevel, default "unit"
            What an unkeyed address with no stated level names. The default is the
            two-level reading every dataset-free constructor uses; a caller placing into
            existing rows passes that metadata's own item level.
        label_level : FactorLevel, default "instance"
            What a keyed address with no stated level names.

        Raises
        ------
        ValueError
            When two entries name the same row.
        """
        # Transposed in one C-level step and carried as arrays from there on. A source
        # index from compute_stats holds one entry per detection, so every Python-level
        # pass over it is paid per detection.
        #
        # The transposed columns are then taken by field *name*. Unpacking them
        # positionally is what made the third slot's identity load-bearing here, so that
        # retiring a field silently rebound the names beside it; naming them costs one
        # pass over the fields rather than one over the entries.
        columns = dict(zip(SourceIndex._fields, zip(*source_index, strict=True), strict=True)) if source_index else {}
        raw_items = columns.get("item", ())
        raw_keys = columns.get("key", ())
        raw_levels = columns.get("level", ())

        count = len(source_index)
        items = np.fromiter(raw_items, dtype=np.intp, count=count)
        keys = np.fromiter((_UNKEYED if k is None else k for k in raw_keys), dtype=np.intp, count=count)
        # Resolving here rather than per consumer is what makes the two spellings of one
        # address the same group. Whether any entry states a level at all is settled by one
        # C-level pass over the column, because the answer is no for everything
        # ``compute_stats`` emits and reading the levels one at a time to learn that is a
        # third Python-level pass over a source index holding one entry per detection.
        item_code = np.int8(_LEVEL_CODES[item_level])
        label_code = np.int8(_LEVEL_CODES[label_level])
        if set(raw_levels) <= {None}:
            codes = np.where(keys < 0, item_code, label_code).astype(np.int8)
        else:
            codes = np.fromiter((_LEVEL_CODES[level] for level in raw_levels), dtype=np.int8, count=count)
            unstated = codes < 0
            codes[unstated & (keys < 0)] = item_code
            codes[unstated & (keys >= 0)] = label_code

        # ``np.unique`` returns the codes sorted, and the codes are the canonical
        # coarsest-to-finest order, so the levels come back in that order without a second
        # sort. Row order within a level follows the labels rather than the sequence the
        # entries arrived in, which is the point of taking a source index at all; lexsort
        # is stable, so entries that tie keep their incoming order.
        by_level: dict[FactorLevel, LevelRows] = {}
        for code in np.unique(codes):
            positions = np.flatnonzero(codes == code)
            positions = positions[np.lexsort((keys[positions], items[positions]))]
            by_level[_CODE_LEVELS[code]] = LevelRows(
                positions=positions,
                items=items[positions],
                keys=keys[positions],
            )

        rows = cls(item_level=item_level, label_level=label_level, by_level=by_level)
        rows._reject_duplicate_rows()
        return rows

    def _reject_duplicate_rows(self) -> None:
        """Reject a source index that names the same row twice.

        Two values for one row have no resolution — silently keeping the last would make
        the result depend on the input ordering, which is the very thing a source index
        removes.

        Every level leaves :meth:`parse` sorted, so a repeat is adjacent and the common
        case is settled by one vectorised comparison per level. The keys themselves are
        materialised only to name the offenders in the message, where a second pass costs
        nothing.

        A level whose rows carry no key compares on the item alone, which falls out of the
        same comparison: every one of its keys is :data:`_UNKEYED`, so that half is always
        equal and the item decides.
        """
        for level, rows in self.by_level.items():
            repeats = (np.diff(rows.items) == 0) & (np.diff(rows.keys) == 0)
            if not np.any(repeats):
                continue
            named = list(zip(rows.items.tolist(), rows.keys.tolist(), strict=True))
            # Ordered on the sentinel, then rendered: a level can hold keyed and unkeyed
            # rows at once — the coinciding-levels schema does — and sorting the rendered
            # form would compare a None against an int rather than name the offenders.
            repeated = [
                (item, None if key < 0 else key)
                for item, key in sorted(pair for pair, total in Counter(named).items() if total > 1)
            ]
            raise ValueError(f"source_index names the same {level}-level row more than once: {repeated}.")

    def reject_levels_beyond_two(self) -> None:
        """Reject addresses this source index cannot be *built* from, only placed by.

        Placing values into rows that already exist can honour any level: the rows carry
        their own parentage and the address only has to name one of them. **Building** the
        rows from addresses alone cannot. An address deliberately says nothing about
        parentage — that is what lets one tuple name a row at any level of a graph that
        branches — so nothing in a source index says which frame or which track a detection
        belongs to, and a store with a level between the item and the label cannot be
        reconstructed from it. :meth:`parent_positions` works two-level only because there
        is exactly one candidate parent per item there.

        A stated level that agrees with the two-level reading is accepted, so the fully
        explicit spelling of an ordinary result builds exactly as the minimal one does.

        Raises
        ------
        ValueError
            When an address names a level between the two, or states one of the two in a
            way its key contradicts.
        """
        beyond = [level for level in self.by_level if level not in (self.item_level, self.label_level)]
        contradictory: list[str] = []
        if self.item_rows.is_keyed:
            contradictory.append(f"{self.item_level!r} with a key")
        if len(self.label_rows) and not bool(np.all(self.label_rows.keys >= 0)):
            contradictory.append(f"{self.label_level!r} with no key")
        if not beyond and not contradictory:
            return

        named = ", ".join([*(repr(level) for level in beyond), *contradictory])
        raise ValueError(
            f"source_index names {named}, and metadata built from a source index alone has only "
            f"{self.item_level!r} and {self.label_level!r} rows. An address names a row without "
            "saying what it sits inside, so a level between the two cannot be built from one — "
            "construct the metadata from its dataset, then place these values with "
            "add_factors(source_index=...), which addresses every level the metadata has.",
        )

    @property
    def item_rows(self) -> LevelRows:
        """The rows at :attr:`item_level`, empty when the source index names none."""
        return self.by_level.get(self.item_level, _NO_ROWS)

    @property
    def label_rows(self) -> LevelRows:
        """The rows at :attr:`label_level`, empty when the source index names none."""
        return self.by_level.get(self.label_level, _NO_ROWS)

    @property
    def item_positions(self) -> NDArray[np.intp]:
        """Positions of the item-level entries, ordered as the rows they describe."""
        return self.item_rows.positions

    @property
    def label_positions(self) -> NDArray[np.intp]:
        """Positions of the label-level entries, ordered as the rows they describe."""
        return self.label_rows.positions

    @property
    def item_ids(self) -> NDArray[np.intp]:
        """Source item index of each item-level row."""
        return self.item_rows.items

    @property
    def label_items(self) -> NDArray[np.intp]:
        """Source item index of each label-level row."""
        return self.label_rows.items

    @property
    def label_targets(self) -> NDArray[np.intp]:
        """Index of each label-level row within its item."""
        return self.label_rows.keys

    @property
    def spans_two_levels(self) -> bool:
        """Whether both kinds of entry are present, and so both levels have rows."""
        return len(self.item_positions) > 0 and len(self.label_positions) > 0

    def parent_positions(self) -> NDArray[np.intp]:
        """Position within the item-level block of each label-level row's own item.

        Two-level only, and deliberately so: it is read by the dataset-free *constructor*,
        which builds one item level and one label level and can look a parent up because
        there is exactly one candidate per item. Addresses carry no parentage — that is
        what lets one tuple name a row at any level of a graph that branches — so a store
        with a level between the two has to be built from a dataset rather than from
        addresses.

        Raises
        ------
        ValueError
            When a label-level entry names an item that has no item-level entry, where
            the label row has no parent to inherit from.
        """
        positions = np.searchsorted(self.item_ids, self.label_items)
        orphaned = positions >= len(self.item_ids)
        # Guarded rather than clamped unconditionally: with no per-item entries at all
        # there is no last position to clamp to, and ``item_ids[-1]`` on an empty array
        # raises IndexError in place of the ValueError this promises. Every label is
        # orphaned there anyway, which the line above has already established.
        if len(self.item_ids):
            orphaned |= self.item_ids[np.minimum(positions, len(self.item_ids) - 1)] != self.label_items
        if orphaned.any():
            missing = sorted(set(self.label_items[orphaned].tolist()))
            raise ValueError(
                f"source_index has per-label entries for item(s) {missing} but no per-item entry "
                "for them, so those labels have no item row to hang from. Give every labelled "
                "item a key=None entry, or drop the per-item entries entirely.",
            )
        return positions.astype(np.intp)
