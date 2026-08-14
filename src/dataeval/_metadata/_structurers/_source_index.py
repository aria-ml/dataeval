"""Reading row placement out of a source index, when there is no dataset to walk."""

__all__ = []

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from dataeval.types import SourceIndex


@dataclass(frozen=True)
class SourceIndexRows:
    """The rows a :class:`~dataeval.types.SourceIndex` sequence describes.

    A source index labels each value with what it measures rather than relying on a
    positional convention, which is exactly the information needed to lay rows out when
    there is no dataset to read them from. Parsing it once, here, keeps the placement
    rule in a single implementation shared by both of
    :class:`~dataeval.Metadata`'s constructors.

    Attributes
    ----------
    item_positions : NDArray[np.intp]
        Positions within the source index of the per-item entries (``target`` is None),
        ordered as the item-level rows they describe.
    label_positions : NDArray[np.intp]
        Positions within the source index of the per-label entries, ordered as the
        label-level rows they describe.
    item_ids : NDArray[np.intp]
        Source item index of each item-level row.
    label_items : NDArray[np.intp]
        Source item index of each label-level row.
    label_targets : NDArray[np.intp]
        Index of each label-level row within its item.
    """

    item_positions: NDArray[np.intp]
    label_positions: NDArray[np.intp]
    item_ids: NDArray[np.intp]
    label_items: NDArray[np.intp]
    label_targets: NDArray[np.intp]

    @classmethod
    def parse(cls, source_index: Sequence[SourceIndex]) -> "SourceIndexRows":
        """Group a source index into item-level and label-level rows.

        Raises
        ------
        ValueError
            When an entry carries a channel, which has no single-column form, or when
            two entries name the same row.
        """
        # Transposed in one C-level step and carried as arrays from there on. A source
        # index from compute_stats holds one entry per detection, so every Python-level
        # pass over it is paid per detection.
        raw_items, raw_targets, raw_channels = zip(*source_index, strict=True) if source_index else ((), (), ())
        if any(channel is not None for channel in raw_channels):
            raise ValueError(
                "source_index contains per-channel entries, which have no single-column "
                "representation. Reduce channel-wise statistics to one value per row "
                "before adding them.",
            )

        count = len(source_index)
        items = np.fromiter(raw_items, dtype=np.intp, count=count)
        # -1 stands in for the None that marks a per-item entry. It is both the flag that
        # tells the two kinds apart and, sorting below every real target, the key that puts
        # an item's own value ahead of that item's labels — the order compute_stats emits.
        targets = np.fromiter((-1 if t is None else t for t in raw_targets), dtype=np.intp, count=count)

        # Sorting rather than trusting the incoming order is the point of taking a source
        # index at all: rows follow the labels, not the sequence they arrived in. Both
        # sorts are stable, so entries that tie keep their incoming order.
        item_positions = np.flatnonzero(targets < 0)
        item_positions = item_positions[np.argsort(items[item_positions], kind="stable")]
        label_positions = np.flatnonzero(targets >= 0)
        label_positions = label_positions[np.lexsort((targets[label_positions], items[label_positions]))]

        rows = cls(
            item_positions=item_positions,
            label_positions=label_positions,
            item_ids=items[item_positions],
            label_items=items[label_positions],
            label_targets=targets[label_positions],
        )
        rows._reject_duplicate_rows()
        return rows

    def _reject_duplicate_rows(self) -> None:
        """Reject a source index that names the same row twice.

        Two values for one row have no resolution — silently keeping the last would make
        the result depend on the input ordering, which is the very thing a source index
        removes.

        Both blocks leave :meth:`parse` sorted, so a repeat is adjacent and the common
        case is settled by one vectorised comparison. The keys themselves are materialised
        only to name the offenders in the message, where a second pass costs nothing.
        """
        keys_by_kind: tuple[tuple[str, tuple[NDArray[np.intp], ...]], ...] = (
            ("item", (self.item_ids,)),
            ("label", (self.label_items, self.label_targets)),
        )
        for kind, columns in keys_by_kind:
            repeats = np.logical_and.reduce([np.diff(column) == 0 for column in columns])
            if not np.any(repeats):
                continue
            keys = list(zip(*(column.tolist() for column in columns), strict=True))
            repeated = sorted(key for key, total in Counter(keys).items() if total > 1)
            raise ValueError(f"source_index names the same {kind}-level row more than once: {repeated}.")

    @property
    def spans_two_levels(self) -> bool:
        """Whether both kinds of entry are present, and so both levels have rows."""
        return len(self.item_positions) > 0 and len(self.label_positions) > 0

    def parent_positions(self) -> NDArray[np.intp]:
        """Position within the item-level block of each label-level row's own item.

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
                "item a target=None entry, or drop the per-item entries entirely.",
            )
        return positions.astype(np.intp)
