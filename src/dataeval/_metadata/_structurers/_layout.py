"""The positional map from dataframe rows back to the level hierarchy."""

__all__ = []

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval._metadata._structurers._block import RowBlock
from dataeval._metadata._structurers._gather import take
from dataeval.types import FactorLevel


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
            column.extend([None] * size if positions is None else take(values, positions))
        return column
