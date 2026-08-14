"""One level's rows, and the positions that tie them to their ancestors."""

__all__ = []

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval.types import FactorLevel


@dataclass(frozen=True)
class RowBlock:
    """A contiguous run of dataframe rows belonging to a single level.

    Attributes
    ----------
    level : str
        Level every row in this block belongs to.
    size : int
        Number of rows in the block.
    columns : Mapping[str, Sequence[Any] | NDArray[Any]]
        Reserved (non-factor) column values for the block, as arrays or sequences.
    ancestor_pos : Mapping[str, NDArray[np.intp]]
        For the block's own level and each ancestor level, the position of the
        corresponding row within *that* level's block. This is what makes
        downward factor propagation a gather rather than a join.

        A **negative** position marks a row with no ancestor at that level, and
        propagates as None. A level absent from the mapping entirely is the different,
        block-wide statement that no row here has such an ancestor. Both arise from the
        diamond in the level graph: an untracked detection has a frame but no track, so
        its ``track`` position is negative, while a frame row has no ``track`` key at all
        because ``unit`` and ``track`` are siblings.
    """

    level: FactorLevel
    size: int
    columns: Mapping[str, Sequence[Any] | NDArray[Any]]
    ancestor_pos: Mapping[FactorLevel, NDArray[np.intp]]

    def positions_at(self, level: FactorLevel) -> NDArray[np.intp]:
        """Locate this block's rows within ``level``'s block, marked where there are none.

        A level absent from :attr:`ancestor_pos` is the block-wide statement that no row
        here has such an ancestor, so it answers with the no-ancestor marker throughout
        rather than raising: the two ways of saying "this row has no ancestor there" —
        an absent key and a negative position — reach a positional reader as one.

        Parameters
        ----------
        level : str
            Level to locate this block's rows within.

        Returns
        -------
        NDArray[np.intp]
            One position per row in this block, ``-1`` where the row has no ancestor at
            ``level``.
        """
        positions = self.ancestor_pos.get(level)
        if positions is None:
            return np.full(self.size, -1, dtype=np.intp)
        return np.asarray(positions, dtype=np.intp)
