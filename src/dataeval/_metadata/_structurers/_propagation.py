"""Building the ancestor position maps that make propagation a gather."""

__all__ = []

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from dataeval.types import FactorLevel


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
