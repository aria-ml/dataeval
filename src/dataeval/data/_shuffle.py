from dataeval.config import get_seed

__all__ = []

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import BitGenerator, Generator, SeedSequence
from numpy.typing import NDArray

from dataeval.data._view import Operation, View
from dataeval.protocols import Array
from dataeval.utils._internal import as_numpy


class Shuffle(Operation):
    """
    Select dataset indices in a random order.

    Parameters
    ----------
    seed : int, ArrayLike, SeedSequence, BitGenerator, Generator or None, default None
        Seed for the random number generator. If None, results are not reproducible.

    See Also
    --------
    :class:`numpy.random.Generator`
    """

    seed: int | NDArray[Any] | SeedSequence | BitGenerator | Generator | None

    def __init__(
        self,
        seed: int | Sequence[int] | Array | SeedSequence | BitGenerator | Generator | None = None,
    ) -> None:
        _seed = get_seed() if seed is None else seed
        self.seed = as_numpy(_seed) if isinstance(_seed, Sequence | Array) else _seed

    def apply(self, view: View[Any]) -> None:
        rng = np.random.default_rng(self.seed)
        rng.shuffle(view.selection)
