from dataeval.config import get_seed

__all__ = []

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import BitGenerator, Generator, SeedSequence
from numpy.typing import NDArray

from dataeval.protocols import Array
from dataeval.selection._select import Select, Selection, SelectionStage
from dataeval.utils.arrays import as_numpy


class Shuffle(Selection[Any]):
    """
    Select dataset indices in a random order.

    Parameters
    ----------
    seed : int, ArrayLike, SeedSequence, BitGenerator, Generator or None, default None
        Seed for the random number generator. If None, results are not reproducible.

    See Also
    --------
    `NumPy Random Generator <https://numpy.org/doc/stable/reference/random/generator.html>`_
    """

    seed: int | NDArray[Any] | SeedSequence | BitGenerator | Generator | None
    stage = SelectionStage.ORDER

    def __init__(
        self, seed: int | Sequence[int] | Array | SeedSequence | BitGenerator | Generator | None = None
    ) -> None:
        _seed = get_seed() if seed is None else seed
        self.seed = as_numpy(_seed) if isinstance(_seed, Sequence | Array) else _seed

    def __call__(self, dataset: Select[Any]) -> None:
        rng = np.random.default_rng(self.seed)
        rng.shuffle(dataset._selection)
