from __future__ import annotations

__all__ = []

from typing import Any, Sequence

import numpy as np
from numpy.random import BitGenerator, Generator, SeedSequence
from numpy.typing import NDArray

from dataeval.data._selection import Select, Selection, SelectionStage
from dataeval.typing import Array
from dataeval.utils._array import as_numpy


class Shuffle(Selection[Any]):
    """
    Shuffle the dataset using a seed.

    Parameters
    ----------
    seed : int, ArrayLike, SeedSequence, BitGenerator, Generator or None, default None
        Seed for the random number generator.

    See Also
    --------
    `NumPy Random Generator <https://numpy.org/doc/stable/reference/random/generator.html>`_
    """

    seed: int | NDArray[Any] | SeedSequence | BitGenerator | Generator | None
    stage = SelectionStage.ORDER

    def __init__(
        self, seed: int | Sequence[int] | Array | SeedSequence | BitGenerator | Generator | None = None
    ) -> None:
        self.seed = as_numpy(seed) if isinstance(seed, (Sequence, Array)) else seed

    def __call__(self, dataset: Select[Any]) -> None:
        rng = np.random.default_rng(self.seed)
        rng.shuffle(dataset._selection)
