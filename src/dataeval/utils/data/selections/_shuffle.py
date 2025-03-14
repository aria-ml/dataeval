from __future__ import annotations

__all__ = []

from typing import Any

import numpy as np

from dataeval.utils.data._selection import Select, Selection, SelectionStage


class Shuffle(Selection[Any, Any]):
    """
    Shuffle the dataset using a seed.

    Parameters
    ----------
    seed
        Seed for the random number generator.
    """

    stage = SelectionStage.ORDER

    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, dataset: Select[Any, Any]) -> None:
        rng = np.random.default_rng(self.seed)
        rng.shuffle(dataset._selection)
