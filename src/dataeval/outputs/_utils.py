from __future__ import annotations

__all__ = []

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from dataeval.outputs._base import Output


@dataclass(frozen=True)
class TrainValSplit:
    """
    Dataclass containing train and validation indices.

    Attributes
    ----------
    train: NDArray[np.intp]
        Indices for the training set
    val: NDArray[np.intp]
        Indices for the validation set
    """

    train: NDArray[np.intp]
    val: NDArray[np.intp]


@dataclass(frozen=True)
class SplitDatasetOutput(Output):
    """
    Output class containing test indices and a list of TrainValSplits.

    Attributes
    ----------
    test: NDArray[np.intp]
        Indices for the test set
    folds: Sequence[TrainValSplit]
        List of train and validation split indices
    """

    test: NDArray[np.intp]
    folds: Sequence[TrainValSplit]
