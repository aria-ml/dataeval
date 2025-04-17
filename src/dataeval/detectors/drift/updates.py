"""
Update strategies inform how the :term:`drift<Drift>` detector classes update the reference data when monitoring.
for drift.
"""

from __future__ import annotations

__all__ = ["LastSeenUpdate", "ReservoirSamplingUpdate"]

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from dataeval.utils._array import flatten


class BaseUpdateStrategy(ABC):
    """
    Updates reference dataset for drift detector

    Parameters
    ----------
    n : int
        Update with last n instances seen by the detector.
    """

    def __init__(self, n: int) -> None:
        self.n = n

    @abstractmethod
    def __call__(self, x_ref: NDArray[np.float32], x_new: NDArray[np.float32], count: int) -> NDArray[np.float32]: ...


class LastSeenUpdate(BaseUpdateStrategy):
    """
    Updates reference dataset for :term:`drift<Drift>` detector using last seen method.

    Parameters
    ----------
    n : int
        Update with last n instances seen by the detector.
    """

    def __call__(self, x_ref: NDArray[np.float32], x_new: NDArray[np.float32], count: int) -> NDArray[np.float32]:
        return np.concatenate([x_ref, flatten(x_new)], axis=0)[-self.n :]


class ReservoirSamplingUpdate(BaseUpdateStrategy):
    """
    Updates reference dataset for :term:`drift<Drift>` detector using reservoir sampling method.

    Parameters
    ----------
    n : int
        Update with last n instances seen by the detector.
    """

    def __call__(self, x_ref: NDArray[np.float32], x_new: NDArray[np.float32], count: int) -> NDArray[np.float32]:
        if x_new.shape[0] + count <= self.n:
            return np.concatenate([x_ref, flatten(x_new)], axis=0)

        n_ref = x_ref.shape[0]
        output_size = min(self.n, n_ref + x_new.shape[0])
        shape = (output_size,) + x_new.shape[1:]

        x_reservoir = np.zeros(shape, dtype=x_ref.dtype)
        x_reservoir[:n_ref] = x_ref

        for item in x_new:
            count += 1
            if n_ref < self.n:
                x_reservoir[n_ref, :] = item
                n_ref += 1
            else:
                r = np.random.randint(0, count)
                if r < self.n:
                    x_reservoir[r, :] = item
        return x_reservoir
