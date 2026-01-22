"""
Update strategies inform how the :term:`drift<Drift>` detector classes update the reference data when monitoring.
for drift.
"""

__all__ = ["LastSeenUpdateStrategy", "ReservoirSamplingUpdateStrategy"]


import numpy as np
from numpy.typing import NDArray

from dataeval.protocols import UpdateStrategy
from dataeval.utils.arrays import flatten_samples


class LastSeenUpdateStrategy(UpdateStrategy):
    """
    Updates reference dataset for :term:`drift<Drift>` detector using last seen method.

    Parameters
    ----------
    n : int
        Update with last n instances seen by the detector.
    """

    def __init__(self, n: int) -> None:
        self.n = n

    def __call__(self, x_ref: NDArray[np.float32], x_new: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.concatenate([x_ref, flatten_samples(x_new)], axis=0)[-self.n :]


class ReservoirSamplingUpdateStrategy(UpdateStrategy):
    """
    Updates reference dataset for :term:`drift<Drift>` detector using reservoir sampling method.

    Parameters
    ----------
    n : int
        Update with last n instances seen by the detector.
    """

    def __init__(self, n: int) -> None:
        self.n = n
        self._count = 0

    def __call__(self, x_ref: NDArray[np.float32], x_new: NDArray[np.float32]) -> NDArray[np.float32]:
        if x_new.shape[0] + self._count <= self.n:
            self._count += x_new.shape[0]
            result = np.concatenate([x_ref, flatten_samples(x_new)], axis=0)
            return result[: self.n]

        n_ref = x_ref.shape[0]
        output_size = min(self.n, n_ref + x_new.shape[0])
        shape = (output_size,) + x_new.shape[1:]

        x_reservoir = np.zeros(shape, dtype=x_ref.dtype)
        x_reservoir[:n_ref] = x_ref

        for item in x_new:
            self._count += 1
            if n_ref < self.n:
                x_reservoir[n_ref, :] = item
                n_ref += 1
            else:
                r = np.random.randint(0, self._count)
                if r < self.n:
                    x_reservoir[r, :] = item
        return x_reservoir
