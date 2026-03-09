"""
Update strategies inform how the drift detector classes update the reference data.

These strategies are used when monitoring for drift.
"""

__all__ = ["LastSeenUpdateStrategy", "ReservoirSamplingUpdateStrategy"]


import numpy as np
from numpy.typing import NDArray

from dataeval.protocols import UpdateStrategy
from dataeval.types import ReprMixin
from dataeval.utils._internal import flatten_samples


class LastSeenUpdateStrategy(ReprMixin, UpdateStrategy):
    """
    Updates reference dataset for :term:`drift<Drift>` detector using last seen method.

    Parameters
    ----------
    n : int
        Update with last n instances seen by the detector.
    """

    def __init__(self, n: int) -> None:
        self.n = n

    def __call__(self, reference_data: NDArray[np.float32], data: NDArray[np.float32]) -> NDArray[np.float32]:
        """Update reference data using last seen instances."""
        return np.concatenate([reference_data, flatten_samples(data)], axis=0)[-self.n :]


class ReservoirSamplingUpdateStrategy(ReprMixin, UpdateStrategy):
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

    def __call__(self, reference_data: NDArray[np.float32], data: NDArray[np.float32]) -> NDArray[np.float32]:
        """Update reference data using reservoir sampling."""
        if data.shape[0] + self._count <= self.n:
            self._count += data.shape[0]
            result = np.concatenate([reference_data, flatten_samples(data)], axis=0)
            return result[: self.n]

        n_ref = reference_data.shape[0]
        output_size = min(self.n, n_ref + data.shape[0])
        shape = (output_size,) + data.shape[1:]

        x_reservoir = np.zeros(shape, dtype=reference_data.dtype)
        x_reservoir[:n_ref] = reference_data

        for item in data:
            self._count += 1
            if n_ref < self.n:
                x_reservoir[n_ref, :] = item
                n_ref += 1
            else:
                r = np.random.randint(0, self._count)
                if r < self.n:
                    x_reservoir[r, :] = item
        return x_reservoir
