__all__ = ["GeometricSchedule", "ManualSchedule"]

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from dataeval.utils.arrays import as_numpy


class GeometricSchedule:
    """
    Geometric spacing schedule for sufficiency evaluation.

    Creates evaluation points using geometric progression from 1% to
    100% of dataset length. This provides denser sampling at small
    dataset sizes where performance changes rapidly.

    Parameters
    ----------
    substeps : int
        Number of evaluation points to generate

    Examples
    --------
    >>> schedule = GeometricSchedule(substeps=5)
    >>> schedule.get_steps(dataset_length=100)
    array([  1,   3,  10,  31, 100])
    """

    def __init__(self, substeps: int) -> None:
        self.substeps = substeps

    def get_steps(self, dataset_length: int) -> NDArray[np.intp]:
        """Generate geometric spacing from 1% to 100% of dataset."""
        start = 0.01 * dataset_length
        stop = dataset_length
        return np.geomspace(start, stop, self.substeps, dtype=np.intp)


class ManualSchedule:
    """
    Custom evaluation schedule with user-specified points.

    Allows explicit control over evaluation points, useful for
    comparing specific dataset sizes or focusing on particular
    data regimes.

    Parameters
    ----------
    eval_points : int | Iterable[int] | NDArray[integer]
        Evaluation points. Can be:
        - Single int: evaluate at one size
        - Iterable of ints: evaluate at multiple sizes
        - NumPy array: evaluate at array values

    Raises
    ------
    ValueError
        If eval_points contains non-numeric values

    Examples
    --------
    Single point:

    >>> schedule = ManualSchedule(50)
    >>> schedule.get_steps(dataset_length=100)
    array([50])

    Multiple points:

    >>> schedule = ManualSchedule([10, 50, 100])
    >>> schedule.get_steps(dataset_length=100)
    array([ 10,  50, 100])
    """

    def __init__(self, eval_points: int | Iterable[int] | NDArray[np.intp]) -> None:
        # Convert to list if iterable
        points = list(eval_points) if isinstance(eval_points, Iterable) else [eval_points]

        # Convert to integer array
        arr = as_numpy(points, dtype=np.intp)

        self.eval_points = arr

    def get_steps(self, dataset_length: int) -> NDArray[np.intp]:
        """Return the pre-specified evaluation points."""
        return self.eval_points
