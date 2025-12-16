__all__ = []

import numpy as np
from numpy.typing import NDArray

from dataeval.protocols import ArrayLike


class ResultAggregator:
    """
    Accumulates and aggregates evaluation results across runs and substeps.

    Handles storage initialization, automatic shape detection, and proper
    accumulation of both scalar and array-valued metrics.

    Parameters
    ----------
    runs : int
        Number of independent runs
    substeps : int
        Number of evaluation steps per run

    Examples
    --------
    >>> aggregator = ResultAggregator(runs=3, substeps=5)
    >>> aggregator.add_result(run=0, step=0, metric_name="accuracy", value=0.95)
    >>> results = aggregator.get_results()
    >>> results["accuracy"].shape
    (3, 5)

    Notes
    -----
    The aggregator automatically detects metric types:
    - Scalar values (float or 1-element array) → shape (runs, substeps)
    - Array values (multi-element) → shape (runs, substeps, array_length)

    Metric names and types are determined dynamically from the evaluation
    strategy's output, allowing complete flexibility in what metrics are tracked.
    """

    def __init__(self, runs: int, substeps: int) -> None:
        self.runs = runs
        self.substeps = substeps
        self._storage: dict[str, NDArray[np.floating]] = {}

    def add_result(self, run: int, step: int, metric_name: str, value: ArrayLike) -> None:
        """
        Add a single evaluation result.

        Parameters
        ----------
        run : int
            Run index (0-based)
        step : int
            Step index (0-based)
        metric_name : str
            Name of the metric
        value : float | NDArray
            Metric value (scalar or array)
        """
        # Convert value to array and flatten
        value_array = np.array(value).ravel()

        # Initialize storage if first time seeing this metric
        if metric_name not in self._storage:
            # Determine shape based on value
            if len(value_array) == 1:
                # Scalar metric: (runs, substeps)
                shape = (self.runs, self.substeps)
            else:
                # Array metric: (runs, substeps, array_length)
                shape = (self.runs, self.substeps, len(value_array))

            self._storage[metric_name] = np.zeros(shape)

        # Store the value
        self._storage[metric_name][run, step] = value_array

    def get_results(self) -> dict[str, NDArray[np.floating]]:
        """
        Get accumulated results.

        Returns
        -------
        dict[str, NDArray]
            Dictionary mapping metric names to result arrays.
            Shapes are (runs, substeps) for scalars or
            (runs, substeps, array_length) for arrays.
        """
        return self._storage.copy()
