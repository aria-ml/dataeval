__all__ = []

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval.core._calculate import CalculationResult

StatsMap = Mapping[str, NDArray[Any]]


def add_calculation_results(a: StatsMap, b: StatsMap) -> StatsMap:
    return {k: np.concatenate([a[k], b[k]]) for k in a if k in b}


def combine_results(results: CalculationResult | Sequence[CalculationResult]) -> tuple[StatsMap, list[int]]:
    """Combine multiple CalculationResult dicts into one."""
    if isinstance(results, dict):
        return results["stats"], []

    output: StatsMap = {}
    dataset_steps = []
    cur_len = 0
    for r in results:
        output = r["stats"] if not output else add_calculation_results(output, r["stats"])
        # Get length from source_index
        cur_len += len(r["source_index"])
        dataset_steps.append(cur_len)
    if output is None:
        raise TypeError("Cannot combine empty sequence of stats.")
    return output, dataset_steps


def get_dataset_step_from_idx(idx: int, dataset_steps: Sequence[int]) -> tuple[int, int]:
    last_step = 0
    for i, step in enumerate(dataset_steps):
        if idx < step:
            return i, idx - last_step
        last_step = step
    return -1, idx
