from __future__ import annotations

from copy import deepcopy
from typing import Sequence, TypeVar

import numpy as np

from dataeval._internal.metrics.stats.base import BaseStatsOutput

TStatsOutput = TypeVar("TStatsOutput", bound=BaseStatsOutput)


def add_stats(a: TStatsOutput, b: TStatsOutput) -> TStatsOutput:
    if type(a) is not type(b):
        raise TypeError(f"Types {type(a)} and {type(b)} cannot be added.")

    sum_dict = deepcopy(a.dict())

    for k in sum_dict:
        if isinstance(sum_dict[k], list):
            sum_dict[k].extend(b.dict()[k])
        else:
            sum_dict[k] = np.concatenate((sum_dict[k], b.dict()[k]))

    return type(a)(**sum_dict)


def combine_stats(stats: Sequence[TStatsOutput]) -> tuple[TStatsOutput, list[int]]:
    output = None
    dataset_steps = []
    cur_len = 0
    for s in stats:
        output = s if output is None else add_stats(output, s)
        cur_len += len(s)
        dataset_steps.append(cur_len)
    if output is None:
        raise TypeError("Cannot combine empty sequence of stats.")
    return output, dataset_steps


def get_dataset_step_from_idx(idx: int, dataset_steps: list[int]) -> tuple[int, int]:
    last_step = 0
    for i, step in enumerate(dataset_steps):
        if idx < step:
            return i, idx - last_step
        last_step = step
    return -1, idx
