from __future__ import annotations

from copy import deepcopy
from typing import Sequence, TypeVar

import numpy as np

from dataeval._internal.metrics.stats import BaseStatsOutput

TStatsOutput = TypeVar("TStatsOutput", bound=BaseStatsOutput)


def add_stats(a: TStatsOutput, b: TStatsOutput) -> TStatsOutput:
    if type(a) is not type(b):
        raise TypeError(f"Types {type(a)} and {type(b)} cannot be added.")

    stats_cls = type(a)
    a_dict = deepcopy(a.dict())
    b_dict = b.dict()

    for k in a_dict:
        if isinstance(a_dict[k], list):
            a_dict[k].extend(b_dict[k])
        else:
            a_dict[k] = np.concatenate((a_dict[k], b_dict[k]))

    return stats_cls(**a_dict)


def combine_stats(stats: Sequence[TStatsOutput]) -> tuple[TStatsOutput, list[int]]:
    output = None
    dataset_steps = []
    cur_len = 0
    stat_type = type(stats[0]) if stats else None
    for s in stats:
        if stat_type is None or not isinstance(s, BaseStatsOutput) or not isinstance(s, stat_type):
            raise TypeError("Cannot combine outputs.")
        output = s if output is None else add_stats(output, s)
        cur_len += len(s)
        dataset_steps.append(cur_len)
    if output is None:
        raise TypeError("Cannot combine outputs.")
    return output, dataset_steps


def get_dataset_step_from_idx(idx: int, dataset_steps: list[int]) -> tuple[int, int]:
    last_step = 0
    for i, step in enumerate(dataset_steps):
        if idx < step:
            return i, idx - last_step
        last_step = step
    return -1, idx
