from __future__ import annotations

__all__ = []

import re
from collections import ChainMap
from collections.abc import Sequence
from copy import deepcopy
from typing import Any, TypeVar

import numpy as np

from dataeval.core._calculate import CalculationResult
from dataeval.outputs._stats import BaseStatsOutput
from dataeval.protocols import Array

DTYPE_REGEX = re.compile(r"NDArray\[np\.(.*?)\]")

TStatsOutput = TypeVar("TStatsOutput", bound=BaseStatsOutput, covariant=True)


def convert_output(
    output_cls: type[TStatsOutput],
    output_dict: CalculationResult,
) -> TStatsOutput:
    output: dict[str, Any] = {}
    attrs = dict(ChainMap(*(getattr(c, "__annotations__", {}) for c in output_cls.__mro__)))

    # Merge stats field into the main dict for processing
    stats_dict = output_dict["stats"]
    combined_dict = {**output_dict, **stats_dict}

    for key in (key for key in combined_dict if key in attrs):
        stat_type: str = attrs[key]
        dtype_match = re.match(DTYPE_REGEX, stat_type)
        if dtype_match is not None:
            output[key] = np.asarray(combined_dict[key], dtype=np.dtype(dtype_match.group(1)))
        else:
            output[key] = combined_dict[key]
    return output_cls(**output)


def add_stats(a: TStatsOutput, b: TStatsOutput) -> TStatsOutput:
    if type(a) is not type(b):
        raise TypeError(f"Types {type(a)} and {type(b)} cannot be added.")

    sum_dict = deepcopy(a.data())

    for k in sum_dict:
        if isinstance(sum_dict[k], Sequence):
            sum_dict[k].extend(b.data()[k])
        elif isinstance(sum_dict[k], Array):
            sum_dict[k] = np.concatenate((sum_dict[k], b.data()[k]))
        else:
            sum_dict[k] += b.data()[k]

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
