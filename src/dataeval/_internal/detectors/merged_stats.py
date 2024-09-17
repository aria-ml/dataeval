from __future__ import annotations

from typing import Sequence, cast
from warnings import warn

import numpy as np

from dataeval._internal.metrics.stats import StatsOutput
from dataeval._internal.output import populate_defaults


def add_stats(a: StatsOutput, b: StatsOutput) -> StatsOutput:
    if not isinstance(a, StatsOutput) or not isinstance(b, StatsOutput):
        raise TypeError(f"Cannot add object of type {type(a)} and type {type(b)}.")

    a_dict = a.dict()
    b_dict = b.dict()
    a_keys = set(a_dict)
    b_keys = set(b_dict)

    missing_keys = a_keys - b_keys
    if missing_keys:
        raise ValueError(f"Required keys are missing: {missing_keys}.")

    extra_keys = b_keys - a_keys
    if extra_keys:
        warn(f"Extraneous keys will be dropped: {extra_keys}.")

    # perform add of multi-channel stats
    if "ch_idx_map" in a_dict:
        for k, v in a_dict.items():
            if k == "ch_idx_map":
                offset = sum([len(idxs) for idxs in v.values()])
                for ch_k, ch_v in b_dict[k].items():
                    if ch_k not in v:
                        v[ch_k] = []
                    a_dict[k][ch_k].extend([idx + offset for idx in ch_v])
            else:
                for ch_k in b_dict[k]:
                    if ch_k not in v:
                        v[ch_k] = b_dict[k][ch_k]
                    else:
                        v[ch_k] = np.concatenate((v[ch_k], b_dict[k][ch_k]), axis=1)
    else:
        for k in a_dict:
            if isinstance(a_dict[k], list):
                a_dict[k].extend(b_dict[k])
            else:
                a_dict[k] = np.concatenate((a_dict[k], b_dict[k]))

    return StatsOutput(**populate_defaults(a_dict, StatsOutput))


def combine_stats(stats) -> tuple[StatsOutput | None, list[int]]:
    dataset_steps = []

    if isinstance(stats, StatsOutput):
        return stats, dataset_steps

    output = None
    if isinstance(stats, Sequence) and isinstance(stats[0], StatsOutput):
        stats = cast(Sequence[StatsOutput], stats)
        cur_len = 0
        for s in stats:
            output = s if output is None else add_stats(output, s)
            cur_len += len(s)
            dataset_steps.append(cur_len)

    return output, dataset_steps


def get_dataset_step_from_idx(idx: int, dataset_steps: list[int]) -> tuple[int, int]:
    last_step = 0
    for i, step in enumerate(dataset_steps):
        if idx < step:
            return i, idx - last_step
        last_step = step
    return -1, idx
