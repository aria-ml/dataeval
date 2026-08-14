"""Numbering rows within their parent, which is where a level's own key comes from.

``instance_index``, ``target_index`` and ``track_index`` are all the same question asked
of different parents: how far into its parent's run does this row sit. The two functions
here differ only in what they may assume about the order the rows arrive in, and that
distinction is load-bearing — a structurer builds its own rows and can promise they are
grouped, while a caller handing over ``item_indices`` cannot.
"""

__all__ = []

import numpy as np
from numpy.typing import NDArray


def running_index(parents: NDArray[np.intp]) -> NDArray[np.intp]:
    """Index each row within its parent group, assuming rows are grouped by parent.

    Parameters
    ----------
    parents : NDArray[np.intp]
        Parent position of each row, in row order and grouped by parent.

    Returns
    -------
    NDArray[np.intp]
        0, 1, 2, ... restarting at each new parent.
    """
    count = len(parents)
    if count == 0:
        return np.empty(0, dtype=np.intp)
    starts = np.concatenate(([0], np.flatnonzero(parents[1:] != parents[:-1]) + 1))
    group_sizes = np.diff(np.append(starts, count))
    return np.arange(count, dtype=np.intp) - np.repeat(starts, group_sizes)


def index_within_parent(parents: NDArray[np.intp]) -> NDArray[np.intp]:
    """Index each row within its parent group, whatever order the rows arrive in.

    :func:`running_index` requires its rows already grouped, which every dataset
    structurer can promise because it builds them itself. Caller-supplied
    ``item_indices`` cannot: ``[0, 1, 0, 1]`` is an ordinary way to write two items of two
    rows each, and read as grouped it numbers every row 0 — leaving two pairs of rows
    sharing an ``(item_index, target_index)`` identity, which is what a source index is
    matched on later.

    Parameters
    ----------
    parents : NDArray[np.intp]
        Parent of each row, in row order and in any grouping.

    Returns
    -------
    NDArray[np.intp]
        0, 1, 2, ... within each parent, in the rows' own order.
    """
    order = np.argsort(parents, kind="stable")
    within = np.empty(len(parents), dtype=np.intp)
    within[order] = running_index(parents[order])
    return within
