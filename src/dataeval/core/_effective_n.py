"""Helpers for computing and applying effective sample size to contingency tables.

When metadata factors are defined at coarser hierarchy levels than the rows being
evaluated (e.g. per-sequence factors evaluated on per-detection rows), repeated
observations artificially inflate sample size. These functions validate, combine,
and rescale contingency tables to reflect independent entity counts.
"""

__all__ = []

from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval.types import Array1D
from dataeval.utils._array import as_numpy


def validate_effective_n(
    effective_n: Array1D[int] | None,
    num_columns: int,
    num_rows: int,
) -> NDArray[np.intp] | None:
    """Check effective sample size counts against the contingency table dimensions.

    Ensures ``effective_n`` matches ``num_columns`` and contains positive values.
    Counts exceeding ``num_rows`` are clamped to ``num_rows``.
    """
    if effective_n is None:
        return None
    counts = as_numpy(effective_n, dtype=np.intp, required_ndim=1)
    if counts.shape[0] != num_columns:
        raise ValueError(
            f"effective_n has {counts.shape[0]} entries for {num_columns} columns. It is indexed "
            "the way the statistic's own columns are — the conditioning axis at 0 and factor i at "
            f"i+1 — so it needs one entry per factor plus one, {num_columns} in all.",
        )
    if np.any(counts < 1):
        raise ValueError(
            f"effective_n holds {int(counts.min())}, and a column stands over at least one entity. "
            "Zero usually means a level was counted on rows that cannot reach it.",
        )
    return np.minimum(counts, num_rows)


def pair_n(effective_n: NDArray[np.intp] | None, first: int, second: int) -> int | None:
    """Return the effective sample size for a pair of columns.

    Selects the maximum of the two column entity counts, representing the finer
    level of granularity between them.
    """
    return None if effective_n is None else int(max(effective_n[first], effective_n[second]))


def rescaled(table: Any, n_effective: int | None) -> Any:
    """Scale a table so its total is ``n_effective``, where that is fewer than it holds.

    Preserves cell proportions while adjusting marginal totals so that sample-size
    dependent statistics reflect ``n_effective``. If ``n_effective`` is None or
    greater than or equal to the table sum, the table is returned unchanged.
    """
    if n_effective is None:
        return table
    total = float(table.sum())
    if total <= 0.0 or n_effective >= total:
        return table
    return table * (n_effective / total)
