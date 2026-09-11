"""How many independent observations stand behind a contingency table.

A statistic read off a contingency table is corrected against the number of draws the
table was built from, and that is not always its row count. A factor defined above the
rows being read arrives replicated once per descendant — a per-sequence factor on
detection rows takes one value per sequence and arrives once per detection — so the same
values are counted many times over with no new draw behind them.

Shared by ``dataeval.core._mutual_info`` and ``dataeval.core._parity``, which correct
different statistics against the same count and would otherwise each carry their own copy
of the reasoning.
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
    """Check a caller's entity counts against the table they describe.

    A wrong length is refused rather than trimmed or padded, because it means the caller's
    column list and this call's are not the same list, and either repair would score some
    factor against another factor's entity count — a silent answer to a question nobody
    asked. Counts above the row count are clamped instead: a column cannot vary over more
    entities than there are rows carrying it, and a caller reading a filtered view can
    arrive here with a stale total without having said anything false about the level.
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
    """Entities behind a pair of columns: the larger of the two counts.

    The finer factor governs. Where one nests inside the other — a per-sequence factor
    against a per-detection one — the pair takes a distinct value per *detection*, so the
    detections are the draws and the sequences are not a ceiling on them. Taking the
    smaller would correct a pair that is already right, and over-correct it by about as
    much as leaving the replicated pair uncorrected gets it wrong.

    Two factors on incomparable branches — a per-frame factor against a per-track one, which
    meet only on detection rows — have no nesting between them, and the larger of the two
    counts understates the distinct combinations their rows carry. That direction leaves the
    correction too large rather than too small, which is the safe way to be wrong.
    """
    return None if effective_n is None else int(max(effective_n[first], effective_n[second]))


def rescaled(table: Any, n_effective: int | None) -> Any:
    """Scale a table so its total is ``n_effective``, where that is fewer than it holds.

    Rescaling is what makes an effective ``n`` bind, and forgetting it is the failure mode
    worth naming. Both statistics corrected here read a table's **margins** as well as its
    ``n``: the G-test takes its whole statistic from the counts, and
    ``expected_mutual_information`` sums over ranges the margins set. Handing either a
    smaller ``n`` beside margins that still sum to the row count does not correct harder —
    it answers a question about a table nobody built, and moves the result in the same
    direction the replication did. The correction then looks applied and is not.

    Scaling every cell by one factor leaves the table's proportions alone, so what the
    statistic reads as *association* does not move; only what it reads as *evidence* does,
    which is the whole intent. A total already at or below ``n_effective`` is returned
    untouched, which is the ordinary single-level case and every caller passing None.
    """
    if n_effective is None:
        return table
    total = float(table.sum())
    if total <= 0.0 or n_effective >= total:
        return table
    return table * (n_effective / total)
