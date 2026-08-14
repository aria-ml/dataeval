"""Reading one level's values from another level's rows.

Propagation is positional: every row block records, for its own level and each ancestor
level, the position of the row it inherits from, so a factor defined above a row is a
gather rather than a join. This module holds those gathers in the two forms the pipeline
needs — a Python list for the flat row builder, and a :class:`polars.Series` for the
frame builder — together with the two questions the frame builder asks about a column
before handing it to polars.

A **negative position** means the row has no ancestor at that level and reads as None.
That is not a defensive nicety: it is the only representation of partial ancestry the
layout has, and the diamond in the level graph makes partial ancestry real.
"""

__all__ = []

from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from dataeval._metadata._links import gather_nulling


def take(values: Any, positions: NDArray[np.intp]) -> list[Any]:
    """Gather ``values`` at ``positions``, tolerating arrays, lists and other sequences.

    A **negative position** means the row has no ancestor at that level and yields None.
    That is not a defensive nicety: it is the only representation of partial ancestry the
    layout has, and the diamond in the level graph makes partial ancestry real — a
    detection no tracker linked has a frame but no track, so a per-track factor has no
    value for it. Gathering such a row naively would index from the end of the array and
    silently attribute another track's value to it.
    """
    missing = positions < 0
    if missing.all():
        return [None] * len(positions)

    # Clamped rather than filtered so the gather stays one vectorized operation; every
    # clamped slot is overwritten with None below.
    safe = np.where(missing, 0, positions)
    if isinstance(values, np.ndarray):
        gathered = values[safe].tolist()
    else:
        sequence = values if isinstance(values, (list, tuple)) else list(values)
        gathered = [sequence[position] for position in safe]

    for index in np.flatnonzero(missing):
        gathered[index] = None
    return gathered


def holds_only_nulls(values: Sequence[Any] | NDArray[Any]) -> bool:
    """Whether a column is entirely null, and so contributes no type to its frame.

    A block emits every legacy reserved column whether or not it has values for one, so
    a ``unit`` block carries ``class_label`` as nothing but nulls. Such a column has
    polars dtype ``Null``, and ``Null`` does **not** supertype against a real dtype
    during a diagonal concat — it raises ``SchemaError``. Omitting the column instead
    lets the concat fill it from the blocks that do type it, which is the same result
    the flat builder produced by inference.

    Short-circuits on the first non-null, so a populated column costs one comparison.
    """
    return not isinstance(values, np.ndarray) and all(value is None for value in values)


def column_values(values: Sequence[Any] | NDArray[Any] | None, size: int) -> list[Any]:
    """One block's values for a column as a plain list, null-filled when it has none.

    The conversion the structurers no longer pay (see ``_reserved._as_column``) lands
    here instead, on the flat row form that genuinely needs Python scalars.
    :meth:`StructuredData.to_frame` skips it entirely.
    """
    if values is None:
        return [None] * size
    return values.tolist() if isinstance(values, np.ndarray) else list(values)


def gather_series(name: str, values: Any, positions: NDArray[np.intp]) -> pl.Series:
    """Gather ``values`` at ``positions`` as a Series, nulling the negative positions.

    The vectorized form of :func:`take`: it never builds a Python list, so an array
    factor is gathered by polars directly. A **negative position** means the row has no
    ancestor at that level and yields null — see :func:`take` for why that case is
    load-bearing rather than defensive.

    Delegated to :func:`gather_nulling` rather than restated, so this and
    :meth:`GatherLink.broadcast`, which is the same gather reached through a link, cannot
    drift about what a negative position means; that function documents why the null goes
    in the index rather than in the gathered column.
    """
    return gather_nulling(name, values, positions)
