"""Attaching values that name their rows by key rather than by position.

Every other way into :meth:`~dataeval.Metadata.add_factors` hands over values already in a
level's row order. Per-track statistics do not: :func:`~dataeval.core.track_stats` indexes
its results by **sorted track id within one sequence**, while a metadata track row is keyed
``(item_index, track_index)`` with ``track_index`` dense in order of first appearance. The
two orders coincide only by accident, so attaching one to the other positionally is a
silent scramble, and building the mapping by hand is the ergonomic complaint the normalized
model exists to remove.

**This is a hash join, and here that is allowed.** The prohibition the store is built around
is on the *projection* path — reading a factor from a descendant level must be a positional
gather, because it happens on every read of every array-shaped accessor. This runs once, at
write time, over as many rows as the dataset has tracks, and what it produces is exactly the
positional array the projection path then uses. Paying for a join once to avoid one per read
is the trade the store is for, not a departure from it.
"""

__all__ = []

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from dataeval._metadata._links import gather_nulling
from dataeval.exceptions import ShapeMismatchError
from dataeval.types import FactorLevel

if TYPE_CHECKING:
    from dataeval._metadata._metadata import Metadata

# Consumed from the factor mapping rather than stored, because they say *which row* a value
# belongs to rather than anything about it.
_ITEM = "item_index"


def _key_values(factors: Mapping[str, Any], key: str) -> tuple[NDArray[Any], set[str]]:
    """Read the join column out of the factor mapping, and name what was consumed.

    The plural spelling is accepted because :func:`~dataeval.core.track_stats` returns its
    position map as ``track_ids`` while the column it matches is ``track_id``. Naming the
    *column* keeps the parameter meaning one thing — a column of the level's frame — rather
    than shifting to mean a key of the mapping depending on which was passed.
    """
    for name in (key, f"{key}s"):
        if name in factors:
            return np.asarray(factors[name]).reshape(-1), {name}
    raise ValueError(
        f"key={key!r} names the column to match on, so the factors must carry the values to "
        f"match with under {key!r} or {key + 's'!r}; got {sorted(factors)}. "
        f"track_stats returns them as 'track_ids'.",
    )


def _item_values(md: "Metadata", factors: Mapping[str, Any], rows: int, key: str) -> NDArray[np.intp]:
    """One source item per incoming row, supplied or inferred.

    ``track_stats`` describes a single sequence and says nothing about which, so a dataset
    holding exactly one item can supply the answer itself. A dataset holding several cannot:
    track ids restart per sequence, so a bare id names a row in every one of them.

    Which item a value belongs to has to be *said*, whether the caller attaches every
    sequence at once or one per call. Repeated calls fold into one column rather than
    colliding — see ``Metadata._merge_keyed`` — but each still has to name the item its
    keys are scoped to.
    """
    if _ITEM in factors:
        return np.asarray(factors[_ITEM], dtype=np.intp).reshape(-1)
    items = np.unique(md._store.frame(md._item_level)["item_index"].to_numpy())
    if len(items) == 1:
        return np.full(rows, items[0], dtype=np.intp)
    raise ValueError(
        f"key={key!r} matches on (item_index, {key}), and {key} restarts per item, so values "
        f"for a dataset with {len(items)} items have to say which item each belongs to. Add an "
        f"'item_index' entry to the factors — one entry per value, naming the item that value's "
        f"{key} is scoped to. track_stats describes one sequence at a time, so attaching a "
        "dataset's worth of them means saying which sequence each result came from, whether "
        "they go in one call or one call per sequence.",
    )


def resolve_keyed(
    md: "Metadata",
    factors: Mapping[str, Any],
    level: FactorLevel,
    key: str,
) -> tuple[list[tuple[str, FactorLevel, pl.Series]], NDArray[np.bool_]]:
    """Place each factor on the rows whose ``(item_index, key)`` its values name.

    A row the incoming values do not name is null rather than absent, so the column still
    has one entry per row at ``level`` and every downstream reader — binning, projection,
    the flat frame — sees the shape it expects.

    Which rows *were* named is returned alongside, because it is the difference between a
    write that leaves the rest of the column alone and one that blanks it. Every factor in
    a call is placed by the same keys, so one mask covers them all.

    Parameters
    ----------
    md : Metadata
        Instance being written to, already structured.
    factors : Mapping[str, Any]
        Values to attach. The key column and any ``item_index`` are consumed rather than
        stored: they say which row a value belongs to, not anything about it.
    level : str
        Level whose rows the values describe.
    key : str
        Column of that level's frame to match on.

    Returns
    -------
    list[tuple[str, str, pl.Series]]
        One entry per remaining factor, already in the level's row order.
    NDArray[np.bool_]
        One flag per row at ``level``, True where the incoming keys named it.

    Raises
    ------
    ValueError
        When ``key`` is not a column of the level's frame, when the factors carry no values
        for it, when the incoming keys are not unique, or when the dataset holds several
        items and the values do not say which they belong to.
    ShapeMismatchError
        When the factors disagree with the key column on how many rows they describe.
    """
    frame = md._store.frame(level)
    if key not in frame.columns:
        raise ValueError(
            f"key={key!r} is not a column of the {level!r} rows, which hold {frame.columns}. "
            "The key names the column to match on.",
        )

    keys, consumed = _key_values(factors, key)
    items = _item_values(md, factors, len(keys), key)
    consumed = consumed | ({_ITEM} if _ITEM in factors else set())
    if len(items) != len(keys):
        raise ShapeMismatchError(
            f"'item_index' has {len(items)} entries but the key column has {len(keys)}; they label "
            "the same rows and must be the same length.",
        )

    payload = {name: values for name, values in factors.items() if name not in consumed}
    if mismatched := {name: len(values) for name, values in payload.items() if len(values) != len(keys)}:
        raise ShapeMismatchError(
            f"Keyed factors must have one value per key: the key column has {len(keys)} entries, got {mismatched}.",
        )

    incoming = list(zip(items.tolist(), keys.tolist(), strict=True))
    if len(set(incoming)) != len(incoming):
        raise ValueError(
            f"The (item_index, {key}) pairs being added are not unique, so a row would have several "
            "values to choose between. Each key must name one row.",
        )

    # The join. A dict rather than a polars join because what is wanted is the *position*
    # of each match, which is what the rest of the store is addressed by; a frame join would
    # give the values back in a third order to be re-sorted.
    source = {pair: position for position, pair in enumerate(incoming)}
    wanted = zip(frame["item_index"].to_list(), frame[key].to_list(), strict=True)
    positions = np.fromiter((source.get(pair, -1) for pair in wanted), dtype=np.intp, count=frame.height)
    placed: list[tuple[str, FactorLevel, pl.Series]] = [
        (name, level, gather_nulling(name, values, positions)) for name, values in payload.items()
    ]
    return placed, positions >= 0
