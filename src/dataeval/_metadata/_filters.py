"""Evaluating a filter predicate, and reporting what the filter left behind.

The two halves of :meth:`~dataeval.Metadata.where` and :meth:`~dataeval.Metadata.having`
that are about the *query* rather than about the rows: deciding whether a predicate can be
answered at the level it was aimed at, and counting the rows a filter orphaned. The survivor
closure itself belongs to ``LevelStore``, which owns the
edges the closure walks.

Free functions rather than methods, because none of this is polymorphic and keeping it out
of the class keeps the class's surface to what a caller can reach.
"""

__all__ = []

import logging
from collections.abc import Mapping

import numpy as np
import polars as pl
from numpy.typing import NDArray

from dataeval._log import get_logger
from dataeval._metadata._store import LevelStore
from dataeval.types import FactorLevel

_logger = get_logger(__name__)


def _reject_unreadable_columns(store: LevelStore, level: FactorLevel, predicate: pl.Expr) -> list[str]:
    """Refuse a predicate naming a column that ``level``'s rows have no value for.

    This is the check that has to exist rather than be left to polars. :meth:`LevelStore.resolve`
    gives every column the store holds, filling the ones this level cannot reach with typed
    nulls — so a predicate over a track factor evaluated on unit rows does not fail, it
    answers null on every row, and the filter silently keeps nothing. A missing column would
    at least raise; a column that is *present and vacuous* is the dangerous shape, and it is
    exactly what asking a question at the wrong level produces.

    Returns
    -------
    list[str]
        The columns the predicate reads, once each — every one of them resolvable at
        ``level``, so the caller can read exactly these rather than the whole store.

    Raises
    ------
    ValueError
        When the predicate names a column no level supplies for these rows. The message
        names the level that does hold it, and points at ``having`` when that level is
        below this one — which is the whole reason the second filter exists.
    """
    names = list(dict.fromkeys(predicate.meta.root_names()))
    unreadable = [name for name in names if store.source_of(level, name) is None]
    if not unreadable:
        return names
    reasons: list[str] = []
    for name in unreadable:
        holder: FactorLevel | None = next((other for other in store.frames if name in store.frame(other).columns), None)
        if holder is None:
            reasons.append(f"{name!r} is not a column of this metadata")
        elif store.schema.is_ancestor(level, holder):
            reasons.append(
                f"{name!r} is defined at {holder!r}, which is below {level!r}: a {level!r} row has many "
                f"{holder!r} rows, not one. Use having(..., level={holder!r}) to keep the {level!r} rows "
                f"that have a matching {holder!r} row",
            )
        else:
            reasons.append(
                f"{name!r} is defined at {holder!r}, which is on a different branch of the level graph "
                f"from {level!r}, so these rows have no value for it. Filter at {holder!r} instead",
            )
    raise ValueError(f"Cannot evaluate this predicate on {level!r} rows: " + "; ".join(reasons) + ".")


def evaluate(store: LevelStore, level: FactorLevel, predicate: pl.Expr) -> NDArray[np.bool_]:
    """Answer ``predicate`` once per row at ``level``.

    Evaluated against the *resolved* frame, so a predicate may read any factor defined at
    this level or gathered down from one of its ancestors, spelled exactly as it is in
    :attr:`~dataeval.Metadata.dataframe`.

    A null answer is read as "does not match". Nulls survive the column check whenever a
    level's ancestry is partial — an untracked detection has no track factor to compare —
    and dropping those rows is what the predicate asked for, whereas keeping them would
    put a row in the result that was never shown to satisfy anything.

    Parameters
    ----------
    store : LevelStore
        Store whose rows the predicate is evaluated over.
    level : str
        Level supplying the rows.
    predicate : pl.Expr
        A polars expression answering one boolean per row.

    Returns
    -------
    NDArray[np.bool_]
        One flag per row at ``level``.

    Raises
    ------
    ValueError
        When the predicate names a column these rows have no value for, or does not answer
        one boolean per row.
    """
    names = _reject_unreadable_columns(store, level, predicate)
    # Only the columns the predicate reads. ``resolve`` would widen every column the store
    # holds — each ancestor factor gathered down, the rest typed nulls — to answer a
    # comparison against one of them, which is the cost ``select`` exists to avoid.
    given = store.select(level, names) if names else store.resolve(level)
    answered_frame = given.select(predicate.fill_null(value=False))
    if answered_frame.width != 1:
        raise ValueError(
            f"A filter predicate must answer one column, but this one answered "
            f"{answered_frame.width} ({list(answered_frame.columns)}). A multi-column selector "
            "like pl.col('a', 'b') asks several questions at once; combine them, e.g. "
            "pl.col('a').gt(0) & pl.col('b').gt(0).",
        )
    answered = answered_frame.to_series()
    if answered.len() != store.height(level):
        raise ValueError(
            f"A filter predicate must answer one value per row: this one answered {answered.len()} "
            f"time(s) for {store.height(level)} {level!r} row(s). An aggregate like pl.col(...).mean() "
            "collapses the rows it is given; compare it against something instead.",
        )
    if answered.dtype != pl.Boolean:
        raise ValueError(
            f"A filter predicate must answer a boolean per row, not {answered.dtype}. "
            "Compare the column against a value, e.g. pl.col('blur') > 0.5.",
        )
    return answered.to_numpy().astype(np.bool_, copy=False)


def _has_children(store: LevelStore, level: FactorLevel) -> NDArray[np.bool_]:
    """Which rows at ``level`` some row at a child level points at.

    Read through :meth:`LinkIndex.counts`, which is what already answers "how many children
    does each parent row have" and answers it without materializing one position per child —
    for a run-length edge it is a subtraction over the offsets.
    """
    reached = np.zeros(store.height(level), dtype=np.bool_)
    for child in store.schema:
        if level not in store.schema.parents_of(child):
            continue
        reached |= store.link(child, level).counts() > 0
    return reached


def _orphaned_by(
    before: LevelStore, after: LevelStore, keep: Mapping[FactorLevel, NDArray[np.intp]]
) -> dict[FactorLevel, int]:
    """Count the surviving rows a filter left with no children it used to have.

    ``where`` filters downwards and sideways but never upwards, so a filter at one level
    leaves its siblings whole: cutting frames keeps every track, including the tracks whose
    every observation was in a dropped frame. Those rows are correct — ``track`` is not a
    descendant of ``unit`` — and they are also not what most callers picture, so they are
    counted and reported rather than removed.

    Rows that were *already* childless are excluded. An empty frame is a legitimate shape a
    dataset can arrive in, and reporting it as a consequence of the filter would be wrong.

    Returns
    -------
    dict[str, int]
        Per level, how many surviving rows lost their last child. Levels that lost none are
        omitted, so an empty mapping means the filter orphaned nothing.
    """
    counts: dict[FactorLevel, int] = {}
    for level in after.frames:
        if not any(level in after.schema.parents_of(child) for child in after.schema):
            continue
        was = _has_children(before, level)[keep[level]]
        lost = int(np.count_nonzero(was & ~_has_children(after, level)))
        if lost:
            counts[level] = lost
    return counts


def report_orphaned_rows(
    before: LevelStore, after: LevelStore, keep: Mapping[FactorLevel, NDArray[np.intp]], level: FactorLevel
) -> None:
    """Note the rows a filter orphaned.

    Informational rather than a warning, following the same reasoning as
    ``log_items_without_targets`` in the structuring layer: the rows are correct
    and cost nothing, and a caller who wanted them gone can say so with a second filter.

    The counting is inside the level check rather than outside it, because it is several
    passes over every edge of both stores and its only consumer is this line: a caller who
    is not listening at ``INFO`` should pay nothing for a message nobody will read.
    """
    if not _logger.isEnabledFor(logging.INFO):
        return
    counts = _orphaned_by(before, after, keep)
    if not counts:
        return
    _logger.info(
        "Filtering at %r left %s with no remaining rows below them. Those levels are not below %r in "
        "the level graph, so this filter does not remove them; filter them directly, or with having(), "
        "if the analysis should exclude them.",
        level,
        ", ".join(f"{count} of {after.height(name)} {name!r} row(s)" for name, count in counts.items()),
        level,
    )
