"""Rolling a level's rows up into one value per row of a level above it.

The counterpart to the store's gather. Reading a coarse factor from a fine level replicates
it downwards; this collapses fine rows upwards, and the two are not symmetric — replication
is lossless, aggregation is a choice about what the fan-out means.

That asymmetry is the whole reason ``unique_by`` exists. A per-frame measurement read on
detection rows appears once per detection, so averaging it over a track weights each frame
by how many detections it happened to contain. That is almost never the intended question,
so an expression over an ancestor's column is refused unless the caller says how to
de-duplicate the fan-out first.

Grouping is positional throughout: a row's group *is* its parent's row position, so the
result scatters straight into an array as long as the parent level and no key is ever
hashed or joined.
"""

__all__ = []

from collections.abc import Sequence

import numpy as np
import polars as pl
from numpy.typing import NDArray

from dataeval._metadata._links import gather_nulling
from dataeval._metadata._store import LevelStore
from dataeval.types import FactorLevel

# Column names for the two keys added to the working frame. Leading underscores keep them
# out of the way of a factor named after them; they never leave this module.
_GROUP = "__agg_group"
_UNIQUE = "__agg_unique"


def validate(
    store: LevelStore,
    from_level: FactorLevel,
    to_level: FactorLevel,
    exprs: Sequence[pl.Expr],
    unique_by: FactorLevel | None,
) -> None:
    """Refuse an aggregation whose answer would be weighted by fan-out.

    Five checks, in the order a caller is likely to get them wrong:

    - ``to_level`` must sit strictly above ``from_level``. Aggregating into a sibling has
      no meaning — a frame and a track share detections but neither contains the other.
    - Both levels must have rows of their own. A level the schema declares but that
      produced no frame has nowhere for the result to land, and writing it would be a
      silent no-op rather than a factor.
    - ``unique_by`` must be ``from_level`` or an ancestor of it, since it names the thing
      being counted once. It need *not* be below ``to_level``: ``unique_by="unit"`` for an
      instance-to-track roll-up is the motivating case, and ``unit`` is ``track``'s sibling.
    - An expression must read only columns ``from_level``'s rows have a value for. This is
      the same hazard the filter layer's ``_reject_unreadable_columns`` guards against: the
      working frame gives every column the store holds, filling the unreachable ones with
      typed nulls, so an expression over a column defined *below* ``from_level`` or on a
      sibling branch does not fail — it answers null for every group.
    - An expression reading a column native to a strict ancestor of ``from_level`` needs
      ``unique_by``, because that column repeats across the fan-out.

    A count never trips the last two checks. :meth:`pl.Expr.meta.root_names` reports the
    columns an expression reads, and ``pl.len()`` reads none, so counting rows at
    ``from_level`` is always a question about ``from_level``.

    Raises
    ------
    ValueError
        When any of the five does not hold. The fan-out message names both ways out.
    """
    if not store.schema.is_ancestor(to_level, from_level):
        raise ValueError(
            f"agg rolls rows up into a level above them, but {to_level!r} does not sit above "
            f"{from_level!r} in this dataset's level graph. Levels above {from_level!r} are "
            f"{list(store.schema.ancestors(from_level))}.",
        )
    if absent := [level for level in (from_level, to_level) if level not in store.frames]:
        raise ValueError(
            f"agg needs rows at both levels, but {absent} have none in this dataset. Levels with "
            f"rows are {list(store.frames)}.",
        )
    if unique_by is not None and unique_by != from_level and not store.schema.is_ancestor(unique_by, from_level):
        raise ValueError(
            f"unique_by={unique_by!r} must be {from_level!r} itself or one of the levels above it "
            f"{list(store.schema.ancestors(from_level))}; it names the entity each row is counted "
            "once for.",
        )
    _reject_unreadable_columns(store, from_level, exprs)
    if unique_by is None:
        _reject_fanout(store, from_level, to_level, exprs)


def _read_names(exprs: Sequence[pl.Expr]) -> list[str]:
    """Every column the expressions read, once each, in first-seen order."""
    return list(dict.fromkeys(name for expr in exprs for name in expr.meta.root_names()))


def _reject_unreadable_columns(store: LevelStore, from_level: FactorLevel, exprs: Sequence[pl.Expr]) -> None:
    """Refuse an expression naming a column ``from_level``'s rows have no value for.

    The dangerous shape is a column that is *present and vacuous*: the working frame
    supplies every column the store holds, filling the ones these rows cannot reach with
    typed nulls, so an expression over a factor defined below ``from_level`` — or on a
    sibling branch of the level graph — aggregates nulls and answers null for every group
    instead of failing. A name no level holds at all is left alone, so that a typo still
    reaches polars and comes back as its own ``ColumnNotFound``, and so that a wildcard or
    regex selector, which names no column here, is not mistaken for one.

    Raises
    ------
    ValueError
        When an expression reads a column some level holds but ``from_level`` cannot.
    """
    held = set(store.columns)
    unreadable = [name for name in _read_names(exprs) if name in held and store.source_of(from_level, name) is None]
    if not unreadable:
        return
    holders = {
        name: next((other for other in store.frames if name in store.frame(other).columns), None) for name in unreadable
    }
    raise ValueError(
        f"Cannot aggregate {unreadable} at {from_level!r}: "
        + "; ".join(
            f"{name!r} is defined at {holders[name]!r}, which {from_level!r} rows have no value for — "
            f"aggregate from {holders[name]!r} instead, e.g. agg({holders[name]!r}, ...)"
            for name in unreadable
        )
        + ".",
    )


def _reject_fanout(store: LevelStore, from_level: FactorLevel, to_level: FactorLevel, exprs: Sequence[pl.Expr]) -> None:
    """Refuse an un-deduplicated expression over a column an ancestor defines.

    Such a column repeats once per ``from_level`` row beneath the entity that holds it, so
    aggregating it here weights each value by that fan-out. The message names both ways out.
    """
    inherited = sorted(
        {
            name
            for name in _read_names(exprs)
            if (source := store.source_of(from_level, name)) is not None and source != from_level
        },
    )
    if not inherited:
        return
    holders = {name: store.source_of(from_level, name) for name in inherited}
    raise ValueError(
        f"{inherited} are defined above {from_level!r} "
        f"({', '.join(f'{name} at {holders[name]!r}' for name in inherited)}), so each value repeats "
        f"once per {from_level!r} row beneath it and aggregating here would weight it by that "
        f"fan-out. Either aggregate at the level that defines it — agg({holders[inherited[0]]!r}, "
        f"{to_level!r}, ...) — or pass unique_by= to count each one once, e.g. "
        f"unique_by={holders[inherited[0]]!r}.",
    )


def _unique_keys(store: LevelStore, from_level: FactorLevel, unique_by: FactorLevel) -> NDArray[np.intp]:
    """One key per row at ``from_level``, equal exactly when two rows share a ``unique_by`` entity.

    A row with no ancestor at ``unique_by`` gets a key of its own rather than sharing the
    marker with every other such row: they are not the same entity, they are rows for which
    the question does not apply, and collapsing them together would silently drop all but
    one untracked detection.
    """
    positions = store.link(from_level, unique_by).positions() if unique_by != from_level else None
    if positions is None:
        return np.arange(store.height(from_level), dtype=np.intp)
    return np.where(positions >= 0, positions, -(np.arange(len(positions), dtype=np.intp) + np.intp(1)))


def _working_frame(
    store: LevelStore, from_level: FactorLevel, exprs: Sequence[pl.Expr], group: pl.Series
) -> pl.DataFrame:
    """Build the narrowest frame the expressions can be evaluated over, plus the group key.

    Only the columns the expressions actually read, resolved one at a time the way
    :meth:`LevelStore.select` does. Routing this through :meth:`LevelStore.resolve` instead
    would widen every column the store holds — gathering each ancestor factor down and
    filling the rest with typed nulls — to read the one or two an expression names, which is
    precisely the cost the normalized store keeps ``select`` separate from ``resolve`` to
    avoid.

    A name the store cannot resolve is a wildcard, a regex selector or a typo — never a
    vacuous column, which :func:`_reject_unreadable_columns` has already refused — and only
    the widened frame can answer the first two or raise on the last, so those fall back.
    """
    names = _read_names(exprs)
    if any(store.source_of(from_level, name) is None for name in names):
        return store.resolve(from_level).with_columns(group)
    return pl.DataFrame([*(store.column(from_level, name) for name in names), group])


def aggregate(
    store: LevelStore,
    from_level: FactorLevel,
    to_level: FactorLevel,
    exprs: Sequence[pl.Expr],
    unique_by: FactorLevel | None,
) -> list[pl.Series]:
    """Evaluate ``exprs`` per ``to_level`` row, over the ``from_level`` rows beneath it.

    Rows with no ancestor at ``to_level`` take no part: an untracked detection belongs to no
    track, so it is neither counted by nor averaged into one. A ``to_level`` row with no rows
    beneath it answers null rather than zero — nothing was measured there, which is a
    different statement from measuring zero.

    Returns
    -------
    list[pl.Series]
        One series per expression, each as long as ``to_level`` has rows and in its row
        order, named by the expression's output name.
    """
    groups = store.link(from_level, to_level).positions()
    frame = _working_frame(store, from_level, exprs, pl.Series(_GROUP, groups))
    if unique_by is not None:
        frame = frame.with_columns(pl.Series(_UNIQUE, _unique_keys(store, from_level, unique_by)))
    frame = frame.filter(pl.col(_GROUP) >= 0)
    if unique_by is not None:
        frame = frame.unique(subset=[_GROUP, _UNIQUE], keep="first", maintain_order=True)

    rolled = frame.group_by(_GROUP, maintain_order=True).agg(*exprs)
    # The scatter, without a scatter: each ``to_level`` row records which result row is its
    # own, and a level with no rows beneath it keeps the -1 that ``gather_nulling`` turns
    # into a null. ``Series.scatter`` is unimplemented for nested dtypes, and an expression
    # is free to return one.
    slot = np.full(store.height(to_level), -1, dtype=np.intp)
    slot[rolled[_GROUP].to_numpy()] = np.arange(rolled.height, dtype=np.intp)
    return [gather_nulling(name, rolled[name], slot) for name in rolled.columns if name != _GROUP]
