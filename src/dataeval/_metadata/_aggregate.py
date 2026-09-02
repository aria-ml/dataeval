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

import re
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
import polars as pl
from numpy.typing import NDArray

from dataeval._log import get_logger
from dataeval._metadata._links import gather_nulling
from dataeval._metadata._store import LevelStore
from dataeval.types import FactorLevel

_logger = get_logger(__name__)

# Column names for the two keys added to the working frame. Leading underscores keep them
# out of the way of a factor named after them; they never leave this module.
_GROUP = "__agg_group"
_UNIQUE = "__agg_unique"
_COVERAGE = "__agg_coverage"
_GAPS = "__agg_gaps"


class Rolled(NamedTuple):
    """One roll-up's columns, and how much of each level the answer covers.

    The counts travel with the columns rather than being logged and forgotten. How many
    rows took no part is what separates a roll-up straight to an ancestor from one routed
    through a partial branch, and the lowest coverage any destination saw is what tells a
    caller which ``min_coverage`` to ask for — neither is recoverable from the column
    afterwards, because a null does not say which of the three reasons produced it.

    Attributes
    ----------
    columns : list[pl.Series]
        One per expression, as long as the destination level and in its row order.
    took_part : int
        Source rows that had an ancestor at the destination and were summarized.
    no_ancestor : int
        Source rows excluded for having no ancestor there. Zero for every complete route.
    childless : int
        Destination rows with nothing beneath them, which answer the identity or null.
    coverage : tuple[float, ...]
        Per column, the lowest share of recorded values any destination with rows beneath
        it saw. 1.0 where every value was recorded, and where there was nothing to ask.
    uncovered : tuple[int, ...]
        Per column, destinations nulled for falling below the threshold.
    gaps : int
        Steps in the ordering key larger than the tightest step within the same destination,
        summed over all of them, for a roll-up that read its rows as an ordered series; 0
        for one that did not. It counts unevenness, whatever caused it — a filter that
        removed rows, a key-frame selection, or a source that genuinely sampled unevenly —
        because from the ordering key those are the same observation.
    """

    columns: list[pl.Series]
    took_part: int
    no_ancestor: int
    childless: int
    coverage: tuple[float, ...]
    uncovered: tuple[int, ...]
    gaps: int = 0


def validate(
    store: LevelStore,
    from_level: FactorLevel,
    to_level: FactorLevel,
    exprs: Sequence[pl.Expr],
    unique_by: FactorLevel | None,
    via: FactorLevel | None = None,
) -> None:
    """Refuse an aggregation whose answer would be weighted by fan-out.

    Six checks, in the order a caller is likely to get them wrong:

    - ``to_level`` must sit strictly above ``from_level``. Aggregating into a sibling has
      no meaning — a frame and a track share detections but neither contains the other.
    - ``via``, where given, must name a level some route to ``to_level`` steps through.
      Delegated to :meth:`LevelStore.link`, which owns the route vocabulary, and called
      here so a route that does not exist is refused alongside the other shape errors
      rather than midway through the roll-up.
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
        When any of the six does not hold. The fan-out message names both ways out.
    """
    _reject_bad_levels(store, from_level, to_level, via)
    if unique_by is not None and unique_by != from_level and not store.schema.is_ancestor(unique_by, from_level):
        raise ValueError(
            f"unique_by={unique_by!r} must be {from_level!r} itself or one of the levels above it "
            f"{list(store.schema.ancestors(from_level))}; it names the entity each row is counted "
            "once for.",
        )
    _reject_multi_output(store, exprs)
    _reject_unreadable_columns(store, from_level, exprs)
    if unique_by is None:
        _reject_fanout(store, from_level, to_level, exprs)


def _reject_bad_levels(
    store: LevelStore,
    from_level: FactorLevel,
    to_level: FactorLevel,
    via: FactorLevel | None,
) -> None:
    """Refuse a roll-up whose levels or route do not describe a direction through the graph.

    The three checks that are about the *shape* of the request rather than about what the
    expressions read, kept together so :func:`validate` reads as the four questions it asks
    rather than as six branches.

    Raises
    ------
    ValueError
        When ``to_level`` does not sit above ``from_level``, when no route between them
        passes through ``via``, or when either level has no rows.
    """
    if not store.schema.is_ancestor(to_level, from_level):
        raise ValueError(
            f"agg rolls rows up into a level above them, but {to_level!r} does not sit above "
            f"{from_level!r} in this dataset's level graph. Levels above {from_level!r} are "
            f"{list(store.schema.ancestors(from_level))}.",
        )
    if via is not None:
        # Raised by the store, which owns the route vocabulary; called for the raise alone,
        # and the composed link it builds is memoized for the roll-up that follows.
        store.link(from_level, to_level, via)
    if absent := [level for level in (from_level, to_level) if level not in store.frames]:
        raise ValueError(
            f"agg needs rows at both levels, but {absent} have none in this dataset. Levels with "
            f"rows are {list(store.frames)}.",
        )


def _read_names(exprs: Sequence[pl.Expr]) -> list[str]:
    """Every column the expressions read, once each, in first-seen order."""
    return list(dict.fromkeys(name for expr in exprs for name in expr.meta.root_names()))


def _names_one_output(expr: pl.Expr, columns: frozenset[str]) -> bool:
    """Whether an expression resolves to a single output column."""
    try:
        name = expr.meta.output_name()
    except pl.exceptions.ComputeError:
        return False
    if name in columns:
        return True
    # The pinned minimum polars (1.0.0) reports a selector's text as the output name --
    # pl.col('^time.*$').mean() comes back as '^time.*$' -- instead of raising like newer
    # versions do. A name that is no real column but matches real columns as a wildcard or
    # regex pattern is selector text naming several outputs, not one.
    try:
        return not [c for c in columns if re.fullmatch(name, c)]
    except re.error:
        return True  # not a valid pattern; a typo or alias that polars will answer for


def _reject_multi_output(store: LevelStore, exprs: Sequence[pl.Expr]) -> None:
    """Refuse an expression that names more than one output column.

    Every roll-up reads its results back one per expression, because that is what pairs a
    result with the coverage measured for it. A wildcard or regex selector names no single
    output, so ``output_name()`` raises on recent polars -- and the caller saw a raw polars
    ``ComputeError`` about root column names, mentioning neither ``agg`` nor the level
    graph. On the pinned minimum it comes back as the selector's text and the failure only
    surfaced as a ``ColumnNotFoundError`` inside the roll-up, so a name that is no real
    column is resolved against the columns this dataset holds to catch it there too.
    Refusing it here says which expression and what to write instead.

    Raises
    ------
    ValueError
        When an expression selects more than one column.
    """
    columns = frozenset(store.columns)
    if selectors := [expr for expr in exprs if not _names_one_output(expr, columns)]:
        raise ValueError(
            f"agg reads one result per expression, so each has to name one output column, but "
            f"{selectors} selects several. Name them one at a time -- pl.col('a').mean().alias('a_mean'), "
            f"pl.col('b').mean().alias('b_mean') -- so each result can be reported with the "
            f"coverage measured for it.",
        )


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


def _as_missing(frame: pl.DataFrame) -> pl.DataFrame:
    """Read NaN as the missing value this library spells it as, in every float column.

    DataEval writes an unrecorded number as ``NaN`` and not as a null — it is what
    ``Metadata._as_orderable`` carries ``NaT`` across to, what ``_holds_no_values`` tests
    for, and what the binning layer reserves ``BinSpec.missing_code`` for. Polars does not
    agree: to it a ``NaN`` is a *value*, so ``mean`` over a column holding one answers
    ``NaN`` and ``count`` counts it.

    Left alone, that makes one frame with no recorded timestamp poison the mean of the
    whole sequence it sits in, and makes a coverage threshold blind to exactly the rows it
    exists to notice. Converted here, at the one place a roll-up reads its values, so that
    the reductions and the coverage share one definition of *absent*.
    """
    floats = [name for name, dtype in frame.schema.items() if dtype in (pl.Float32, pl.Float64)]
    return frame.with_columns(pl.col(name).fill_nan(None) for name in floats) if floats else frame


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

    A name the store cannot resolve is a typo — never a vacuous column, which
    :func:`_reject_unreadable_columns` has already refused — and only the widened frame can
    raise on it, so it falls back. A wildcard or regex selector reaches here too and takes
    the same path, but it does not survive the roll-up: the result is read back by each
    expression's output name, which a multi-output selector does not have.
    """
    names = _read_names(exprs)
    if any(store.source_of(from_level, name) is None for name in names):
        return _as_missing(store.resolve(from_level)).with_columns(group)
    return _as_missing(pl.DataFrame([store.column(from_level, name) for name in names])).with_columns(group)


def _report_participation(
    store: LevelStore,
    from_level: FactorLevel,
    to_level: FactorLevel,
    via: FactorLevel | None,
    groups: NDArray[np.intp],
    filled: int,
) -> None:
    """Say how much of each level the answer actually covers.

    Not a diagnostic. The count of rows that took no part is what distinguishes a roll-up
    straight to an ancestor from one routed through a partial branch — ``instance`` to
    ``sequence`` counts every detection, the same roll-up ``via="track"`` counts only the
    ones a tracker linked — and the two are different questions with plausible answers
    either way. Zero for every total route, so it costs nothing to say and is the only
    signal that a caller wrote one and meant the other.

    Emitted at ``info`` and only when something did not participate; a full roll-up says
    nothing. This is the logging form of what the aggregation report will carry.
    """
    excluded = int(np.count_nonzero(groups < 0))
    childless = store.height(to_level) - filled
    if not excluded and not childless:
        return
    clauses = []
    if excluded:
        clauses.append(
            f"{excluded} of {groups.size} {from_level!r} row(s) took no part, having no {to_level!r} ancestor",
        )
    if childless:
        clauses.append(
            f"{childless} of {store.height(to_level)} {to_level!r} row(s) have nothing beneath them and answer null",
        )
    _logger.info(
        "agg %r -> %r%s: %s.",
        from_level,
        to_level,
        "" if via is None else f" via {via!r}",
        "; ".join(clauses),
    )


def aggregate(
    store: LevelStore,
    from_level: FactorLevel,
    to_level: FactorLevel,
    exprs: Sequence[pl.Expr],
    unique_by: FactorLevel | None,
    via: FactorLevel | None = None,
    empty: Any = None,
    min_coverage: float = 0.0,
    order: str | None = None,
) -> Rolled:
    """Evaluate ``exprs`` per ``to_level`` row, over the ``from_level`` rows beneath it.

    A ``NaN`` is read as a missing value rather than as a number, which is how the rest of
    the library spells one — see :func:`_as_missing`. So a reduction summarizes the values
    that were recorded, and ``min_coverage`` is what says how few of them is too few.

    Rows with no ancestor at ``to_level`` take no part: an untracked detection belongs to no
    track, so it is neither counted by nor averaged into one. A ``to_level`` row with no rows
    beneath it answers null rather than zero — nothing was measured there, which is a
    different statement from measuring zero.

    ``via`` selects a branch of the level graph rather than taking every route, which makes
    a roll-up over a partial branch expressible. It changes which rows take part, not how
    they are combined, so a roll-up and the same roll-up through a branch are different
    questions rather than different spellings — see :func:`_report_participation`.

    ``empty`` is the answer for a destination with no rows beneath it, and defaults to null
    because an arbitrary expression has no identity element to fall back on: nothing here
    can know that ``pl.len()`` of nothing is zero while ``pl.col(x).mean()`` of nothing is
    undefined. A *named* reduction does know, which is the one thing the named surface can
    say that this one cannot, and it says it by passing the value in.

    ``min_coverage`` nulls a destination whose rows beneath it did not all carry a value —
    the share that did, against the threshold. It defaults to 0, meaning summarize whatever
    is there, which is what this surface has always done. The all-or-nothing spelling is
    ``min_coverage=1.0``: the rule the structurers used to apply to a whole *factor*, at the
    granularity of one destination row, where it can be relaxed rather than only obeyed. A
    destination with nothing beneath it is not uncovered — it is empty, which ``empty``
    answers — so the two never fight over the same row.

    Returns
    -------
    Rolled
        One series per expression, each as long as ``to_level`` has rows and in its row
        order, named by the expression's output name, together with how much of each level
        the answer covers.
    """
    groups = store.link(from_level, to_level, via).positions()
    frame = _working_frame(store, from_level, exprs, pl.Series(_GROUP, groups))
    if unique_by is not None:
        frame = frame.with_columns(pl.Series(_UNIQUE, _unique_keys(store, from_level, unique_by)))
    frame = frame.filter(pl.col(_GROUP) >= 0)
    if unique_by is not None:
        frame = frame.unique(subset=[_GROUP, _UNIQUE], keep="first", maintain_order=True)

    measured = [*_coverage_exprs(exprs), *_gap_exprs(order)]
    rolled = frame.group_by(_GROUP, maintain_order=True).agg(*exprs, *measured)
    # The scatter, without a scatter: each ``to_level`` row records which result row is its
    # own, and a level with no rows beneath it keeps the -1 that ``gather_nulling`` turns
    # into a null. ``Series.scatter`` is unimplemented for nested dtypes, and an expression
    # is free to return one.
    slot = np.full(store.height(to_level), -1, dtype=np.intp)
    slot[rolled[_GROUP].to_numpy()] = np.arange(rolled.height, dtype=np.intp)
    _report_participation(store, from_level, to_level, via, groups, rolled.height)

    filled = rolled[_GROUP].to_numpy()
    columns: list[pl.Series] = []
    uncovered: list[int] = []
    for position, expr in enumerate(exprs):
        below = _below(rolled, position, min_coverage)
        covered = _covered(slot, filled, below)
        name = expr.meta.output_name()
        columns.append(_filled(gather_nulling(name, rolled[name], covered), slot, empty))
        uncovered.append(int(below.sum()))
    no_ancestor = int(np.count_nonzero(groups < 0))
    return Rolled(
        columns=columns,
        # Counted after the de-duplication rather than before it, since ``unique_by`` drops
        # rows precisely so they are *not* summarized twice, and a count that included them
        # would sit on a different denominator from the coverage measured in the same pass.
        took_part=frame.height,
        no_ancestor=no_ancestor,
        childless=store.height(to_level) - rolled.height,
        coverage=tuple(_lowest(rolled, position) for position in range(len(exprs))),
        uncovered=tuple(uncovered),
        gaps=int(rolled[_GAPS].sum()) if order is not None and rolled.height else 0,
    )


def _gap_exprs(order: str | None) -> list[pl.Expr]:
    """How many steps in the ordering key were larger than the tightest step, per group.

    Measured against the group's own smallest step rather than against a declared sampling
    rate, because nothing here knows one: a series is even when every step matches its
    shortest, and any longer step is a place where the reading either was not taken or was
    not kept. That makes irregular sampling read as gaps too, which is honest — a reduction
    that assumes evenly spaced observations is equally wrong in both cases.

    The smallest **positive** step, because a zero one is not a sampling interval. Two rows
    sharing a key are one observation recorded twice, not a series sampled infinitely fast,
    and measuring against their zero step reported every ordinary step in the group as a
    gap — which is the reverse of what the count is for.
    """
    if order is None:
        return []
    steps = pl.col(order).sort().diff()
    return [(steps > steps.filter(steps > 0).min()).sum().alias(_GAPS)]


def successive_differences(
    store: LevelStore,
    from_level: FactorLevel,
    to_level: FactorLevel,
    via: FactorLevel | None,
    column: str,
    order: str,
) -> NDArray[np.float64]:
    """Absolute change between consecutive values within each destination, pooled over all of them.

    What a *relative* tolerance is relative to. Pooled rather than measured per destination,
    so one number governs the whole roll-up and the runs it finds are comparable between the
    destinations it produces; a per-destination tolerance would make a noisy sequence and a
    clean one report the same run length for different amounts of movement.
    """
    groups = store.link(from_level, to_level, via).positions()
    # ``dict.fromkeys`` because a factor may *be* its own ordering: rolling ``time_s`` up
    # with a tolerance reads the same column as value and as key, and handing polars the
    # series twice raised ``column with name 'time_s' has more than one occurrence``.
    read = [store.column(from_level, name) for name in dict.fromkeys((column, order))]
    frame = _as_missing(pl.DataFrame(read))
    frame = frame.with_columns(pl.Series(_GROUP, groups)).filter(pl.col(_GROUP) >= 0)
    if not frame.height:
        return np.empty(0, dtype=np.float64)
    deltas = frame.select(pl.col(column).sort_by(pl.col(order)).diff().abs().over(_GROUP)).to_series()
    return deltas.drop_nulls().to_numpy().astype(np.float64)


def _lowest(rolled: pl.DataFrame, position: int) -> float:
    """Lowest coverage any destination with rows beneath it saw, for one expression.

    1.0 when there was no destination to ask. A roll-up that reached nothing is not badly
    covered — it has nothing to be covered *of* — and reporting 0.0 there would read as a
    complaint about the data rather than about reach, which ``no_ancestor`` already makes.
    """
    column = rolled[f"{_COVERAGE}{position}"]
    return 1.0 if not column.len() else float(column.min())  # type: ignore[arg-type]


def _coverage_exprs(exprs: Sequence[pl.Expr]) -> list[pl.Expr]:
    """One share-of-values-present expression per aggregating expression.

    Coverage is per *expression*, not per group: two expressions over two columns in one
    call have their own missing values and must be judged on them. An expression reading no
    column at all — a row count — has nothing that could be missing and is fully covered by
    construction; one reading several is covered only as far as its least present column,
    since it read them all.

    Computed whether or not a threshold will act on it, because the number is what tells a
    caller which threshold to ask for. A column that came back all nulls under the default
    is otherwise a result with no explanation attached to it.
    """
    built = []
    for position, expr in enumerate(exprs):
        read = expr.meta.root_names()
        present = pl.min_horizontal([pl.col(name).count() for name in read]) if read else pl.len()
        built.append((present / pl.len()).alias(f"{_COVERAGE}{position}"))
    return built


def _below(rolled: pl.DataFrame, position: int, min_coverage: float) -> NDArray[np.bool_]:
    """Which *result* rows fell short of the threshold, one per group the roll-up produced."""
    if min_coverage <= 0:
        return np.zeros(rolled.height, dtype=np.bool_)
    return np.asarray(rolled[f"{_COVERAGE}{position}"].to_numpy() < min_coverage)


def _covered(slot: NDArray[np.intp], filled: NDArray[np.intp], below: NDArray[np.bool_]) -> NDArray[np.intp]:
    """``slot`` with the under-covered destinations marked as having nothing to read.

    Marked in the *index* rather than nulled afterwards, so that a destination nulled for
    coverage and one nulled for having no rows beneath it stay distinguishable: the caller
    still holds the original ``slot``, and only the rows it calls empty get an identity
    element. An under-covered destination measured something and is not entitled to one.

    Written by scattering along ``filled`` — the destination each result row belongs to,
    which is the map ``slot`` was built from — rather than by reading ``slot`` back through
    a clamped index. The clamped form had to invent an index for every destination with no
    result at all, and when *no* destination had one it invented a read from an empty
    array: an ordinary outcome, since a roll-up routed through a branch that reaches
    nothing produces no groups, and it raised ``IndexError`` instead of answering null.
    """
    if not below.any():
        return slot
    marked = slot.copy()
    marked[filled[below]] = -1
    return marked


def _filled(series: pl.Series, slot: NDArray[np.intp], empty: Any) -> pl.Series:
    """Give the destinations with nothing beneath them the reduction's identity.

    Positional rather than :meth:`polars.Series.fill_null`, which cannot tell a group that
    measured nothing from a group whose values were themselves null — the first is what an
    identity answers for, and the second is a real measurement of missing data.
    """
    if empty is None:
        return series
    positions = np.flatnonzero(slot < 0)
    if not positions.size:
        return series
    if series.dtype == pl.Null or series.dtype.is_nested():
        # Two cases ``scatter`` cannot serve. A ``Null`` column has no dtype to widen, and
        # a nested one — a box, a list-valued factor — is unimplemented for scatter, which
        # is the same hazard ``gather_nulling`` exists to avoid. Built positionally so that
        # a destination the caller did not call empty keeps whatever it measured: filling
        # the whole column would hand the identity to rows that were nulled for coverage,
        # which is exactly the confusion positional filling is here to prevent.
        #
        # Read through ``to_list`` and rebuilt at the column's own dtype: *iterating* a
        # nested Series yields inner ``Series`` objects, and mixing those with a plain
        # Python ``empty`` gave polars a list of two kinds and it refused to build one —
        # so the branch written for the nested case was the one case it could not serve.
        # The ``Null`` column keeps its inferred dtype, since ``Null`` is what it has
        # instead of one.
        kept = zip(slot, series.to_list(), strict=True)
        values = [empty if position < 0 else value for position, value in kept]
        dtype = None if series.dtype == pl.Null else series.dtype
        return pl.Series(series.name, values, dtype=dtype)
    return series.scatter(positions, empty)
