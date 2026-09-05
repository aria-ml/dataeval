"""Named reductions: the contract a roll-up's name carries that an expression cannot.

``pl.col("brightness").mean()`` says what to compute and nothing else. It does not say that
a mean is meaningful over brightness and meaningless over a class label, that a destination
with nothing beneath it has no mean but does have a count of zero, or that the result should
be called ``brightness_mean`` in every report that prints it. Those are properties of the
*name*, and this module is where the names keep them.

Three things each entry declares, and each one refuses something an expression would let
through:

- **A value type.** ``mean`` over a class label is a category error that reaches polars and
  comes back as a number. Declared here, it is refused before evaluation, by name.
- **An identity element.** A frame with no detections has genuinely zero of them; a frame
  with no detections has no *mean* of anything. Blanket null loses the first, blanket zero
  invents the second, and only the reduction knows which it is.
- **A kind.** A positional reduction is invariant to the order of the rows it consumes; a
  temporal one is a function of the ordered series and needs an ordering to exist -- so it
  refuses to run at a level that has none, rather than quietly reading row order as time.

Counting *rows* is deliberately absent. Every reduction here is a question about one
factor's values, and ``count`` is the count of the values it has -- which for a factor with
no missing values is the row count anyway. The row count as such is
``agg(source, target, pl.len().alias(name))``, which needs no factor to be about.
"""

__all__ = []

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import polars as pl

from dataeval._log import get_logger
from dataeval.types import REDUCTION_NAMES, Aggregator, FactorLevel, FactorLevelSchema

_logger = get_logger(__name__)

# Which value types a reduction applies to.
#
# ``orderable`` is wider than ``numeric`` on purpose: the smallest and largest capture time
# in a sequence are as meaningful as the smallest and largest brightness, and a capture time
# is not a number. ``any`` is everything with equality, which excludes only the nested
# dtypes -- a box column has no mode and no distinct count worth the name.
Domain = Literal["numeric", "orderable", "boolean", "any"]


@dataclass(frozen=True)
class Reduction:
    """One named reduction and what its name promises.

    Attributes
    ----------
    expr : Callable[[str], pl.Expr]
        Builds the aggregating expression over one column.
    domain : {"numeric", "orderable", "boolean", "any"}
        Value types the reduction applies to.
    identity : Any or None, default None
        What a group with no rows answers. None means *undefined* -- the destination row is
        null, because nothing was measured there. No reduction's identity is None itself,
        so the two are never confused.
    kind : {"positional", "temporal"}, default "positional"
        Whether the answer depends on the order of the rows consumed. A temporal reduction
        is built with the ordering column as well as the value column.
    gap_sensitive : bool, default False
        Whether an uneven ordering key distorts the answer. ``variability`` and ``trend``
        divide by the key delta, so a missing observation changes nothing they report;
        ``changes`` and ``longest_run`` count positions and cannot tell a run that ended
        from a run whose middle is absent.
    options : tuple of str, default ()
        Names of the reduction-specific parameters it accepts. An option a reduction does
        not take is refused at construction rather than accepted and ignored, which is the
        whole reason they are declared rather than passed through.
    coverage_sensitive : bool, default True
        Whether missing values beneath a destination distort the answer. False for the
        reductions that are *about* missingness rather than damaged by it: counting the
        values present, and counting the distinct ones, are the right answers however many
        were absent, so a coverage threshold has nothing to protect there and would only
        null the one question that survives incomplete data.
    """

    expr: Callable[..., pl.Expr]
    domain: Domain
    identity: Any = None
    kind: Literal["positional", "temporal"] = "positional"
    coverage_sensitive: bool = True
    gap_sensitive: bool = False
    options: tuple[str, ...] = ()

    def build(self, column: str, order: str | None, options: Mapping[str, Any]) -> pl.Expr:
        """Build the aggregating expression for one column, ordered where the reduction needs it.

        A positional reduction never sees ``order``: it is invariant to the order of what it
        consumes, so handing it one would suggest otherwise.
        """
        if self.kind == "positional":
            return self.expr(column)
        if order is None:
            raise ValueError(
                f"{self.expr.__name__} reads its rows as an ordered series and was given no ordering "
                "column. Resolution supplies one or refuses, so reaching here means it was bypassed.",
            )
        return self.expr(column, order, **dict(options))


def _positioned(order: str) -> pl.Expr:
    """Read the ordering key, restricted to the rows that recorded one."""
    return pl.col(order).filter(pl.col(order).is_not_null())


def _ordered(column: str, order: str) -> pl.Expr:
    """Read a group's values in the order its ordering column puts them in.

    A row whose ordering key is missing **leaves the series**. It has no position, and
    both ends of the series are a fabrication: polars sorts nulls first, so an untimed
    frame was read as the *earliest* observation and its value joined the front. A
    four-frame sequence whose per-frame counts run 1, 2, 2, 1 reported one change instead
    of two the moment its last frame lost its timestamp -- the reading was not wrong about
    the data, it was answering about a series the data never described.

    Reporting over the readings that do have a position is the same answer
    :attr:`Reduction.gap_sensitive` already gives for a reading that was never taken: the
    reduction cannot place it, so it does not pretend to.
    """
    return pl.col(column).filter(pl.col(order).is_not_null()).sort_by(_positioned(order))


def _variability(column: str, order: str) -> pl.Expr:
    """Mean absolute change per unit of the ordering key.

    A *rate*, not a raw mean of successive differences, and that is what makes it right
    across a gap: two observations a second apart differing by 2 have moved as fast as two a
    tenth of a second apart differing by 0.2, and a series with a frame missing from the
    middle is not twice as jittery for it. On an evenly sampled series the key delta is
    constant, so this is the raw mean of the absolute differences divided by that constant.

    Distinct from ``var``, which is order-invariant and therefore positional. A slow
    illumination drift and a strobing feed can have identical variance; their mean absolute
    rate of change differs by an order of magnitude, and that difference is the whole reason
    this reduction exists.

    A step of **zero** in the ordering key contributes nothing rather than infinity. Two
    rows sharing a key -- two detections in one frame, two frames stamped with the same
    coarse clock -- have no time between them, so there is no rate to read off the pair;
    dividing by the zero step instead answered ``inf``, and one such pair made the whole
    destination's answer ``inf`` however many real pairs it had.
    """
    steps = _positioned(order).sort().diff()
    return (_ordered(column, order).diff() / pl.when(steps != 0).then(steps)).abs().mean()


def _trend(column: str, order: str) -> pl.Expr:
    """Least-squares slope of the values against the ordering key, in units per key unit.

    A regression needs no sorting -- it reads pairs, not a sequence -- so this is written
    without one. It is still temporal: without an ordering key there is nothing to regress
    against, and row position is not one.

    Both halves read the **same** rows. ``cov`` sees only the rows where value and key are
    both recorded, so the variance beneath it has to be the variance of those keys and not
    of every key the group holds: measured over all of them, one row whose value is missing
    still widened the denominator, and a single unrecorded reading at a distant key
    flattened the slope by orders of magnitude.
    """
    paired = pl.col(order).filter(pl.col(column).is_not_null() & pl.col(order).is_not_null())
    return pl.cov(pl.col(order), pl.col(column)) / paired.var()


def _changes(column: str, order: str) -> pl.Expr:
    """Transitions between consecutive distinct values, among the values that were recorded.

    A missing value neither counts as a change nor conceals one. Comparing *against* it
    answers null, which the sum skips -- so leaving it in place made both comparisons that
    touch it disappear, and a series reading 1, 1, absent, 2, 2 reported no change at all.
    Dropping it first joins the two recorded stretches, which is the same thing that happens
    to a reading the ordering key never carried: the reduction cannot tell an absent value
    from an absent row, and says so through ``gap_sensitive``.
    """
    ordered = _ordered(column, order).drop_nulls()
    return (ordered != ordered.shift()).sum()


def _longest_run(column: str, order: str, tolerance: float | None = None) -> pl.Expr:
    """Longest consecutive stretch of one value, or of values within ``tolerance`` of the last.

    Written as a run *id* per position -- a running count of the breaks -- rather than as a
    bespoke scan, so the exact and the tolerant forms are one expression with one definition
    of where a run ends.

    Missing values are dropped before the scan, for the reason :func:`_changes` gives: the
    only null the ``fill_null(True)`` below should read as a break is the first position's,
    which has nothing before it. Left in, an unrecorded reading broke a run that ``changes``
    read straight through, and the two reductions disagreed about what an absence means.
    """
    ordered = _ordered(column, order).drop_nulls()
    breaks = (ordered != ordered.shift()) if tolerance is None else (ordered.diff().abs() > tolerance)
    return breaks.fill_null(True).cum_sum().rle().struct.field("len").max()


# Every reduction that can see a missing value drops them first. Polars treats a null as an
# ordinary value in ``mode`` and ``n_unique``, which is the opposite of what this library
# means by one: without the drop, ``mode`` answers *missing* whenever absence ties for most
# common — and ``.sort()``, which is here so that a tie between two real values resolves the
# same way on every run, sorts nulls first and so hands them the tie. ``n_unique`` would
# likewise count "absent" as one of the distinct values, inflating every group holding one
# by exactly 1, and its ``coverage_sensitive=False`` means no threshold could ever mask it.
# ``changes`` and ``longest_run`` drop them for a third reason: they compare each value
# against the one before it, and a comparison touching a null answers null -- which made the
# first hide a transition that spanned an absence and the second call one a break, so the
# two disagreed about the same missing reading. ``count``, ``sum``, ``mean`` and the rest
# already skip nulls in polars.
REDUCTIONS: Mapping[str, Reduction] = {
    "count": Reduction(lambda c: pl.col(c).count(), "any", identity=0, coverage_sensitive=False),
    "n_unique": Reduction(lambda c: pl.col(c).drop_nulls().n_unique(), "any", identity=0, coverage_sensitive=False),
    "sum": Reduction(lambda c: pl.col(c).sum(), "numeric", identity=0),
    "mean": Reduction(lambda c: pl.col(c).mean(), "numeric"),
    "median": Reduction(lambda c: pl.col(c).median(), "numeric"),
    "std": Reduction(lambda c: pl.col(c).std(), "numeric"),
    "var": Reduction(lambda c: pl.col(c).var(), "numeric"),
    "min": Reduction(lambda c: pl.col(c).min(), "orderable"),
    "max": Reduction(lambda c: pl.col(c).max(), "orderable"),
    "mode": Reduction(lambda c: pl.col(c).drop_nulls().mode().sort().first(), "any"),
    "first": Reduction(lambda c: pl.col(c).first(), "any"),
    "last": Reduction(lambda c: pl.col(c).last(), "any"),
    "any": Reduction(lambda c: pl.col(c).any(), "boolean", identity=False),
    "all": Reduction(lambda c: pl.col(c).all(), "boolean", identity=True),
    "variability": Reduction(_variability, "numeric", kind="temporal"),
    "trend": Reduction(_trend, "numeric", kind="temporal"),
    "changes": Reduction(_changes, "any", identity=0, kind="temporal", gap_sensitive=True),
    "longest_run": Reduction(_longest_run, "any", kind="temporal", gap_sensitive=True, options=("tolerance",)),
}

# Columns that carry an order, in the order they are preferred. A wall-clock time beats a
# presentation timestamp beats a position, because the first two survive a series being
# resampled and the third does not. Only a column a level holds *itself* is eligible: an
# ordering read down from an ancestor repeats across the fan-out, which the roll-up would
# then have to be told how to de-duplicate before it could sort by it.
ORDERINGS: tuple[str, ...] = ("time_s", "pts", "unit_index")


# The vocabulary is declared in `dataeval.types`, beside the field that takes it, because
# that module cannot import this one. This is the other half of that split: what a caller
# enumerates and what this registry implements are the same set, proved once at import
# rather than left to a test somebody remembers to run.
if set(REDUCTIONS) != set(REDUCTION_NAMES):
    raise RuntimeError(
        "The reduction registry and `dataeval.types.REDUCTION_NAMES` disagree: "
        f"registry-only {sorted(set(REDUCTIONS) - set(REDUCTION_NAMES))}, "
        f"names-only {sorted(set(REDUCTION_NAMES) - set(REDUCTIONS))}. "
        "A name a caller can enumerate but not use, or use but not enumerate, is worse "
        "than neither -- add it to both.",
    )


def lookup(how: str) -> Reduction:
    """Find a reduction by name.

    Raises
    ------
    ValueError
        When no reduction has that name. The message lists every one that does, since the
        vocabulary is small enough to read and a near-miss is the likeliest mistake.
    """
    if (reduction := REDUCTIONS.get(how)) is not None:
        return reduction
    raise ValueError(f"{how!r} is not a reduction. The reductions are {sorted(REDUCTIONS)}.")


def admits(reduction: Reduction, dtype: pl.DataType) -> bool:
    """Whether a reduction applies to values of this type."""
    if isinstance(dtype, pl.List | pl.Array | pl.Struct | pl.Null):
        return False
    if reduction.domain == "any":
        return True
    if reduction.domain == "boolean":
        return dtype == pl.Boolean
    if reduction.domain == "numeric":
        return dtype.is_numeric()
    return dtype.is_numeric() or dtype.is_temporal() or dtype in (pl.String, pl.Boolean)


def expressions(aggregator: Aggregator) -> list[pl.Expr]:
    """One aliased expression per factor of a resolved aggregator, in its factor order."""
    reduction = lookup(aggregator.how)
    return [
        reduction.build(factor, aggregator.order_by, aggregator.options).alias(aggregator.name_for(factor))
        for factor in aggregator.factors
    ]


def is_gap_sensitive(how: str) -> bool:
    """Whether an uneven ordering key distorts what a reduction of this name answers.

    Asked by name rather than of an aggregator, because the answer is the reduction's and a
    caller holding only a record of what ran should not have to build a declaration around
    the name to reach it.
    """
    return lookup(how).gap_sensitive


def tolerance_of(aggregator: Aggregator) -> Any:
    """Return the tolerance this roll-up was asked for, or None where it names none."""
    return aggregator.options.get("tolerance")


def with_tolerance(aggregator: Aggregator, factor: str, tolerance: float) -> Aggregator:
    """One factor's roll-up, with its tolerance fixed to the number it resolved to.

    A relative tolerance is read off the data, so it is a *fit* in the same sense a derived
    bin edge is: the resolved aggregator carries the number rather than the recipe, and
    replaying it against a second dataset reuses it instead of re-deriving a different one.
    Split to one factor because the number is that factor's -- two columns pooled together
    would share a tolerance neither of them chose.
    """
    return replace(
        aggregator,
        factors=(factor,),
        options={**aggregator.options, "tolerance": tolerance},
        provenance="derived",
    )


def identity_of(aggregator: Aggregator) -> Any:
    """Return what a destination with nothing beneath it answers under this aggregator."""
    return lookup(aggregator.how).identity


def coverage_for(aggregator: Aggregator) -> float:
    """Coverage threshold this aggregator's reduction actually answers to.

    Zero for a reduction missing values do not distort, whatever the aggregator asked for:
    a threshold there would null the answer precisely because some values were absent, which
    is the thing the answer was reporting.
    """
    return aggregator.min_coverage if lookup(aggregator.how).coverage_sensitive else 0.0


def _reject_unknown_options(aggregator: Aggregator, reduction: Reduction) -> None:
    """Refuse an option the reduction does not read.

    An option that is accepted and ignored is the worst outcome available: the caller stated
    something, the result does not reflect it, and nothing says so. Declaring them per
    reduction is what makes ``Aggregator("mean", ..., options={"tolerance": 0.1})`` an error
    rather than a no-op.

    Raises
    ------
    ValueError
        When an option is not one this reduction takes.
    """
    if unknown := sorted(set(aggregator.options) - set(reduction.options)):
        takes = f"takes {list(reduction.options)}" if reduction.options else "takes no options"
        raise ValueError(f"{aggregator.how!r} {takes}, so {unknown} would be accepted and ignored.")


def _reject_bare_tolerance(aggregator: Aggregator) -> None:
    """Refuse a tolerance written as a bare number, whose meaning is not the obvious one.

    A tolerance is a :data:`~dataeval.protocols.ThresholdLike`, and there a bare number is a
    *multiplier on the default* -- so many modified z-scores of the observed changes, not so
    many of the factor's own units. The naive reading is the wrong one and it is silent, so
    the two spellings are asked for by name instead.

    Checked only on a declaration. A resolved aggregator carries the number its threshold
    fitted to, which is the whole point of resolving it, and would otherwise refuse itself
    on replay.

    Raises
    ------
    ValueError
        When ``tolerance`` is a bare number.
    """
    spec = aggregator.options.get("tolerance")
    if spec is None or isinstance(spec, bool) or not isinstance(spec, int | float):
        return
    raise ValueError(
        f"tolerance={spec!r} is a bare number, which a ThresholdLike reads as a multiplier on the "
        f"default rather than as a distance. Say which is meant: ('constant', (None, {spec})) for "
        f"{spec} in the factor's own units, or ('iqr', (None, {spec})) for {spec} times the spread "
        "of the changes actually observed.",
    )


def _reject_untolerable(
    aggregator: Aggregator,
    chosen: Sequence[tuple[str, FactorLevel]],
    dtypes: Mapping[str, pl.DataType],
) -> None:
    """Refuse a tolerance on a factor whose values have no distance between them.

    ``longest_run`` reads any type -- a run of one category is as real as a run of one
    number -- but a *tolerance* asks how far apart two readings are, and subtraction is
    what answers that. Asked of a string factor, polars refused mid-reduction with ``sub
    operation not supported for dtypes 'str' and 'str'`` and a dump of the internal sort
    expression: the raw-polars leak the named reductions exist to prevent.

    Raises
    ------
    ValueError
        When a factor the tolerance would govern does not hold numbers.
    """
    if "tolerance" not in aggregator.options:
        return
    if untolerable := sorted({name for name, _ in chosen if not dtypes[name].is_numeric()}):
        raise ValueError(
            f"tolerance says how far apart two readings may be and still count as unchanged, which "
            f"needs values that can be subtracted, but {untolerable} holds "
            f"{', '.join(sorted({str(dtypes[name]) for name in untolerable}))}. Drop the tolerance to "
            f"count a run as an unbroken stretch of one value.",
        )


def _stated_ordering(order_by: str, source: FactorLevel, columns: frozenset[str]) -> str:
    """Check an ordering the caller named against the columns the source level holds itself.

    Raises
    ------
    ValueError
        When the source level does not hold it.
    """
    if order_by not in columns:
        raise ValueError(
            f"order_by={order_by!r} is not a column {source!r} holds itself, so there is nothing to "
            f"sort its rows by. {source!r} holds {sorted(columns)}.",
        )
    return order_by


def _ordering_for(
    aggregator: Aggregator,
    reduction: Reduction,
    source: FactorLevel,
    native_columns: Mapping[FactorLevel, frozenset[str]],
) -> str | None:
    """Which column a temporal roll-up reads its rows in the order of, or None if it is positional.

    Only a column the source level holds *itself* is eligible. An ordering read down from an
    ancestor repeats once per row beneath it, so sorting by it would order a track's
    detections by the frame they share rather than by anything -- the same fan-out
    ``unique_by`` exists to refuse, arriving through the sort key instead of the values.

    A level with no ordering refuses the reduction rather than falling back to row order.
    Row order is an artifact of the walk that built the store; reading it as time would give
    a confident answer about a sequence that does not exist -- and ``track`` under
    ``sequence`` is exactly such a level, which is why an ``instance -> sequence`` temporal
    roll-up has to go through ``unit``.

    Raises
    ------
    ValueError
        When ``order_by`` names a column the source level does not hold, or when a temporal
        reduction is asked for at a level that carries no ordering.
    """
    if reduction.kind != "temporal":
        return None
    columns = native_columns.get(source, frozenset())
    if aggregator.order_by is not None:
        return _stated_ordering(aggregator.order_by, source, columns)
    found = next((candidate for candidate in ORDERINGS if candidate in columns), None)
    if found is not None:
        return found
    raise ValueError(
        f"{aggregator.how!r} reads {source!r} rows as an ordered series, and {source!r} carries no "
        f"ordering to read them in: none of {list(ORDERINGS)} is among its own columns "
        f"{sorted(columns)}. Name one with order_by=, or ask for a reduction that does not "
        "depend on order.",
    )


def resolve(
    aggregator: Aggregator,
    schema: FactorLevelSchema,
    factor_levels: Mapping[str, FactorLevel],
    dtypes: Mapping[str, pl.DataType],
    native_columns: Mapping[FactorLevel, frozenset[str]],
) -> tuple[Aggregator, ...]:
    """Turn a declaration into fully specified aggregators, one per source level.

    Resolution answers two questions the declaration may leave open: which level each
    factor is rolled up *from*, and — where the factor set is empty — which factors the
    rule selects. Both are read off a dataset, which makes the result a **fit**: it is
    ``provenance="derived"``, and replaying it against a second dataset reuses what it
    recorded rather than asking these questions again, exactly as a recorded
    :class:`~dataeval.types.BinSpec` reapplies its edges instead of re-deriving them.

    Grouped by source level rather than returned one per factor, so that factors sharing a
    source are rolled up in a single grouped pass. A ``target`` fed from two levels — a
    sequence-level roll-up over factors living at ``unit`` and at ``track`` — is genuinely
    two groupings, and comes back as two aggregators.

    **A named factor is a request; an empty factor set is a rule.** Naming a factor the
    reduction cannot apply to raises, because the caller asked for that factor. Leaving the
    set empty selects what fits and says at ``info`` what it passed over, because selecting
    is what the rule is for.

    An already-resolved aggregator is resolved again rather than passed through. Nothing it
    names changes, but the checks live on this path: whether each factor exists, and whether
    the reduction applies to the values it holds. Returning early for it made those checks a
    property of *which surface* the caller used — the string form refused a mean over a
    class label and the declaration form wrote an all-null column that failed much later,
    inside binning, naming neither the factor nor the reduction.

    Parameters
    ----------
    aggregator : Aggregator
        The declaration to resolve.
    schema : FactorLevelSchema
        The dataset's level graph.
    factor_levels : Mapping[str, str]
        Level each factor is defined at.
    dtypes : Mapping[str, pl.DataType]
        Value type of each factor.
    native_columns : Mapping[str, frozenset[str]]
        Columns each level holds itself, which is where a temporal reduction's ordering
        must come from.

    Returns
    -------
    tuple[Aggregator, ...]
        One resolved aggregator per source level, in schema order.

    Raises
    ------
    ValueError
        When ``how`` names no reduction, when the levels do not describe a roll-up, when a
        named factor is unknown, sits at or below ``target``, or holds values the reduction
        does not apply to, or when nothing at all is left to roll up.
    """
    reduction = lookup(aggregator.how)
    aggregator.validate(schema)
    _reject_unknown_options(aggregator, reduction)
    if aggregator.provenance == "declared":
        _reject_bare_tolerance(aggregator)
    if aggregator.factors:
        chosen = _requested(aggregator, reduction, schema, factor_levels, dtypes)
    else:
        chosen = _selected(aggregator, reduction, schema, factor_levels, dtypes)
    if not chosen:
        raise ValueError(
            f"Nothing to roll up into {aggregator.target!r} with {aggregator.how!r}: no factor "
            f"below it holds values {aggregator.how!r} applies to.",
        )
    _reject_untolerable(aggregator, chosen, dtypes)
    grouped = _grouped(aggregator, chosen, schema, reduction, native_columns)
    if not grouped:
        raise ValueError(
            f"Nothing to roll up into {aggregator.target!r} with {aggregator.how!r}: no level below it "
            f"both holds values {aggregator.how!r} applies to and carries an ordering to read them in.",
        )
    return grouped


def _unordered(
    aggregator: Aggregator,
    reduction: Reduction,
    source: FactorLevel,
    native_columns: Mapping[FactorLevel, frozenset[str]],
) -> bool:
    """Whether a *rule* passes this source level over for carrying no ordering.

    The same distinction :func:`_requested` and :func:`_selected` draw about value types,
    applied to the one property that belongs to the level rather than to the factor. A
    named factor is a request, so a level that cannot answer it raises; an empty factor set
    is a rule, and a level with no ordering is one more thing the rule does not select.

    Without this, ``aggregate(level="sequence", how="variability")`` -- the documented rule
    form -- refused outright on any tracking dataset, because ``track`` sits below
    ``sequence`` and carries no ordering, and one such level was enough to take the whole
    call down along with every level that could have answered.

    An ordering the caller *named* is never passed over: they asked for that column, so
    :func:`_ordering_for` still refuses a level that does not hold it.
    """
    if aggregator.factors or reduction.kind != "temporal" or aggregator.order_by is not None:
        return False
    if any(candidate in native_columns.get(source, frozenset()) for candidate in ORDERINGS):
        return False
    _logger.info(
        "Rolling up into %r with %r passed over every factor at %r, which carries no ordering to read "
        "its rows in: none of %s is among its own columns.",
        aggregator.target,
        aggregator.how,
        source,
        list(ORDERINGS),
    )
    return True


def _grouped(
    aggregator: Aggregator,
    chosen: Sequence[tuple[str, FactorLevel]],
    schema: FactorLevelSchema,
    reduction: Reduction,
    native_columns: Mapping[FactorLevel, frozenset[str]],
) -> tuple[Aggregator, ...]:
    """Collect the chosen factors into one aggregator per source level, in schema order."""
    by_source: dict[FactorLevel, list[str]] = {}
    for factor, source in chosen:
        by_source.setdefault(source, []).append(factor)
    order = {level: position for position, level in enumerate(schema.levels)}
    resolved = tuple(
        _at(aggregator, source, tuple(factors), _ordering_for(aggregator, reduction, source, native_columns))
        for source, factors in sorted(by_source.items(), key=lambda item: order[item[0]])
        if not _unordered(aggregator, reduction, source, native_columns)
    )
    for one in resolved:
        one.validate(schema)
    return resolved


def _at(aggregator: Aggregator, source: FactorLevel, factors: tuple[str, ...], order_by: str | None) -> Aggregator:
    """One resolved aggregator, recording that its source and factor set were inferred.

    A fit stays a fit. An aggregator that already arrived ``derived`` carries something read
    off a dataset — a resolved level, a selected factor set, a fitted tolerance — and
    re-labelling it ``declared`` because it now names both would claim a caller wrote what a
    previous resolution measured, and hand it back to the checks a declaration answers to.
    """
    inferred = aggregator.provenance == "derived" or aggregator.source is None or not aggregator.factors
    return Aggregator(
        how=aggregator.how,
        source=source,
        target=aggregator.target,
        factors=factors,
        unique_by=aggregator.unique_by,
        via=aggregator.via,
        order_by=order_by,
        options=aggregator.options,
        min_coverage=aggregator.min_coverage,
        suffix=aggregator.suffix,
        provenance="derived" if inferred else "declared",
    )


def _source_of(aggregator: Aggregator, factor: str, factor_levels: Mapping[str, FactorLevel]) -> FactorLevel:
    """Level a factor is rolled up from: the declared one, else the one that defines it."""
    return aggregator.source if aggregator.source is not None else factor_levels[factor]


def _requested(
    aggregator: Aggregator,
    reduction: Reduction,
    schema: FactorLevelSchema,
    factor_levels: Mapping[str, FactorLevel],
    dtypes: Mapping[str, pl.DataType],
) -> list[tuple[str, FactorLevel]]:
    """Resolve the factors a caller named, refusing every one that does not fit."""
    chosen: list[tuple[str, FactorLevel]] = []
    for factor in aggregator.factors:
        if factor not in factor_levels:
            raise ValueError(
                f"{factor!r} is not a factor of this metadata. Its factors are {sorted(factor_levels)}.",
            )
        source = _source_of(aggregator, factor, factor_levels)
        if not schema.is_ancestor(aggregator.target, source):
            raise ValueError(
                f"Cannot roll {factor!r} up into {aggregator.target!r}: it is defined at {source!r}, "
                f"which {aggregator.target!r} does not sit above. Levels above {source!r} are "
                f"{list(schema.ancestors(source))}.",
            )
        if not admits(reduction, dtypes[factor]):
            applicable = sorted(name for name, other in REDUCTIONS.items() if admits(other, dtypes[factor]))
            raise ValueError(
                f"{aggregator.how!r} does not apply to {factor!r}, whose values are "
                f"{dtypes[factor]}; {aggregator.how!r} takes {reduction.domain} values. "
                f"Reductions that apply to {factor!r} are {applicable}.",
            )
        chosen.append((factor, source))
    return chosen


def _in_scope(
    aggregator: Aggregator,
    factor: str,
    factor_levels: Mapping[str, FactorLevel],
    schema: FactorLevelSchema,
) -> bool:
    """Whether a rule reaches this factor at all, before asking what its values are."""
    if aggregator.source is not None and factor_levels[factor] != aggregator.source:
        return False
    return schema.is_ancestor(aggregator.target, _source_of(aggregator, factor, factor_levels))


def _selected(
    aggregator: Aggregator,
    reduction: Reduction,
    schema: FactorLevelSchema,
    factor_levels: Mapping[str, FactorLevel],
    dtypes: Mapping[str, pl.DataType],
) -> list[tuple[str, FactorLevel]]:
    """Select every factor the rule admits, saying out loud what it passed over."""
    in_scope: list[tuple[str, FactorLevel]] = [
        (factor, _source_of(aggregator, factor, factor_levels))
        for factor in sorted(factor_levels)
        if _in_scope(aggregator, factor, factor_levels, schema)
    ]
    chosen: list[tuple[str, FactorLevel]] = [pair for pair in in_scope if admits(reduction, dtypes[pair[0]])]
    skipped = [factor for factor, _ in in_scope if not admits(reduction, dtypes[factor])]
    if skipped:
        _logger.info(
            "Rolling up into %r with %r passed over %s, whose values %r does not apply to.",
            aggregator.target,
            aggregator.how,
            sorted(skipped),
            aggregator.how,
        )
    return chosen
