"""Normalizing what a caller passed before anything trusts it.

:meth:`~dataeval.Metadata.from_factors` and :meth:`~dataeval.Metadata.add_factors` accept
the same shapes of input — a factor mapping, a whole stats result — and the two must
agree on what each one means. These functions are that agreement, held once rather than
restated on each path, and they are pure: they translate and reject, and never touch the
metadata they are about to be used on.
"""

__all__ = []

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from dataeval.exceptions import ShapeMismatchError
from dataeval.types import Aggregator, Array1D, SourceIndex


def _is_stats_result(candidate: Any) -> bool:
    """Whether a mapping is a :class:`~dataeval.core.StatsResult` rather than a factor mapping.

    ``StatsResult`` is a :class:`~typing.TypedDict`, so at runtime it is an ordinary dict;
    there is nothing to check with ``isinstance``. ``stats`` is the marker. A factor mapping
    could hold a factor named ``stats``, but not one whose value is itself a mapping rather
    than an array of values.

    ``source_index`` is checked only when present. A producer that places its values by level
    and key carries no addresses, and requiring them would hide its result.
    """
    if not isinstance(candidate, Mapping) or "stats" not in candidate:
        return False
    if not isinstance(candidate["stats"], Mapping):
        return False
    if "source_index" not in candidate:
        return True
    source_index = candidate["source_index"]
    # The first entry stands for the sequence: a stats result's index is homogeneous by
    # construction, and this runs on every add_factors call.
    return (
        isinstance(source_index, Sequence)
        and not isinstance(source_index, str)
        and (not source_index or isinstance(source_index[0], SourceIndex))
    )


def unpack_stats_result(
    factors: Any,
    source_index: Sequence[SourceIndex] | None,
    *,
    level: Any = None,
) -> tuple[Mapping[str, Array1D[Any]], Sequence[SourceIndex] | None, tuple[Aggregator, ...]]:
    """Accept a whole stats result wherever a factor mapping is accepted.

    A producer returns its statistics, their placement, and the recipes for rolling them up
    in one object. Unpacking them here keeps callers from passing one without the others.

    Returns
    -------
    tuple
        The factors, the placement, and the producer's roll-up declarations. The declarations
        are an empty tuple where it declared none, or where the argument was an ordinary
        factor mapping.

    Raises
    ------
    ValueError
        When a level is named as well as a source index carried by the result. The result
        already says what each value describes, so the two cannot both be honoured. A result
        with no source index places nothing by address, so a level for it is allowed.
    """
    if not _is_stats_result(factors):
        return factors, source_index, ()
    carried = factors.get("source_index")
    if level is not None and carried is not None:
        raise ValueError(
            f"`level` and the source_index carried by this stats result are mutually exclusive; "
            f"the result already labels each value with what it describes. Pass the result's "
            f"['stats'] mapping instead to place its values at level={level!r}.",
        )
    aggregations = tuple(factors.get("aggregations", ()))
    return factors["stats"], source_index if source_index is not None else carried, aggregations


def reject_length_mismatch(factors: Mapping[str, Any], source_index: Sequence[SourceIndex]) -> None:
    """Reject factors that do not hold exactly one value per source-index entry.

    Shared by both constructors: the source index is the placement, so a factor that is
    not as long as it names rows the caller never described, whichever spelling was used
    to get here.
    """
    mismatched = {name: len(values) for name, values in factors.items() if len(values) != len(source_index)}
    if mismatched:
        raise ShapeMismatchError(
            f"All factors must have one value per source_index entry ({len(source_index)}); got {mismatched}.",
        )


def build_index2label(
    provided: Mapping[int, str] | None,
    observed_labels: Iterable[Any],
) -> dict[int, str]:
    """Map each class index to a name, backfilling observed labels missing from ``provided``.

    When ``provided`` is given it is the source of truth; any observed label without an
    entry gets an ``UNDEFINED_CLASS_<i>`` placeholder. Otherwise labels name themselves.
    """
    if provided is not None:
        index2label = {int(k): str(v) for k, v in provided.items()}
        for lbl in observed_labels:
            index2label.setdefault(int(lbl), f"UNDEFINED_CLASS_{int(lbl)}")
        return index2label
    return {int(lbl): str(int(lbl)) for lbl in observed_labels}


def resolve_aggregations(
    declared: tuple[Aggregator, ...],
    aggregate: bool,
    how: Mapping[str, str] | None,
    aggregations: Sequence[Aggregator] | None,
) -> tuple[Aggregator, ...]:
    """Decide which roll-ups a call actually runs.

    Three forms, in order of precedence: suppress everything, replace everything, or swap the
    reduction for named factors and keep the rest of the declaration.

    Raises
    ------
    ValueError
        When both override forms are given, or when an override names a factor the result
        does not declare. An override that would change nothing is an error, not a silent
        no-op.
    """
    if how is not None and aggregations is not None:
        raise ValueError(
            "`how` and `aggregations` are mutually exclusive: the first swaps a reduction in the "
            "result's own declarations, the second replaces them entirely.",
        )
    if not aggregate:
        # Suppressing nothing is a no-op, not an error.
        return ()
    if aggregations is not None:
        return tuple(aggregations)
    if how is None:
        return declared
    return _apply_how(declared, how)


def _apply_how(declared: tuple[Aggregator, ...], how: Mapping[str, str]) -> tuple[Aggregator, ...]:
    """Swap the reduction of each named factor across a set of declarations.

    Split out of :func:`resolve_aggregations` to keep that function's branching within the
    project's complexity ceiling. The two checks below make `how` raise when it would
    override nothing.
    """
    if not declared:
        raise ValueError(
            "This result declares no roll-ups, so there is nothing for `how` to override. "
            "Pass `aggregations=[...]` to declare them yourself.",
        )
    unknown = sorted(set(how) - {factor for one in declared for factor in one.factors})
    if unknown:
        raise ValueError(
            f"This result does not declare a roll-up for {unknown}, so `how` cannot override it. "
            f"It declares: {sorted({f for one in declared for f in one.factors})}.",
        )
    return tuple(agg for one in declared for agg in _swapped(one, how))


def _swapped(aggregator: Aggregator, how: Mapping[str, str]) -> tuple[Aggregator, ...]:
    """One declaration, split around an override that names some of its factors.

    A declaration covering several factors is split, not just relabelled. The factors the
    override names move to their own declaration under the new reduction. Any factor it does
    not name stays on the original one, unchanged. Naming none of the declaration's factors
    returns it unchanged, in a one-tuple.
    """
    overridden = {factor: how[factor] for factor in aggregator.factors if factor in how}
    if not overridden:
        return (aggregator,)
    reductions = set(overridden.values())
    if len(reductions) > 1:
        # One declaration cannot carry two reductions; split it per reduction instead.
        raise ValueError(
            f"`how` asks for {sorted(reductions)} across factors declared together as "
            f"{list(aggregator.factors)}; override them in separate calls or pass `aggregations`.",
        )
    swapped = replace(aggregator, how=reductions.pop(), factors=tuple(overridden))
    remaining = tuple(factor for factor in aggregator.factors if factor not in overridden)
    return (swapped,) if not remaining else (swapped, replace(aggregator, factors=remaining))
