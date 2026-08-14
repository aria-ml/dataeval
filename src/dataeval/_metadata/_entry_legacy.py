"""The v1.1 factor-entry spellings that infer a destination instead of being told one.

Two retired ways of saying where a factor's values belong, kept working for one more
release: ``level="combined"``, which named an array ordered by ``(item, target)`` rather
than a level at all, and ``level="auto"``'s inference of a level from an array's length.
Both exist only because the destination used to be a single stacked frame; with one frame
per level there is a level to name, and :meth:`~dataeval.Metadata.add_factors` asks for it.

This module is the whole of that machinery, gathered so that retiring it is deleting a
file rather than unpicking a class. Nothing outside factor entry reaches in here, and
nothing in here is reachable except through a deprecated spelling.

.. deprecated::
    Removed in v1.2.0 along with ``level="combined"``. Pass ``level=`` or
    ``source_index=``.
"""

__all__ = []

import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, NoReturn

import numpy as np
import polars as pl
from numpy.typing import NDArray

from dataeval.exceptions import ShapeMismatchError
from dataeval.types import Array1D, FactorLevel

if TYPE_CHECKING:
    from dataeval._metadata._metadata import Metadata


def _combined_length(md: "Metadata") -> int | None:
    """Length an *inferred* v1.1 ``"combined"`` array has here, or None when there is none.

    Stricter than what :func:`_resolve_combined` accepts, on purpose. Inference is a
    guess made from a length alone, so it is offered only where the guess is worth
    making: a two-level schema over a multi-target task. A classification dataset
    carries one label per image, so its combined length is merely twice the image
    count — far more likely a caller's mistake than a deliberate two-level array, and
    v1.1 did not infer it there either. An explicit ``level="combined"`` is the caller
    asserting the layout rather than the code guessing it, so that spelling is refused
    only where the split is structurally impossible.

    A schema with a third level, as tracking's frames and tracks are, has no combined
    length under either spelling.
    """
    if len(md._levels) != 2 or not md.multi_target:
        return None
    counts = md._store.counts
    return counts.get(md._item_level, 0) + counts.get(md._label_level, 0)


def _reject_unmatched_length(md: "Metadata", factor_len: int, combined_len: int | None) -> NoReturn:
    """Report a factor length that names neither a level nor a combined array."""
    counts = md._store.counts
    expected = ", ".join(f"{level}={counts.get(level, 0)}" for level in md._levels)
    if combined_len is not None:
        expected += f", {md._item_level}+{md._label_level}={combined_len}"
    raise ShapeMismatchError(
        "The lists/arrays in the provided factors have a different length "
        f"than any level of the current metadata. Expected one of ({expected}), got {factor_len}.",
    )


def infer_factor_level(md: "Metadata", factor: Array1D[Any]) -> "FactorLevel | Literal['combined']":
    """Infer the destination of a single factor array from its length.

    A level's own row count wins over the combined length, so a factor that could be
    read either way lands on a level rather than being split.

    Raises
    ------
    ShapeMismatchError
        When the length matches no level and no combined length.
    """
    factor_len = len(factor)
    counts = md._store.counts
    matches: list[FactorLevel] = [level for level in md._levels if counts.get(level, 0) == factor_len]

    if not matches:
        combined_len = _combined_length(md)
        if combined_len is not None and factor_len == combined_len:
            return "combined"
        _reject_unmatched_length(md, factor_len, combined_len)
    if len(matches) > 1:
        # Levels routinely coincide in size — a fully labelled classification dataset has
        # one label per image, and so does an object detection dataset with one
        # detection per image — so this cannot raise; that would break code that works on
        # every other dataset. The coarsest level wins, matching what add_factors has
        # always done.
        chosen = md._levels.highest(matches)
        if not all(_rows_correspond(md, chosen, other) for other in matches if other != chosen):
            warnings.warn(
                f"A factor length of {factor_len} matches the {matches} levels, which currently have "
                f"the same number of rows but do not correspond one-to-one; storing it at the "
                f"{chosen!r} level. Pass an explicit level= to add_factors to choose.",
                UserWarning,
                # caller -> add_factors -> _resolve_factor_levels -> here. One frame
                # shallower than _warn_inferred_combined, which _resolve_destinations
                # reaches from _resolve_factor_levels rather than from the loop.
                # test_ambiguity_warning_points_at_the_caller pins this.
                stacklevel=4,
            )
        return chosen
    return matches[0]


def _rows_correspond(md: "Metadata", coarse: FactorLevel, fine: FactorLevel) -> bool:
    """Whether each ``fine`` row has its own ``coarse`` row, in the same order.

    When it does, the two levels are interchangeable as a destination: the values
    land on the same target rows either way, so there is nothing for the caller to
    disambiguate. When it does not — three detections spread 0/1/2 across three
    images — the choice changes the data and has to be surfaced.
    """
    if not md._levels.is_ancestor(coarse, fine):
        return False
    size = md._store.height(fine)
    return np.array_equal(md._store.positions_from(fine, coarse), np.arange(size, dtype=np.intp))


def resolve_combined(
    md: "Metadata",
    factors: Mapping[str, NDArray[Any]],
) -> tuple[list[tuple[str, FactorLevel, NDArray[Any]]], list[str]]:
    """Split a v1.1 ``"combined"`` array into one factor per level.

    ``"combined"`` was never a level. It was v1.1's name for an array ordered by
    ``(item, target)`` — each item's item-level value ahead of that item's label-level
    ones, exactly as :func:`~dataeval.core.compute_stats` emits them — which is what
    `source_index` replaces with an explicit label per value.

    The order is interleaved, *not* one item-level block followed by one label-level
    block. The two readings agree on nothing beyond the first value, so splitting
    positionally silently scatters every value onto the wrong row. Ranking the rows in
    that order and deferring the gather to :meth:`Metadata._place` keeps the deprecated
    spelling and its replacement placing identical data, and keeps the naming rule in
    one place rather than in two implementations that can drift apart.

    Raises
    ------
    ValueError
        When items and labels sit at the same level, where the split is ambiguous.
    ShapeMismatchError
        When a factor is not as long as the two levels' rows combined.
    """
    item_level, label_level = md._item_level, md._label_level
    if item_level == label_level:
        raise ValueError(
            f"level='combined' describes two levels, but this metadata's items and its labels "
            f"are both at the {item_level!r} level, so there is no split to make. Add each "
            "level's values in its own call, or pass source_index= to place them by label.",
        )
    if len(md._levels) != 2:
        # Two levels is not incidental to "combined": it was v1.1's name for an array
        # over the whole dataframe, and v1.1 had no schema with a third level. On a
        # schema that does — tracking puts ``image`` and ``track`` between ``sequence``
        # and ``instance`` — splitting item/label still type-checks and still produces
        # a plausible-looking pair of factors, while silently describing none of the
        # rows in between. Refuse rather than half-cover the dataframe.
        raise ValueError(
            f"level='combined' describes an array over exactly two levels, but this metadata "
            f"has {list(md._levels)}. An array of {item_level}-level values interleaved with "
            f"{label_level}-level ones would say nothing about the rows at the levels between "
            "them, so there is no array it can name here. Pass source_index= to place values by "
            "label, or level= to name the one level they belong to.",
        )

    counts = md._store.counts
    head, tail = counts.get(item_level, 0), counts.get(label_level, 0)
    mismatched = {name: len(values) for name, values in factors.items() if len(values) != head + tail}
    if mismatched:
        raise ShapeMismatchError(
            f"All combined-level factors must have length {head + tail} "
            f"({item_level} count {head} + {label_level} count {tail}); got {mismatched}.",
        )
    # qualify=True rather than letting the data decide: the deprecation warning has
    # already promised '<item_level>_<name>' and '<label_level>_<name>', and a dataset
    # whose label level happens to have no rows — or whose values at one of them are
    # all null — must not silently rename or remove the columns out from under a
    # caller following that warning. It therefore keeps both halves unconditionally.
    return md._place(factors, _combined_positions(md), qualify=True)


def _combined_positions(md: "Metadata") -> dict[FactorLevel, NDArray[np.intp]]:
    """Position within a v1.1 ``"combined"`` array of each row it described.

    The array was ordered by ``(item, target)`` with an item's own value ahead of that
    item's labels, so each row's position is simply its rank in that order — read off
    the two key columns rather than rebuilt as :class:`~dataeval.types.SourceIndex`
    objects and re-parsed, which would allocate one per row of the dataframe and sort
    it three more times to arrive back here.

    Every row of the dataframe is ranked, which is the whole of it: :func:`_resolve_combined`
    has already established that the schema is exactly the item level and the label
    level, so there are no rows in between for a combined array not to describe.
    """
    # Two columns per level off the store, not :attr:`~dataeval.Metadata.dataframe`: the
    # ranking needs the item and target keys and nothing else, and widening every level
    # to every column to read them would rebuild the whole flat frame — which the very
    # next store write then throws away. The levels are read in the store's own row
    # order, which is the order the flat frame stacks them in, so the ranks are the same
    # ones the flat frame would have produced.
    keys = {
        level: md._store.select(level, ("item_index", "target_index")).select(
            "item_index",
            # Null marks a per-item row. -1 both stands in for it and, sorting below
            # every real target, puts an item's value ahead of that item's labels.
            pl.col("target_index").fill_null(-1),
        )
        for level in md._store.frames
    }
    items = np.concatenate([frame["item_index"].to_numpy() for frame in keys.values()]) if keys else np.empty(0)
    targets = np.concatenate([frame["target_index"].to_numpy() for frame in keys.values()]) if keys else np.empty(0)
    order = np.lexsort((targets, items))
    rank = np.empty(len(order), dtype=np.intp)
    rank[order] = np.arange(len(order), dtype=np.intp)

    starts = np.cumsum([0, *(frame.height for frame in keys.values())])
    offsets = {level: (int(starts[i]), int(starts[i + 1])) for i, level in enumerate(keys)}
    return {level: rank[slice(*offsets.get(level, (0, 0)))] for level in (md._item_level, md._label_level)}


def resolve_destinations(
    md: "Metadata",
    destinations: Sequence[tuple[str, "FactorLevel | Literal['combined']", NDArray[Any]]],
) -> tuple[list[tuple[str, FactorLevel, NDArray[Any]]], list[str]]:
    """Turn inferred destinations into columns, batching the combined ones.

    Batched rather than resolved one at a time because :func:`_resolve_combined` ranks
    every row of the dataframe. The default call —
    ``add_factors(compute_stats(...)["stats"])`` — brings ~20 statistics of the same
    length, so per-factor resolution would repeat that ranking ~20 times.
    """
    resolved: list[tuple[str, FactorLevel, NDArray[Any]]] = [
        (name, level, values) for name, level, values in destinations if level != "combined"
    ]
    _warn_inferred_level(md, [(name, level) for name, level, _ in resolved])
    vacuous: list[str] = []
    combined = {name: values for name, level, values in destinations if level == "combined"}
    if combined:
        _warn_inferred_combined(md, sorted(combined))
        placed, vacuous = resolve_combined(md, combined)
        resolved.extend(placed)
    return resolved, vacuous


def _warn_inferred_level(md: "Metadata", placed: Sequence[tuple[str, FactorLevel]]) -> None:
    """Warn that a factor's level was guessed from its array length.

    The last silent guess in :meth:`~dataeval.Metadata.add_factors`, and the least
    defensible one. A length names a level only by coincidence: levels routinely coincide
    in size — a fully labelled classification dataset has one label per image, and so does
    a detection dataset with one detection per image — so the same array lands somewhere
    different depending on what the dataset happens to contain, and nothing says so. The
    two cases that *are* surfaced today are the ones inference cannot resolve at all; this
    is the one where it resolves confidently and may still be wrong.

    ``level=`` states the destination and ``source_index=`` labels each value, and both
    are available wherever this is: the array came from somewhere that knew what it
    described.

    Scoped to :meth:`~dataeval.Metadata.add_factors`. The other constructor,
    :meth:`~dataeval.Metadata.from_factors`, also defaults its level — bare arrays carry
    nothing that distinguishes an item from a label, so it places every row at one level
    — but that is a documented default rather than a guess between candidates, and it is
    not deprecated here.

    Each factor is named with the level it actually reached, rather than the names and the
    levels being listed separately — a batch may span several levels, and two lists side
    by side can only be read as pairs whether or not they are in the same order.

    Raised once for the whole batch, for the same reason
    :func:`_warn_inferred_combined` is: the call this fires on brings a mapping of many
    statistics at once, and one paragraph per factor buries the action it asks for.
    """
    if not placed:
        return
    levels = {level for _, level in placed}
    named = ", ".join(f"{name!r} at {level!r}" for name, level in placed)
    if len(md._levels) == 1:
        # The coincidence rationale would be false here: with one level the placement is
        # forced. The deprecation still applies — the *call* does not say where its values
        # belong, and the same call resolves differently against a dataset that has more
        # levels — so the warning stands with the reason that is actually true.
        why = (
            f"This metadata has only the {next(iter(md._levels))!r} level, so the placement is "
            "forced here, but the call does not say so and the same call resolves differently "
            "against a dataset with more levels. "
        )
    else:
        why = (
            "Levels can coincide in size, so a length identifies a level only by coincidence and "
            "the same array can land elsewhere on a differently shaped dataset. "
        )
    # A batch that reached several levels cannot be restated as one ``level=`` call, so
    # the remediation has to name the shape that actually replaces it.
    fix = (
        "add each level's factors in its own call with level=, or pass source_index= to label each value"
        if len(levels) > 1
        else "pass level= to state the destination, or source_index= to label each value"
    )
    warnings.warn(
        f"The level of factor(s) {named} was inferred from their array length. {why}"
        f"Inferring this is deprecated and will be removed in v1.2.0; {fix}.",
        DeprecationWarning,
        # caller -> add_factors -> _resolve_factor_levels -> resolve_destinations -> here,
        # the same depth _warn_inferred_combined reaches from the same call site.
        stacklevel=5,
    )


def _warn_inferred_combined(md: "Metadata", names: Sequence[str]) -> None:
    """Warn that factors were placed by the retired combined convention.

    Inference reaching ``"combined"`` means the values were placed by their position
    in an undeclared ordering rather than by a label. That is the same bet
    ``level="combined"`` makes, so it earns the same warning — the caller has a
    `source_index` available whenever the array came from
    :func:`~dataeval.core.compute_stats`, and passing it removes the guess.

    Raised once for the whole batch rather than once per factor: the call this fires
    on is ``add_factors(compute_stats(...)["stats"])``, which brings ~20 statistics of
    the same length, and twenty copies of one paragraph bury the one action it asks
    for.
    """
    warnings.warn(
        f"Factor(s) {list(names)} are as long as the {md._item_level} and {md._label_level} "
        "levels combined, so their values were placed by the ordering compute_stats emits — by "
        f"(item, target), each item's {md._item_level}-level value ahead of that item's "
        f"{md._label_level}-level ones — and each was split into '{md._item_level}_<name>' "
        f"and '{md._label_level}_<name>'. Inferring this is deprecated and will be removed in "
        "v1.2.0; pass source_index= from compute_stats, which labels each value instead.",
        DeprecationWarning,
        # caller -> add_factors -> _resolve_factor_levels -> _resolve_destinations ->
        # here. test_inference_warnings_point_at_the_caller pins this.
        stacklevel=5,
    )
