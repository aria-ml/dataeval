__all__ = []

from collections.abc import Iterable, Mapping, Sequence
from enum import Flag
from typing import Any, Generic, NotRequired

import numpy as np
from typing_extensions import TypedDict, TypeVar

from dataeval._experimental import deprecated
from dataeval.core._calculators._base import Calculator
from dataeval.flags import ImageStats
from dataeval.protocols import ArrayLike, ImageDataset, ProgressCallback
from dataeval.types import Aggregator, SourceIndex, StatsMap
from dataeval.utils.preprocessing import (
    BoxLike,
    ChannelGroupLike,
)

SOURCE_INDEX = "source_index"

BACKGROUND_PREFIX = "background_"
"""Name prefix distinguishing a background statistic from the whole-image one beside it.

Background values share a row with the full image's — both describe the same item and
carry the same :class:`~dataeval.types.SourceIndex` — so they are told apart by name
rather than by position. Adding a row instead is not available: a source index addresses
exactly one item-level row per item, and naming it twice is rejected outright.
"""


StatsRequest = Flag | Mapping[str | None, Flag] | Mapping[str, Flag]
"""What `stats` may be: one flag set for every view, or one flag set per named view.

``Mapping[str, Flag]`` is spelled out beside ``Mapping[str | None, Flag]`` rather than
being covered by it, because a mapping's key type is invariant. Without it a
``dict[str, ImageStats]`` assembled a line before the call — the ordinary way to build one
from configuration, and exactly the mapping-with-no-whole-image-entry the feature exists to
accept — is rejected by a type checker while the identical dict literal written inline is
not. Only reads happen here, so accepting the narrower key type is sound.
"""


TFactors = TypeVar("TFactors", bound=Mapping[str, Any], default=StatsMap)


class FactorResult(TypedDict, Generic[TFactors]):
    """
    The shape every statistics producer returns.

    One shape rather than one per producer. A consumer accepts any result without knowing
    which function made it, and a producer can carry the recipes for rolling its factors up
    alongside the values. Producers that address every value to a row return the richer
    :class:`StatsResult`.

    Attributes
    ----------
    stats : Mapping[str, Any]
        The measurements, as a mapping of factor name to one value per row. Parameterized
        per producer: ``FactorResult[TrackFactors]`` narrows this to the factors
        ``track_stats`` measures, so ``result["stats"]["mean_speed"]`` is checked rather
        than ``Any``. Bare, it defaults to :data:`~dataeval.types.StatsMap`
        (``Mapping[str, NDArray[Any]]``). ``image_stats`` and ``compute_ratios`` return
        one NumPy array per statistic, object dtype for string-valued ones such as hashes.
    aggregations : tuple[Aggregator, ...], optional
        How these factors roll up to a coarser level, declared by the producer.
        :meth:`~dataeval.Metadata.add_factors` applies them unless told not to. Absent where
        a producer declares nothing.
    categorical : tuple[str, ...], optional
        Factors whose integer values are category codes, such as class ids. :meth:`~dataeval.Metadata.add_factors`
        records each as a declared vocabulary, as ``factor_levels=`` would, so consumers treat the column as
        categorical. A vocabulary the caller already declared for the name is kept.
    """

    stats: TFactors
    aggregations: NotRequired[tuple[Aggregator, ...]]
    categorical: NotRequired[tuple[str, ...]]


class AddressedResult(FactorResult[TFactors]):
    """
    A :class:`FactorResult` that also carries an address for every value.

    Rows placed by address, for producers with no image bookkeeping (``video_stats``
    places one row per video). What combining or offsetting results needs, so
    :func:`combine_stats_results` accepts any of these. Producers that place their values
    by level and key instead (``track_stats``) return a bare :class:`FactorResult`, with no
    address to offset or concatenate against.

    Attributes
    ----------
    source_index : Sequence[SourceIndex]
        An address per value, saying which row of which level it belongs to.
    """

    source_index: Sequence[SourceIndex]


class StatsResult(AddressedResult[TFactors]):
    """
    An :class:`AddressedResult` with image bookkeeping.

    What ``image_stats`` and ``compute_ratios`` return.

    Attributes
    ----------
    object_count : Sequence[int]
        Objects per image. Bookkeeping about the run, not a factor.
    invalid_box_count : Sequence[int]
        Invalid boxes per image. Bookkeeping about the run, not a factor.
    image_count : int
        Total images processed. Bookkeeping about the run, not a factor.
    """

    object_count: Sequence[int]
    invalid_box_count: Sequence[int]
    image_count: int


def _reconcile_stats(
    calculator_output: list[dict[str, list[Any]]],
) -> dict[str, list[Any]]:
    """Merge every calculator's output for one row into a single mapping.

    A row's names come from several calculators — the unmasked pass, one per named band
    group, and the masked pass beside them — each returning one value per name, wrapped in
    a single-element list so that a name a calculator does not produce stays absent rather
    than becoming a value.

    Merging is all reconciliation amounts to now that a row is one row. While the
    per-channel path existed a calculator could return one value per band, and this
    function placed each at its own position in a row block sized by the datum's channel
    count; with that path gone there is no second position for a value to land at, and a
    name produced by two calculators keeps the last one's answer exactly as it did when
    both wrote to the same ``channel=None`` slot.

    Raises
    ------
    ValueError
        When a calculator returns other than one value per name. Every calculator is
        expected to reduce its region to a single reading; more than one has nowhere to go
        and would otherwise be silently truncated to the first.
    """
    reconciled_stats: dict[str, list[Any]] = {}
    for output in calculator_output:
        # Checked once per calculator rather than once per name, which is where the
        # per-channel count check sat and what it cost: every name in one output shares a
        # length by construction.
        first_stat_values = next(iter(output.values()))
        if len(first_stat_values) != 1:
            raise ValueError(
                f"Calculator produced {len(first_stat_values)} values for a single row; "
                f"statistics reduce their region to one value each.",
            )
        reconciled_stats.update(output)
    return reconciled_stats


def _produced_stat_names(calculators: Sequence[tuple[type[Calculator[Any]], Flag]]) -> set[str]:
    """Every column name the requested flags will produce, known before any datum is read."""
    names: set[str] = set()
    for calculator_cls, flags in calculators:
        names |= calculator_cls.stat_names(flags)
    return names


@deprecated(since="1.2", removal="2.0", alternative="image_stats")
def compute_stats(
    data: Iterable[ArrayLike] | ImageDataset,
    *,
    boxes: Iterable[Iterable[BoxLike] | None] | None = None,
    stats: StatsRequest = ImageStats.ALL,
    per_image: bool = True,
    per_target: bool = True,
    per_background: bool = False,
    channels: Mapping[str, ChannelGroupLike] | None = None,
    normalize_pixel_values: bool = False,
    value_range: tuple[float, float] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> StatsResult:
    """
    Compute specified statistics on a set of images, optionally within bounding boxes.

    .. deprecated:: 1.2
        Renamed to :func:`~dataeval.core.image_stats`; removed in 2.0. The parameters and the
        result are unchanged.

    See Also
    --------
    :func:`~dataeval.core.image_stats` : The same function under its current name
    """
    # Imported here: `_image_stats` builds on this module's shared types and helpers.
    from dataeval.core._image_stats import image_stats

    return image_stats(
        data,
        boxes=boxes,
        stats=stats,
        per_image=per_image,
        per_target=per_target,
        per_background=per_background,
        channels=channels,
        normalize_pixel_values=normalize_pixel_values,
        value_range=value_range,
        progress_callback=progress_callback,
    )


def require_same_stat_names(
    left: Iterable[str],
    right: Iterable[str],
    *,
    left_label: str,
    right_label: str,
    summary: str,
    remedy: str,
) -> None:
    """Refuse to pair two results that do not describe the same statistics.

    Silently keeping the intersection was harmless while the requested flags fixed the name
    set, since two results could only disagree if the caller had asked for different flags.
    Naming band groups makes the name set a call-site argument: two runs given different
    ``channels=`` mappings differ in exactly the columns the caller cared enough to name,
    and dropping them leaves a result that looks complete.

    Shared by every place that pairs two stat maps, so the rule and the shape of the report
    stay together. Callers filter the names first where their own pairing tolerates a
    difference — `compute_ratios` drops the background columns, which legitimately exist on
    the image side alone.
    """
    only_left = sorted(set(left) - set(right))
    only_right = sorted(set(right) - set(left))
    if not only_left and not only_right:
        return
    detail = ", ".join(
        part
        for part in (
            f"only in {left_label}: {only_left}" if only_left else "",
            f"only in {right_label}: {only_right}" if only_right else "",
        )
        if part
    )
    raise ValueError(f"{summary} ({detail}). {remedy}")


def _reject_unaddressable(result: FactorResult, position: int | None = None) -> None:
    """Refuse a result that carries no `source_index` at all.

    `NotRequired` made this key optional so producers like `track_stats`, which place their
    values by level and key rather than by address, can share `StatsResult`. That fits
    `Metadata.add_factors`, which reads placement either way. It does not fit here: without
    an address per value there is nothing to offset or concatenate against.
    """
    if "source_index" in result:
        return
    where = "" if position is None else f" (result {position} of the sequence)"
    raise TypeError(
        f"Cannot combine a result{where} that carries no 'source_index': it places its "
        "values by level and key rather than by address, as ego_stats and track_stats do, so "
        "there is no address to offset or concatenate here. Attach it directly with "
        "Metadata.add_factors(result, level=..., key=...) instead of combining it.",
    )


def _reject_stat_name_mismatch(combined: StatsMap, stats: StatsMap, position: int) -> None:
    """Refuse to combine results computed over different statistics."""
    require_same_stat_names(
        combined,
        stats,
        left_label="the results so far",
        right_label=f"result {position}",
        summary="Cannot combine results computed over different statistics",
        remedy=(
            "Every result must be computed with the same 'stats' flags and the same "
            "'channels' mapping; otherwise the combined arrays would describe different "
            "things under one name."
        ),
    )


def combine_stats_results(  # noqa: C901
    results: AddressedResult | Sequence[AddressedResult],
) -> tuple[StatsMap, list[SourceIndex], list[int]]:
    """Combine one or more addressed results into unified stats, source_index, and dataset_steps.

    For a single result, returns its stats and source_index directly
    with empty dataset_steps.

    For multiple results, concatenates stats arrays by key, applies cumulative
    item offsets to source_index entries (making item indices globally unique
    across datasets), and computes cumulative dataset_steps boundaries.

    Parameters
    ----------
    results : AddressedResult or Sequence[AddressedResult]
        A single result or sequence of results to combine.

    Returns
    -------
    tuple[StatsMap, list[SourceIndex], list[int]]
        - stats: Combined statistics mapping (arrays concatenated by key).
        - source_index: Combined source indices with globally unique item values.
        - dataset_steps: Cumulative boundaries where each dataset ends in the
          combined arrays. Empty list for a single result.

    Raises
    ------
    TypeError
        If an empty sequence is provided, or if a result carries no `source_index`, as a
        `track_stats` result does, placed by level and key instead.
    """
    if isinstance(results, dict):
        _reject_unaddressable(results)
        return results["stats"], list(results["source_index"]), []

    if len(results) == 0:
        raise TypeError("Cannot combine empty sequence of stats.")

    if len(results) == 1:
        _reject_unaddressable(results[0])
        return results[0]["stats"], list(results[0]["source_index"]), []

    combined_stats: StatsMap | None = None
    combined_source_index: list[SourceIndex] = []
    dataset_steps: list[int] = []
    offset = 0

    for position, r in enumerate(results):
        _reject_unaddressable(r, position)
        stats = r["stats"]
        source_index = r["source_index"]
        if not source_index:
            # A result with no rows has no values to concatenate and no name set worth
            # comparing. An empty split — a filter or a selection that matched nothing —
            # must not read as "computed with different flags", which is what comparing its
            # empty name set against a populated one would say.
            #
            # Gated on the rows rather than on the columns, which are not the same question
            # since a `stats` mapping may name no view that produces a column: such a result
            # carries rows and no columns, and skipping it would leave every array shorter
            # than `combined_source_index` with nothing to say so. It reaches the mismatch
            # check below instead, where a genuine disagreement is what it is.
            #
            # Skipped only for the statistics; it still contributes a `dataset_steps`
            # boundary below, so the boundaries stay one-per-result and a later dataset's
            # index is not silently attributed to this one.
            pass
        elif combined_stats is None:
            combined_stats = dict(stats)
        else:
            _reject_stat_name_mismatch(combined_stats, stats, position)
            combined_stats = {k: np.concatenate([combined_stats[k], stats[k]]) for k in combined_stats}

        # `_replace` rather than a field-by-field rebuild: only `item` moves, and naming the
        # two that do not is what makes a future field silently dropped here.
        combined_source_index.extend(s._replace(item=s.item + offset) for s in source_index)
        offset += len(source_index)
        dataset_steps.append(offset)

    # `None` only where every result was empty, which is an empty result rather than an
    # absent one — the caller asked for a combination and gets the shape of one.
    return combined_stats if combined_stats is not None else {}, combined_source_index, dataset_steps
