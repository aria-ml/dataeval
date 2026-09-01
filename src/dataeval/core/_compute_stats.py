__all__ = []

import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence, Sized
from dataclasses import dataclass
from enum import Flag
from functools import partial, reduce
from itertools import zip_longest
from operator import or_
from typing import Any, TypedDict, cast, get_args

import numpy as np
from numpy.typing import NDArray

# Import calculators to trigger auto-registration
import dataeval.core._calculators._register  # noqa: F401
from dataeval._log import get_logger
from dataeval.config import get_max_processes
from dataeval.core._calculators._base import Calculator, ViewKind
from dataeval.core._calculators._cache import CalculatorCache
from dataeval.core._calculators._registry import CalculatorRegistry
from dataeval.data import unzip_dataset
from dataeval.flags import ImageStats
from dataeval.protocols import ArrayLike, Dataset, ObjectDetectionTarget, ProgressCallback
from dataeval.types import FactorLevel, SourceIndex, StatsMap
from dataeval.utils._internal import PoolWrapper
from dataeval.utils.preprocessing import (
    _UNKNOWN_RANGE,
    BoundingBox,
    BoxLike,
    ChannelGroup,
    ChannelGroupLike,
    ValueRange,
    _validate_declared_range,
    boxes_to_mask,
    get_value_range,
    normalize_image_shape,
    to_bounding_box,
    to_channel_group,
)

_logger = get_logger(__name__)

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


def _flag_list(flags: Flag) -> list[str]:
    """List the single-bit members of `flags`, leaving out the composite aliases.

    Iterating a Flag class yields the composites declared in its body as well — on Python
    3.10 in particular — so a group like ``VISUAL`` would list itself beside each of the
    statistics it stands for.
    """
    return [f.name for f in type(flags) if f in flags and f.name and f.value and (f.value & (f.value - 1)) == 0]


def _flag_names(flags: Flag) -> str:
    """Render :func:`_flag_list` for a message."""
    return ", ".join(_flag_list(flags))


def _calculators_for_view(
    calculators: Sequence[tuple[type[Calculator[Any]], Flag]],
    view: ViewKind,
) -> list[tuple[type[Calculator[Any]], Flag]]:
    """Narrow a calculator list to the statistics that are defined over one view.

    Which those are is each *statistic's* own answer, declared beside its handler, rather
    than a list of flags maintained here: a new statistic would otherwise have to be
    remembered in a module that knows nothing else about it, and would silently default to
    being computed over every view. Narrows the flags rather than dropping whole
    calculators, because one calculator's statistics need not agree — dimension geometry
    is band-invariant while the bit depth beside it is not.
    """
    narrowed: list[tuple[type[Calculator[Any]], Flag]] = []
    for calculator_cls, flags in calculators:
        kept = calculator_cls.flags_for_view(flags, view)
        if kept:
            narrowed.append((calculator_cls, kept))
    return narrowed


def _band_view(datum: NDArray[Any], indices: tuple[int, ...]) -> NDArray[Any]:
    """Narrow the datum to a group's bands, or stand in an all-NaN slice where it cannot be.

    A group the datum cannot fully supply is *substituted*, never skipped. Skipping means
    the calculators never run, so the column is never produced for that datum — and the
    aggregation appends each datum's values to one array per name, so a name missing from
    one datum silently shortens its array and misaligns it against the source index.
    ``_pad_missing_stats`` cannot repair it either, since it learns the name set from one
    datum's own results. Handing the calculators NaN pixels instead produces the right
    column, at the right length, holding the right answer: absent.

    All-or-nothing rather than reduced over whichever bands are present. A group spanning
    bands 2-5 would otherwise mean "bands 2-3" on a 4-band image and "bands 2-5" on an
    8-band one, under one column name, silently incomparable to everything downstream.
    """
    if datum.ndim < 2:
        # Non-spatial data has no band axis, so no group is satisfiable.
        return _absent_band(len(indices), (1, 1))
    image = normalize_image_shape(datum)
    if max(indices) < image.shape[0]:
        return image[list(indices)]
    return _absent_band(len(indices), image.shape[-2:])


def _band_count(datum: NDArray[Any]) -> int:
    """How many bands `datum` carries, counted the way `_band_view` counts them.

    Zero for non-spatial data, which has no band axis and so can satisfy no group. Read
    once per datum and compared against each group's highest index, rather than asking
    `_band_view` per group, so the normalization runs once however many groups were named.
    """
    return 0 if datum.ndim < 2 else normalize_image_shape(datum).shape[0]


def _absent_band(count: int, hw: tuple[int, ...]) -> NDArray[np.float64]:
    """Return the all-NaN stand-in for a band group the datum cannot supply.

    float64 for the same reason `CalculatorCache.nan_like` is: the calculators' reductions
    and `edge_filter` behave differently on a narrower or object dtype.
    """
    return np.full((count, *hw), np.nan, dtype=np.float64)


class StatsResult(TypedDict):
    """
    Type definition for calculation output.

    Attributes
    ----------
    source_index : Sequence[SourceIndex]
        Sequence of SourceIndex objects with image/box info.
    object_count : Sequence[int]
        Sequence of object counts per image.
    invalid_box_count : Sequence[int]
        Sequence of invalid box counts per image.
    image_count : int
        Total number of images processed.
    stats : Mapping[str, NDArray[Any]]
        Mapping of statistic names to NumPy arrays of computed values.
        Keys are the names of statistics requested (e.g., 'mean', 'std', 'brightness').
        Values are NumPy arrays where each element corresponds to a source_index entry.
        String values (e.g., hashes) are stored as object dtype arrays.
    """

    source_index: Sequence[SourceIndex]
    object_count: Sequence[int]
    invalid_box_count: Sequence[int]
    image_count: int
    stats: Mapping[str, NDArray[Any]]


@dataclass
class DatumResult:
    """Result from processing a single image/box combination."""

    source_indices: list[SourceIndex]
    stats: dict[str, list[Any]]


@dataclass(frozen=True)
class BandPlan:
    """Everything about one named band group that is the same for every datum.

    One object per group rather than one mapping per fact. The four mappings this replaces
    were keyed alike but not keyed the *same*: a group dropped as barren left two of them
    and stayed in the others, and the masked calculators covered only the groups a mask
    does not destroy — so a reader had to reconstruct which key sets coincided before
    trusting ``x[name]``, and the worker tested membership of a second mapping to find out.
    Here those relations are fields, and `_compute_batch` walks one mapping and asks
    ``band.background_calculators``. That is the argument :class:`BatchPlan` already makes
    for itself, applied to the level below it.

    `indices` and `value_range` are lifted out of
    :class:`~dataeval.utils.preprocessing.ChannelGroup` rather than holding one: the worker
    wants the bands and the interval, and nothing it does needs the validation that class
    exists to perform.
    """

    indices: tuple[int, ...]
    value_range: tuple[float, float] | None
    calculators: Sequence[tuple[type[Any], Flag]]
    background_calculators: Sequence[tuple[type[Any], Flag]]
    unmeasurable_names: str


@dataclass(frozen=True)
class BatchPlan:
    """Everything about a `compute_stats` call that is the same for every datum.

    Each field is a pure function of the call's arguments, resolved once before any image
    is read. Threading them individually left `_compute_batch` with thirteen parameters,
    four of them calculator lists that must agree with the `ViewKind` passed beside them —
    an invariant that could only be documented, never enforced. Resolving them together
    makes that agreement a property of how the plan is built, in one place, and adding a
    fourth view costs one field rather than two more positional arguments.

    No field carries a default. There is one construction site and it states all of them,
    so a default would only ever hide a field someone forgot to thread — and a mutable one
    is rejected outright by `dataclasses` on Python 3.11, which is the single version where
    ``mappingproxy`` is both unhashable and caught by that check.
    """

    calculators: Sequence[tuple[type[Any], Flag]]
    per_image: bool
    per_target: bool
    normalize_pixel_values: bool
    per_background: bool
    background_calculators: Sequence[tuple[type[Any], Flag]]
    declared_range: tuple[float, float] | None
    band_plans: Mapping[str, BandPlan]
    wide_band_names: str
    unmeasurable_names: str


@dataclass
class DatumBatchResult:
    """Output from processing multiple images."""

    results: list[DatumResult]
    object_count: int
    invalid_box_count: int
    warnings_list: list[str]


def _collect_calculator_stats(
    calculators: Iterable[tuple[type[Any], Flag]],
    datum: NDArray[Any],
    box: BoundingBox | None,
    normalize_pixel_values: bool = False,
    exclude: NDArray[np.bool_] | None = None,
    prefix: str = "",
    value_range: ValueRange | None = None,
    view: ViewKind = ViewKind.WHOLE,
    bands: tuple[int, ...] | None = None,
) -> tuple[list[dict[str, list[Any]]], dict[str, Any], list[str]]:
    """
    Collect stats from all calculators.

    Parameters
    ----------
    exclude : NDArray[np.bool_] or None, default None
        Pixel mask of regions to leave out of every statistic, passed through to the
        calculator cache, which NaNs them out of the region being reduced over.
    prefix : str, default ""
        Prepended to every stat name produced here, which is how a masked pass is kept
        distinguishable from the unmasked one sharing its row.
    value_range : ValueRange or None, default None
        The datum's value range, already established. Every row of one datum shares it, so
        it is read once per datum and handed down rather than rediscovered per row —
        establishing it costs a scan of the whole datum, which a per-row cache would
        repeat once per box.
    view : ViewKind, default ViewKind.WHOLE
        Which view of the datum `datum` is, so that statistics undefined over it are
        skipped. `calculators` is expected to have been narrowed to the same view already;
        passing it here keeps the two from drifting apart.

    Returns
    -------
    tuple[list[dict[str, list[Any]]], dict[str, Any], list[str]]
        A tuple of (stats_list, empty_values_map, warnings) where:
        - stats_list: List of computed stats from each calculator
        - empty_values_map: Mapping of stat names to their empty values (defaults to np.nan)
        - warnings: List of warning messages from calculators
    """
    stats_list = []
    empty_values_map: dict[str, Any] = {}
    warnings: list[str] = []
    processor = CalculatorCache(
        datum,
        box,
        normalize_pixel_values=normalize_pixel_values,
        exclude=exclude,
        value_range=value_range,
        bands=bands,
    )
    for calculator_cls, flags in calculators:
        calculator = calculator_cls(datum, processor)
        computed = calculator.compute(flags, view)
        stats_list.append({f"{prefix}{name}": values for name, values in computed.items()})
        # Collect empty values from this calculator
        empty_values_map.update({f"{prefix}{name}": value for name, value in calculator.get_empty_values().items()})
        # Collect warnings from this calculator
        if hasattr(calculator, "warnings"):
            warnings.extend(calculator.warnings)
        del calculator
    return stats_list, empty_values_map, warnings


def _band_ranges(
    datum: NDArray[Any],
    band_plans: Mapping[str, BandPlan],
    declared_range: tuple[float, float] | None,
) -> dict[str, ValueRange]:
    """Return the interval each named group is measured against, read off the whole datum.

    Established per datum rather than per row, for the reason the whole-datum `value_range`
    is: the answer is the same for every box the datum carries, and finding it costs a scan.
    Anchored on the datum rather than on a row's crop so that a box, its background and the
    whole image all land on one scale.

    Established per *group* rather than inherited from the datum, because bands of one cube
    are different measurements. An RGB+NIR image stored as ``uint16`` with the visible bands
    in 0-255 has a whole-datum range of 65535, which would divide every visible pixel by 257
    times too much and bin the lot into one histogram bucket.

    Only the range is kept. The slice it is read from is a full-resolution copy of the
    selected bands, and no row wants that — each takes its own box's worth through
    `CalculatorCache`, which narrows the band axis after cropping.
    """
    return {
        name: get_value_range(_band_view(datum, band.indices), declared=band.value_range or declared_range)
        for name, band in band_plans.items()
    }


def _collect_band_stats(
    calculators: Sequence[tuple[type[Calculator[Any]], Flag]],
    datum: NDArray[Any],
    bands: tuple[int, ...],
    band_range: ValueRange,
    box: BoundingBox | None,
    name: str,
    *,
    normalize_pixel_values: bool,
    exclude: NDArray[np.bool_] | None,
    prefix: str,
    view: ViewKind = ViewKind.BAND,
) -> tuple[list[dict[str, list[Any]]], dict[str, Any], list[str]]:
    """Run one named band group as its own view of the datum.

    The same calculators over the same region, reading a slice of the channel axis
    instead of all of it, with the group's name prefixed onto every column — the band
    counterpart of what `per_background` does to the spatial axes. Composes with it:
    a group run over a masked region yields ``background_nir_brightness``, and `view`
    then carries both so a statistic undefined over either is still skipped.
    """
    return _collect_calculator_stats(
        calculators,
        datum,
        box,
        normalize_pixel_values=normalize_pixel_values,
        exclude=exclude,
        prefix=f"{prefix}{name}_",
        value_range=band_range,
        view=view,
        bands=bands,
    )


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


def _get_items(
    boxes: list[BoundingBox] | None,
    per_image: bool,
    per_target: bool,
    per_background: bool = False,
) -> list[tuple[int | None, BoundingBox | None, bool, bool]]:
    """Determine what to process based on the per_image, per_target and per_background flags.

    Returns
    -------
    list[tuple[int | None, BoundingBox | None, bool, bool]]
        One entry per row to emit, as ``(target_index, box, unmasked, background)``.
        The last two say which passes that row carries: the whole-region statistics,
        the background ones, or — when both `per_image` and `per_background` are set —
        both, since the two share the item's single row.
    """
    process_items: list[tuple[int | None, BoundingBox | None, bool, bool]] = []

    # The item-level row exists if either pass wants it. Background rides on that row
    # rather than adding one, so requesting it without per_image yields a row carrying
    # background statistics alone.
    if per_image or per_background:
        process_items.append((None, None, per_image, per_background))

    # Boxes are only processed when there are boxes to process.
    if per_target and boxes:
        process_items.extend((i_b, box, True, False) for i_b, box in enumerate(boxes))

    return process_items


def _pad_missing_stats(results: list["DatumResult"], empty_values_map: dict[str, Any]) -> None:
    """Give every row in a datum an entry for every stat name any of its rows carries.

    Rows within one datum no longer all carry the same statistics: background values
    exist on the item-level row and nowhere else. The aggregation downstream appends
    each row's values to one array per name, so a name missing from a row would shorten
    that name's array and silently misalign it against the source index. Filling the
    gaps with each stat's empty value keeps every array one-to-one with the rows.
    """
    names = {name for result in results for name in result.stats}
    for result in results:
        entries = len(result.source_indices)
        for name in names:
            if name not in result.stats:
                result.stats[name] = [empty_values_map.get(name, np.nan)] * entries


def _compute_batch(  # noqa: C901
    args: tuple[int, NDArray[Any], list[BoundingBox] | None],
    plan: BatchPlan,
) -> DatumBatchResult:
    # Bound to locals rather than read through `plan.` at each use: this body runs once per
    # datum in a worker process, and the names below are what the rest of it is written in.
    calculators = plan.calculators
    per_image = plan.per_image
    per_target = plan.per_target
    normalize_pixel_values = plan.normalize_pixel_values
    per_background = plan.per_background
    background_calculators = plan.background_calculators
    declared_range = plan.declared_range
    band_plans = plan.band_plans
    wide_band_names = plan.wide_band_names
    unmeasurable_names = plan.unmeasurable_names

    i, datum, boxes = args
    results: list[DatumResult] = []
    box_count = 0
    invalid_box_count = 0
    warnings_list: list[str] = []
    datum_empty_values: dict[str, Any] = {}

    # Determine the number of channels from the datum shape
    num_channels = datum.shape[-3] if len(datum.shape) >= 3 else 1

    # Determine what to process based on the per_image, per_target and per_background flags
    items = _get_items(boxes, per_image, per_target, per_background)

    # Read once per datum rather than once per row: it is a property of the datum, every
    # view of it shares the value, and finding it costs a scan of the whole array. Not read
    # at all where nothing consumes it: a `stats` mapping with no ``None`` entry leaves the
    # unnamed view measuring nothing, and every consumer below is downstream of
    # `calculators` — the masked pass narrows that list, and `unmeasurable_names` names a
    # subset of the flags it was built from. A band group is anchored on its own bands and
    # takes `band_ranges` instead, so it is unaffected either way.
    value_range = get_value_range(datum, declared=declared_range) if calculators else _UNKNOWN_RANGE

    # The unnamed view means *the image as a picture*, which is only defined for mono or
    # RGB. Averaged over RGB+NIR a brightness is not a brightness, and the grayscale
    # conversion a hash runs reaches a CMYK-versus-RGBA guess at four channels. Said
    # rather than enforced: existing callers who tolerate today's answer keep it, and a
    # cap would take the dimension statistics — well defined at any band count — with it.
    # Not an error: the caller may have declared ranges for the band groups they care
    # about and never asked for this view at all, and one unmeasurable datum should not
    # end a run over a hundred thousand. Said rather than silent, because an all-NaN
    # column is otherwise easy to miss.
    if unmeasurable_names and not value_range.is_known:
        warnings_list.append(
            f"{i}: no value range could be established, so {unmeasurable_names} are NaN. Float data spanning "
            f"more than [0, 255], and any data holding negative values, carries no encoding to decode "
            f"— pass value_range=(low, high) to state the interval these values occupy."
        )

    if wide_band_names and num_channels > 3:
        warnings_list.append(
            f"{i}: {num_channels} bands measured as a single picture for {wide_band_names}; these "
            f"describe visible imagery and have no meaning averaged across a band this wide. Name band groups "
            f"with channels= to measure them separately."
        )

    # The foreground mask: every annotated box painted into one array, so overlapping
    # boxes count once. Built per datum rather than per row — the item-level row is the
    # only one that uses it, but building it here keeps it out of the row loop.
    # Non-spatial data takes no mask. `boxes_to_mask` reads `shape[-2:]`, and a background
    # is a region of an image plane — a 1-D datum has none to carve out. `CalculatorCache`
    # already documents `exclude` as ignored below three dimensions, so this only stops
    # the mask being built at all rather than changing what any statistic answers.
    exclude = boxes_to_mask(datum.shape, boxes or ()) if per_background and datum.ndim >= 2 else None
    # `.all()` is vacuously True for a datum with no pixels, which is an absence of
    # background rather than a full one; `.mean()` on it is NaN and warns. Neither is
    # worth saying, so an empty datum takes neither path.
    if exclude is not None and exclude.size and exclude.all():
        warnings_list.append(f"{i}: Boxes cover the entire datum, leaving no background to measure")

    # Read once per datum for the same reason `value_range` is: a group's interval is a
    # property of the datum, and every row below is measured against the same one.
    band_ranges = _band_ranges(datum, band_plans, declared_range) if band_plans else {}

    # Two unrelated reasons a group's columns come back NaN, told apart before either is
    # reported. A datum that cannot supply the bands a group names is measured over the
    # all-NaN stand-in `_band_view` substitutes, and a range read off that stand-in is
    # unknown for a reason that has nothing to do with encoding — so answering it with the
    # interval advice below would name a remedy that silences the message and leaves every
    # column NaN just the same. Asked first, and asked whatever the range says: a declared
    # interval makes the range knowable without making the bands present.
    #
    # The per-group counterpart of the whole-datum warning above, and not covered by it:
    # each group is measured against its own interval, which `_band_ranges` reads off that
    # group's bands alone. One cube can carry a knowable range for its visible bands and
    # none for the reflectance band beside them, so the two are asked separately.
    bands_present = _band_count(datum) if band_plans else 0
    for group_name, band in band_plans.items():
        if max(band.indices) >= bands_present:
            warnings_list.append(
                f"{i}: channel group '{group_name}' names band {max(band.indices)} but the datum "
                f"carries {bands_present}, so every '{group_name}_' statistic is NaN for it. A group "
                f"is measured all-or-nothing rather than over the bands that are present, since one "
                f"column name has to mean one thing."
            )
        elif band.unmeasurable_names and not band_ranges[group_name].is_known:
            warnings_list.append(
                f"{i}: no value range could be established for channel group '{group_name}', so its "
                f"{band.unmeasurable_names} are NaN. Bands carrying no encoding to decode — a "
                f"reflectance, elevation or temperature band — need their interval stated: pass "
                f"channels={{'{group_name}': ChannelGroup(..., value_range=(low, high))}}."
            )

    # Process each item (full image and/or boxes)
    for i_b, box, unmasked, background in items:
        if box is not None:
            box_count += 1
            if not box.is_clippable():
                invalid_box_count += 1
                source = f"{i}[{i_b}]"
                warnings_list.append(f"{source}: Bounding box {box} for datum shape {datum.shape} is invalid")

        calculator_stats: list[dict[str, list[Any]]] = []
        empty_values_map: dict[str, Any] = {}
        calc_warnings: list[str] = []

        # Collect stats from all calculators
        if unmasked:
            stats, empties, warns = _collect_calculator_stats(
                calculators,
                datum,
                box,
                normalize_pixel_values=normalize_pixel_values,
                value_range=value_range,
            )
            calculator_stats.extend(stats)
            empty_values_map.update(empties)
            calc_warnings.extend(warns)

            # One further pass per named band group, landing as prefixed columns on this
            # same row. The unnamed view above is computed alongside them unless a `stats`
            # mapping deliberately withheld it: the band-invariant statistics need a pass to
            # be emitted from, and a caller adding channels= to a pipeline reading
            # `brightness` must not lose it.
            for name, band in band_plans.items():
                stats, empties, warns = _collect_band_stats(
                    band.calculators,
                    datum,
                    band.indices,
                    band_ranges[name],
                    box,
                    name,
                    normalize_pixel_values=normalize_pixel_values,
                    exclude=None,
                    prefix="",
                )
                calculator_stats.extend(stats)
                empty_values_map.update(empties)
                calc_warnings.extend(warns)

        # The same calculators over the same region with the foreground masked out. Run
        # into the same reconciliation as the unmasked pass, so both land on one row with
        # a single channel layout resolved across the two.
        if background and exclude is not None:
            stats, empties, warns = _collect_calculator_stats(
                background_calculators,
                datum,
                box,
                normalize_pixel_values=normalize_pixel_values,
                exclude=exclude,
                prefix=BACKGROUND_PREFIX,
                value_range=value_range,
                view=ViewKind.MASK,
            )
            calculator_stats.extend(stats)
            empty_values_map.update(empties)
            calc_warnings.extend(warns)
            # Emitted whatever was requested, because every other background value is
            # only as trustworthy as the share of the image it was measured over. A datum
            # with no pixels has no share to report, so it reports NaN rather than 0/0.
            uncovered = float(1.0 - exclude.mean()) if exclude.size else float("nan")
            calculator_stats.append({f"{BACKGROUND_PREFIX}fraction": [uncovered]})

            # Region first, then band: `background_nir_brightness` — is the unannotated
            # scene hot in near-infrared — is a real quantity, and the two views compose
            # rather than competing for one parameter.
            # A group may survive to the band pass and not to the masked one — a hash is
            # band-variant but not NaN-stable — which is a fact about the group, so it is
            # read off the group rather than by asking whether a second mapping holds it.
            for name, band in band_plans.items():
                if not band.background_calculators:
                    continue
                stats, empties, warns = _collect_band_stats(
                    band.background_calculators,
                    datum,
                    band.indices,
                    band_ranges[name],
                    box,
                    name,
                    normalize_pixel_values=normalize_pixel_values,
                    exclude=exclude,
                    prefix=BACKGROUND_PREFIX,
                    view=ViewKind.MASK | ViewKind.BAND,
                )
                calculator_stats.extend(stats)
                empty_values_map.update(empties)
                calc_warnings.extend(warns)

        # Thread calculator warnings with index context
        for w in calc_warnings:
            source = f"{i}" if box is None else f"{i}[{i_b}]"
            warnings_list.append(f"{source}: {w}")

        # Gather every calculator's output for this row onto one row
        reconciled_stats = _reconcile_stats(calculator_stats)
        datum_empty_values.update(empty_values_map)

        source = [SourceIndex(i, i_b if box is not None else None)]

        results.append(DatumResult(source_indices=source, stats=reconciled_stats))

    # Background statistics exist on the item-level row alone, so the box rows need
    # placeholders for them before the rows are flattened into per-name arrays.
    if per_background:
        _pad_missing_stats(results, datum_empty_values)

    return DatumBatchResult(results, box_count, invalid_box_count, warnings_list)


def _produced_stat_names(calculators: Sequence[tuple[type[Calculator[Any]], Flag]]) -> set[str]:
    """Every column name the requested flags will produce, known before any datum is read."""
    names: set[str] = set()
    for calculator_cls, flags in calculators:
        names |= calculator_cls.stat_names(flags)
    return names


def _resolve_channel_groups(
    channels: Mapping[str, ChannelGroupLike],
    calculators: Sequence[tuple[type[Calculator[Any]], Flag]],
) -> dict[str, ChannelGroup]:
    """Validate a channel mapping at the call, where the mistake is.

    Every input to this check is known before any image is read, so a bad group name fails
    here rather than as a confusing rename several layers downstream: a group called
    ``instance`` would produce ``instance_brightness``, indistinguishable from the
    level-qualified column :class:`~dataeval.Metadata` produces for a per-target
    statistic, and would be silently renamed with an ``_added`` suffix instead of
    reported against its cause.
    """
    produced = _produced_stat_names(calculators)
    reserved = {BACKGROUND_PREFIX.rstrip("_"), *get_args(FactorLevel)}
    groups: dict[str, ChannelGroup] = {}

    for name, group in channels.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"channel group names must be valid identifiers; got {name!r}.")
        if name in reserved or name.startswith(BACKGROUND_PREFIX):
            # The prefix and not just the bare word: a group named ``background_x`` would
            # produce ``background_x_mean``, which is also what group ``x`` produces for
            # the masked region — the same column written twice on one row, the second
            # silently overwriting the first.
            raise ValueError(
                f"channel group name {name!r} is reserved. Statistics are named "
                f"'<group>_<statistic>', and this one would be indistinguishable from a "
                f"region prefix or a metadata level qualifier. Reserved: {sorted(reserved)}, "
                f"and any name beginning with '{BACKGROUND_PREFIX}'."
            )
        collisions = sorted(f"{name}_{stat}" for stat in produced if f"{name}_{stat}" in produced)
        if collisions:
            raise ValueError(
                f"channel group name {name!r} would produce {collisions}, which already "
                f"name statistics you requested. Choose another group name."
            )
        groups[name] = to_channel_group(group)

    return groups


def _view_flags(calculators: Sequence[tuple[type[Calculator[Any]], Flag]]) -> Flag:
    """Total the flags a narrowed calculator list will actually produce a column for."""
    return reduce(or_, (flags for _, flags in calculators), ImageStats(0))


def _unmeasurable_flags(flags: Flag, declared: tuple[float, float] | None, normalize_pixel_values: bool) -> Flag:
    """Statistics among `flags` that need an interval, and so go NaN when none can be established.

    The histogram and its entropy always do; so does the whole visual family, which resolves
    the display range, and `depth`, which reports the encoding it decoded. The remaining
    pixel statistics only when they are being normalized against it. A declared range is
    always known, so nothing can go unmeasurable for want of one.

    Asked once for the unnamed view and once per band group, since each is measured against
    its own interval and a cube can carry a decodable range on one group and none on the
    next — which is why `declared` is a parameter rather than read from the call.

    `missing` is the exception in both halves: it measures the *presence* of data rather
    than the data, reads off the raw view, and so still answers when nothing could be
    measured. Naming it would promise a NaN column that never arrives.
    """
    if declared is not None:
        return type(flags)(0)
    needs_range = flags & (
        ImageStats.PIXEL_HISTOGRAM | ImageStats.PIXEL_ENTROPY | ImageStats.VISUAL | ImageStats.DIMENSION_DEPTH
    )
    if normalize_pixel_values:
        needs_range |= flags & ImageStats.PIXEL
    return needs_range & ~ImageStats.PIXEL_MISSING


def _names(keys: Iterable[Any]) -> str:
    """Render a set of mapping keys for a message, in a stable order."""
    return ", ".join(sorted(map(repr, keys))) or "none"


def _check_stats_mapping(stats: Mapping[str | None, Flag], channels: Mapping[str, ChannelGroupLike] | None) -> None:
    """Check that a stats mapping names exactly the views the call defines.

    Both directions, because neither mistake is one the run can recover from and both are
    silent if allowed through: a group with no entry would produce no columns, and an entry
    naming no group would compute nothing, each looking exactly like a statistic that came
    out empty. Every input is known before an image is read, so the typo fails at the call.

    Both lists are always stated rather than only the non-empty one — the mistake is
    usually a name spelled two ways, and seeing which side each spelling landed on is the
    whole of the diagnosis.

    Keys are rendered with ``repr`` rather than compared for type first: a key that is
    neither a string nor ``None`` names no channel group either, and reporting it that way
    costs no separate check.
    """
    named = {key for key in stats if key is not None}
    groups = set(channels or {})
    if named == groups:
        return
    raise ValueError(
        f"a stats mapping states the statistics for each named view, so its keys must be exactly the "
        f"channel group names, plus None for the whole image. Channel groups with no entry: "
        f"{_names(groups - named)}; keys naming no channel group: {_names(named - groups)}."
    )


def _resolve_stat_flags(
    stats: StatsRequest,
    channels: Mapping[str, ChannelGroupLike] | None,
) -> tuple[Flag, dict[str, Flag]]:
    """Split `stats` into the flags for the unnamed view and the flags for each band group.

    A single ``Flag`` is every view's request, which is what `channels` has always meant:
    each group measures the statistics the image does, restricted to its own bands.

    A mapping states them separately, keyed by the prefix the columns carry — ``None`` for
    the unprefixed ones, since that is already what an unqualified position means here
    (:class:`~dataeval.types.SourceIndex` spells "the item itself, not a sub-part" the same
    way). It is read as a *total* statement: a view with no entry is not measured, so
    ``{"rgb": ImageStats.PIXEL}`` returns ``rgb_*`` columns and nothing else.

    Deliberately not defaulted to anything derived from the group entries — not their
    union, and not the statistics no group can answer. Either would make a caller's column
    set depend on a per-statistic table they cannot see from the call site. The cost is
    that a mapping stating only groups silently drops geometry, which `compute_stats`
    warns about where it can see that nothing else asks for it.

    Its named keys must be exactly `channels`' keys — see `_check_stats_mapping`. Two
    arguments listing the same names is already one more than ideal; letting them drift
    would turn a typo into a silently missing set of columns rather than an error.

    Returns
    -------
    tuple[Flag, dict[str, Flag]]
        The unnamed view's flags, and one entry per channel group.
    """
    if not isinstance(stats, Mapping):
        return stats, dict.fromkeys(channels or {}, stats)
    # Read as the wider key type, which both members of the union support: nothing here
    # writes to the mapping, and looking up `None` in one that cannot hold it simply misses.
    mapping = cast(Mapping[str | None, Flag], stats)
    _check_stats_mapping(mapping, channels)
    group_flags = {name: flags for name, flags in mapping.items() if name is not None}
    return mapping.get(None, ImageStats(0)), group_flags


def _enumerate_datum(
    images: Iterable[ArrayLike],
    boxes: Iterable[Iterable[BoxLike] | None] | None,
) -> Iterator[tuple[int, NDArray[Any], list[BoundingBox] | None]]:
    if boxes is None:
        for i, image in enumerate(images):
            yield i, np.asarray(image), None
    else:
        for i, (image, box) in enumerate(zip_longest(images, boxes, fillvalue=None)):
            if image is None:
                continue
            np_image = np.asarray(image)
            bboxes = [to_bounding_box(b, image_shape=np_image.shape) for b in box or ()]
            yield i, np_image, bboxes


def _sort(
    source_indices: list[SourceIndex],
    aggregated_stats: dict[str, list[Any]],
) -> tuple[list[SourceIndex], dict[str, NDArray[Any]]]:
    """Sort results by (item_index, box_index) with None < 0 and convert to numpy arrays.

    No level term: every address ``compute_stats`` emits leaves the level unstated, since
    it measures a datum and a box without knowing whether the datum is an image or a
    frame. Two rows therefore never tie on ``(item, key)`` here, and a level a caller
    states afterwards is theirs to order by — ``SourceIndexRows.parse`` groups by level
    before it orders within one, so an address that arrives from anywhere is ranked
    against the rows of its own level rather than against this ordering.
    """
    sort_indices = sorted(
        range(len(source_indices)),
        key=lambda i: (
            source_indices[i].item,
            -1 if source_indices[i].key is None else source_indices[i].key,
        ),
    )

    sorted_source_indices: list[SourceIndex] = [source_indices[i] for i in sort_indices]
    sorted_aggregated_stats: dict[str, NDArray[Any]] = {}
    for stat_name, stat_values in aggregated_stats.items():
        # Sort the values and convert to numpy array
        sorted_values = [stat_values[i] for i in sort_indices]
        np_array = np.array(sorted_values)
        # If the values are floats, convert to dtype float32 to save memory
        # while avoiding float16 overflow (max ~65504)
        if np.issubdtype(np_array.dtype, np.floating):
            np_array = np_array.astype(np.float32)
        sorted_aggregated_stats[stat_name] = np_array

    return sorted_source_indices, sorted_aggregated_stats


def _aggregate_batch(
    result: DatumBatchResult,
    source_indices: list[SourceIndex],
    aggregated_stats: dict[str, list[Any]],
    object_count: dict[int, int],
    invalid_box_count: dict[int, int],
    warning_list: list[str],
) -> None:
    """Extract and aggregate results from a single StatsProcessorOutput."""
    for r in result.results:
        source_indices.extend(r.source_indices)
        for stat_name, stat_values in r.stats.items():
            aggregated_stats.setdefault(stat_name, []).extend(stat_values)

    if result.results and result.results[0].source_indices:
        img_idx = result.results[0].source_indices[0].item
        object_count[img_idx] = result.object_count
        invalid_box_count[img_idx] = result.invalid_box_count

    warning_list.extend(result.warnings_list)


_UNSET = object()


def compute_stats(  # noqa: C901
    data: Iterable[ArrayLike] | Dataset[ArrayLike] | Dataset[tuple[ArrayLike, Any, Any]],
    *,
    boxes: Iterable[Iterable[BoxLike] | None] | None = None,
    stats: StatsRequest = ImageStats.ALL,
    per_image: bool = True,
    per_target: bool = True,
    per_background: bool = False,
    channels: Mapping[str, ChannelGroupLike] | None = None,
    normalize_pixel_values: bool = _UNSET,  # type: ignore
    value_range: tuple[float, float] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> StatsResult:
    """
    Compute specified statistics on a set of images, optionally within bounding boxes.

    Parameters
    ----------
    data : Iterable[ArrayLike] | Dataset[ArrayLike] | Dataset[tuple[ArrayLike, Any, Any]]
        An iterable of images or a Dataset to compute statistics on.
    boxes : Iterable[Iterable[BoxLike] | None] | None
        Optional bounding boxes for each image. If None, defers to the data provided.
    stats : ImageStats or Mapping[str | None, ImageStats], default ImageStats.ALL
        Flags indicating which statistics to compute. Can combine multiple flags
        using bitwise OR (|). Dependencies are resolved automatically for calculation,
        but intermediate/dependency statistics are not included in the output by
        default unless explicitly requested.

        A single flag set is every view's request: each group named in `channels` measures
        the same statistics the image does, restricted to its own bands.

        A **mapping** asks a different question of each view, keyed by the prefix its
        columns carry — a channel group's name, or ``None`` for the unprefixed whole-image
        ones. Bands of one cube are different measurements, and rarely deserve the same
        ones: a hash of the visible bands identifies a duplicate frame where a hash of the
        whole cube does not, while a thermal band wants its distribution and nothing else::

            stats = {None: ImageStats.DIMENSION, "rgb": ImageStats.HASH, "ir": ImageStats.PIXEL}
            channels = {"rgb": [0, 1, 2], "ir": 3}

        The mapping is read as a *complete* statement: a view with no entry is not
        measured. ``{"rgb": ImageStats.PIXEL}`` therefore returns ``rgb_*`` columns and
        nothing else — no ``width``, no ``mean``. Nothing is inferred for the missing
        ``None`` entry, deliberately: any default derived from the group entries would make
        the column set depend on a per-statistic table that is not visible from the call.
        A warning names any statistic the mapping asks of a group that cannot vary with a
        band subset while no whole-image entry asks for it, since that one is computed
        nowhere.

        Its named keys must be exactly `channels`' keys, checked in both directions; a name
        in one and not the other is an error rather than a silently missing column set.

        .. versionadded:: 1.2
            The mapping form.
    per_image : bool, default True
        If True, compute statistics for entire images. When boxes are provided
        and per_image=True, statistics are computed for both the full image and
        each box (if per_target=True).
    per_target : bool, default True
        If True and boxes are provided, compute statistics for each bounding box.
        Has no effect when boxes is None. At least one of per_image, per_target or
        per_background must be True.
    per_background : bool, default False
        If True and boxes are provided, additionally compute statistics over each
        image's background — every pixel the image's boxes do not cover — describing
        the scene an item was captured in rather than the things annotated within it.

        Background values are returned alongside the whole-image ones, on the same
        rows, under names prefixed with ``background_`` (``background_brightness``).
        They are therefore per-image values, and adding them to
        :class:`~dataeval.Metadata` places them at the media-unit level like any other
        per-image factor. ``background_fraction`` — the share of the image left
        unmasked — is always among them, and should be read before the rest: a
        background statistic measured over a few percent of an image is noise wearing
        a measurement's clothes.

        Only :attr:`~dataeval.flags.ImageStats.PIXEL` and
        :attr:`~dataeval.flags.ImageStats.VISUAL` statistics are computed for the
        background; any hash or dimension statistics in `stats` are computed for the
        image and its boxes as usual and skipped for the background, which has no
        meaningful hash and no geometry of its own.

        Boxes are rounded outwards and unioned into the mask, so the background
        excludes slightly more than the annotations strictly cover — a retained pixel
        is background with high confidence. Where the boxes cover an image entirely,
        every background statistic for it is NaN.

        An image with nothing annotated has all of itself as background, so its
        ``background_fraction`` is 1.0 and its background statistics equal its
        whole-image ones. That holds whether the image is an unannotated member of an
        object-detection dataset or the whole dataset carries no boxes at all — the
        columns produced depend on this argument, not on the data, which is what lets
        results from two such datasets be combined.

        .. versionadded:: 1.1
    channels : Mapping[str, ChannelGroupLike] or None, default None
        Named groups of bands to measure separately, alongside the whole image.

        Each group becomes a set of columns named ``<group>_<statistic>`` on the same rows
        the unprefixed statistics occupy — the band-axis counterpart of `per_background`,
        which does the same thing to the spatial axes. The two compose, giving
        ``background_nir_brightness``.

        A group is measured **jointly**: ``{"rgb": [0, 1, 2]}`` reduces over the three
        visible bands together, which is the ordinary all-channel behavior restricted to a
        subset. That is what scales to hyperspectral data — a 224-band cube is asked about
        as a handful of band groups rather than 224 columns::

            channels = {"visible": range(0, 30), "nir": range(30, 70), "swir": range(100, 150)}

        Values may be an index, a sequence of indices, a range, or a
        :class:`~dataeval.utils.preprocessing.ChannelGroup` where the group needs its own
        `value_range`. Group names must not collide with a statistic name, with
        ``background``, or with a :data:`~dataeval.types.FactorLevel`.

        Every group measures the statistics `stats` asks for. Pass `stats` as a mapping
        keyed by these same names to ask a different question of each.

        The unprefixed statistics are computed as well, so adding `channels` to an existing
        call never removes a column it already returned — passing `stats` as a mapping is
        the one way to drop them. Where an image cannot supply every band a group names,
        that group's statistics are NaN for it rather than reduced over the bands present —
        one column name means one thing, and a datum missing bands is a defect that should
        read as absent.

        Has no effect on statistics that describe geometry rather than values: a band
        subset does not move a bounding box, so ``rgb_width`` is not produced.

        See :doc:`/notebooks/h2_measure_channel_groups` for a worked example.

        .. versionadded:: 1.1
    normalize_pixel_values : bool, default False
        If True, :attr:`~dataeval.flags.ImageStats.PIXEL` statistics are computed on
        values normalized to [0, 1] against each image's range rather than on the raw
        values. This makes a *distribution* comparable across images stored at different
        bit depths — an 8-bit and a 16-bit copy of one picture otherwise report means
        differing by a factor of 257.

        Affects the pixel family only, and within it only `PIXEL_MEAN`, `PIXEL_STD` and
        `PIXEL_VAR`; the rest are already scale-free.
        :attr:`~dataeval.flags.ImageStats.VISUAL` statistics ignore it entirely — they
        always report against the display range, so they are comparable across encodings
        either way. Prefer `VISUAL_BRIGHTNESS` over a normalized `PIXEL_MEAN` when the
        question is how an image looks rather than how its values are distributed.

        .. deprecated:: 1.0
            The default changed to False in v1.1. Pass explicitly to silence
            the deprecation warning. This warning will be removed in v1.2.0.
    value_range : tuple[float, float] or None, default None
        The interval every image's values should be measured against, as ``(low, high)``.

        Leave as None for ordinary imagery. An integer image's range is *decoded* from
        its encoding, and the two conventional float spellings of an image — normalized
        to ``[0, 1]``, or 8-bit values held in a float array — are recognized, so
        ``uint8``, ``uint16`` and ``ToTensor``-style data all need nothing here.

        Declare it for data whose dynamic range is a property of the sensor rather than
        of a file format: elevation below sea level, mean-centred reflectance,
        temperature in Celsius, a 16-bit band holding physical units. Such data carries
        no encoding to decode, so :attr:`~dataeval.flags.ImageStats.DIMENSION_DEPTH`
        reports NaN for it, as does any statistic that needs an interval —
        :attr:`~dataeval.flags.ImageStats.PIXEL_HISTOGRAM`,
        :attr:`~dataeval.flags.ImageStats.PIXEL_ENTROPY`, and under
        `normalize_pixel_values` the rest of the pixel family — rather than deriving one
        from an arbitrary maximum. A warning names this argument when it happens.

        A statistic that does not need an interval is unaffected: an unnormalized mean of
        physical values is a perfectly good mean.

        A declaration applies to every image in `data` and implies no bit depth.

        .. versionadded:: 1.1
    progress_callback : ProgressCallback or None, default None
        Callback to report progress during calculation. Called after each image is processed
        with the current image count and total number of images (if known).

    Returns
    -------
    StatsResult
        Mapping containing computed statistics and metadata:

        - source_index: Sequence[SourceIndex] - SourceIndex objects with image/box info
        - object_count: Sequence[int] - Object counts per image
        - invalid_box_count: Sequence[int] - Invalid box counts per image
        - image_count: int - Total number of images processed
        - stats: Mapping[str, Sequence[Any]] - Mapping of statistic names to sequences of computed values

        Output is sorted by (item_index, box_index) ascending,
        with None values appearing before 0.

    Notes
    -----
    .. versionchanged:: 1.1
        Statistics computed as intermediate dependencies (such as `PIXEL_HISTOGRAM` for
        `PIXEL_ENTROPY` or `VISUAL_PERCENTILES` for `VISUAL_BRIGHTNESS`) are cached at
        runtime and discarded afterwards. They are no longer returned in the final output
        by default unless they are explicitly requested.

    .. versionchanged:: 1.1
        Bit depth is now inferred from the whole image rather than from each region
        separately, so every row of one image is scaled and binned against one range.

    .. versionchanged:: 1.1
        ``VISUAL`` statistics are now always read against the 0–255 display range, so the
        same picture stored as 8-bit, 16-bit or float reports one brightness rather than
        three. They no longer consult `normalize_pixel_values`, which now scopes to
        ``PIXEL`` alone. 8-bit input is unaffected; ``VISUAL`` values from a previous
        ``normalize_pixel_values=True`` run are scaled up by 255 uniformly, which moves no
        threshold that is computed from the data.

    .. versionchanged:: 1.1
        ``VISUAL_SHARPNESS`` previously read raw pixel values, so it alone was comparable
        across encodings under no setting. It now reads the same display range as the rest
        of its family.

    .. versionchanged:: 1.1
        A bit depth is no longer implied for float data spanning more than [0, 255], nor
        for data holding negative values — neither carries an encoding to decode.
        ``DIMENSION_DEPTH`` reports NaN for such data, as do ``PIXEL_HISTOGRAM``,
        ``PIXEL_ENTROPY`` and — under `normalize_pixel_values` — the rest of the pixel
        family, with a warning naming `value_range`.
        Previously a depth was derived from the observed maximum, and data holding
        negative values was binned over [0, 1] regardless of where its values lay.
        ``uint8``, ``uint16``, ``[0, 1]`` float and float-boxed 8-bit data are unaffected.
        This changes two per-target results where a box's own extremes fell in a
        different bit-depth bucket than its image's:

        - ``entropy``'s histogram range, which previously narrowed to the box's own
          maximum. A very dark box in an 8-bit image was binned over ``[0, 1]`` rather
          than ``[0, 255]``.
        - every statistic under ``normalize_pixel_values=True``, which previously scaled
          each box by its own range. A box holding no pixel above 255 in a 12-bit image
          was divided by 255 rather than 4095.

        Values were not comparable across regions before, which is what a per-target
        statistic exists to be. Whole-image results are unchanged.

    .. versionchanged:: 1.1
        Every statistic over a region that was never measured — an out-of-bounds box, an
        image its boxes cover completely, or a band group the datum cannot supply — now
        answers absence rather than a number that reads like an observation:

        - ``entropy`` and ``histogram`` return NaN, where they returned ``0.0`` and 256
          zero bins. ``0.0`` entropy is indistinguishable from a genuinely flat image and
          was reported as an outlier rather than skipped.
        - ``zeros`` returns NaN, where it returned ``0.0``. Its denominator counts pixels
          rather than measurements, so an absent region read as "none of these values are
          zero" — a claim about values there were none of.
        - the hashes return ``""``, where they digested whatever the grayscale conversion
          substituted for NaN. That digest is the same every time, which made every
          unmeasured region a duplicate of every other.

        ``missing`` is the deliberate exception and still answers ``1.0``: it measures the
        presence of data rather than the data, so it is the one statistic that has an
        answer when nothing was measured.

    Examples
    --------
    Compute all statistics:

    >>> from dataeval.flags import ImageStats
    >>> stats = compute_stats(images, boxes=boxes)

    Compute specific statistics:

    >>> stats = compute_stats(images, boxes=boxes, stats=ImageStats.PIXEL_MEAN | ImageStats.VISUAL_BRIGHTNESS)

    Use convenience groups:

    >>> stats = compute_stats(images, boxes=boxes, stats=ImageStats.PIXEL | ImageStats.VISUAL)

    Measure named groups of bands as their own columns:

    >>> stats = compute_stats(images, boxes=boxes, stats=ImageStats.PIXEL_BASIC, channels={"rgb": [0, 1, 2]})

    Ask a different question of each group, and of the image itself. Four-band cubes,
    since ``ir`` names a band that three-channel imagery does not carry:

    >>> cubes = [np.concatenate([image, image[:1]]) for image in images]
    >>> stats = compute_stats(
    ...     cubes,
    ...     stats={None: ImageStats.DIMENSION_WIDTH, "rgb": ImageStats.HASH_XXHASH, "ir": ImageStats.PIXEL_MEAN},
    ...     channels={"rgb": [0, 1, 2], "ir": 3},
    ...     normalize_pixel_values=False,
    ... )
    >>> sorted(stats["stats"])
    ['ir_mean', 'rgb_xxhash', 'width']

    Compute statistics only for bounding boxes (not full images):

    >>> stats = compute_stats(images, boxes=boxes, per_image=False, per_target=True)

    Compute statistics for full images only (ignore boxes):

    >>> stats = compute_stats(images, boxes=boxes, per_image=True, per_target=False)

    Compute background statistics — the scene behind the annotations — alongside the
    whole-image ones:

    >>> stats = compute_stats(
    ...     images, boxes=boxes, stats=ImageStats.VISUAL_BRIGHTNESS, per_background=True, normalize_pixel_values=False
    ... )
    >>> sorted(stats["stats"])
    ['background_brightness', 'background_fraction', 'brightness']
    """
    if normalize_pixel_values is _UNSET:
        warnings.warn(
            "The default value of normalize_pixel_values changed to False in v1.1. "
            "Pass normalize_pixel_values explicitly to silence this warning. "
            "This warning will be removed in v1.2.0.",
            FutureWarning,
            stacklevel=2,
        )
        normalize_pixel_values = False

    # Checked here rather than in a worker: a malformed declaration is a mistake in the
    # call, and surfacing it out of a pool process names the wrong frame.
    if value_range is not None:
        _validate_declared_range(value_range)

    source_indices: list[SourceIndex] = []
    aggregated_stats: dict[str, list[Any]] = {}
    object_count: dict[int, int] = {}
    invalid_box_count: dict[int, int] = {}
    image_count: int = 0
    warning_list: list[str] = []

    is_object_detection_dataset: bool = False

    if isinstance(data, Dataset) and len(data) > 0 and isinstance(data[0], tuple):
        datum = cast(tuple, data[0])
        if len(datum) == 3:
            is_object_detection_dataset = isinstance(datum[1], ObjectDetectionTarget)

    # Per-box rows only exist where there are boxes to make them from, so `per_target`
    # still degrades. `per_background` does not: an image with nothing annotated has a
    # perfectly well-defined background — all of it — and that is already the answer an
    # unannotated image inside an object-detection dataset gets. Degrading it at the
    # dataset level made the *column set* a function of the data rather than of the
    # arguments, so two datasets run with identical arguments could not be combined.
    has_boxes = is_object_detection_dataset or boxes is not None
    per_target = per_target and has_boxes

    # Validate parameters
    if not per_image and not per_target and not per_background:
        raise ValueError("At least one of 'per_image', 'per_target' or 'per_background' must be True")

    if channels is True:
        # Caught by name rather than left to fail inside `_resolve_channel_groups`, where a
        # bool reaches `.items()` and reports itself as an AttributeError naming neither the
        # argument nor its replacement. `channels=True` was the v1.1 spelling of
        # `per_channel=True`, removed in v1.2 along with the row path it selected.
        raise ValueError(
            "channels=True is removed in v1.2: it was a spelling of per_channel=True, which returned "
            "one row per channel. Name the bands instead — for RGB, channels={'r': 0, 'g': 1, 'b': 2} "
            "— which returns them as columns.",
        )
    # Which statistics each view is asked for. One flag set is every view's request; a
    # mapping states them separately, keyed by the prefix the columns carry.
    whole_flags, group_flags = _resolve_stat_flags(stats, channels)

    # Everything the call asks for anywhere. The group-name collision check reads it —
    # `distance` collides with `distance_center` whichever view produces that column — and
    # so does the log line, which should say what was requested rather than what one view got.
    requested = reduce(or_, group_flags.values(), whole_flags)

    # Nothing to compute, said once wherever it came from: ``ImageStats.NONE``, ``{}`` and
    # ``{None: ImageStats.NONE}`` are three spellings of one request and answer the same
    # way — every row, no columns. Gated on the request rather than on the calculators it
    # resolves to, so it stays the single message for a call that asked for nothing and
    # does not pile onto the barren and stranded warnings below, which speak for a call
    # that asked for something and name which part of it produced no column.
    if not requested:
        warnings.warn(
            "stats requests no statistics, so every row will be returned with no columns. "
            "ImageStats.NONE, an empty mapping and {None: ImageStats.NONE} all mean this — "
            "name the statistics you want, or drop the compute_stats call.",
            UserWarning,
            stacklevel=2,
        )

    calculators = CalculatorRegistry.get_calculators(whole_flags)
    channel_groups = _resolve_channel_groups(channels or {}, CalculatorRegistry.get_calculators(requested))

    # One calculator list per group rather than one shared list, since a mapping may ask a
    # different question of each. Narrowed to the band view here so a group that turns out
    # to produce nothing is caught at the call rather than once per datum.
    band_calculators = {
        name: _calculators_for_view(CalculatorRegistry.get_calculators(group_flags[name]), ViewKind.BAND)
        for name in channel_groups
    }
    # What those lists amount to in flags, for the questions asked about the *request*
    # rather than about the work it schedules: which of a group's statistics produce no
    # column, and which go NaN for want of a value range. Read off the lists rather than
    # narrowed a second time through the registry, so the two answers cannot drift. Taken
    # before the barren groups are dropped, since the stranded check below still reads them.
    band_flags = {name: _view_flags(group_calculators) for name, group_calculators in band_calculators.items()}
    # Statistics a mapping asks of a band group that no band group can answer, and that its
    # whole-image entry does not ask for either — so nothing computes them. The loss worth
    # saying is geometry: a mapping is a total statement, so omitting the ``None`` entry
    # drops `width` silently rather than wrongly. Empty by construction for a single flag
    # set, where every view is asked the same question and the last line cancels it.
    #
    # Said before the barren warning below, which the one mistake that trips both — a group
    # asked only for band-invariant statistics, with no whole-image entry — would otherwise
    # let speak first. The two are different questions, one about a group and one about a
    # statistic, and both answers are worth having; but under ``-W error`` only the first
    # survives, and this is the one that names the column that went missing and the entry
    # that brings it back. Barren's remedy changes what the group measures instead, which
    # answers a question the caller did not ask.
    stranded = ImageStats(0)
    for name, flags in group_flags.items():
        stranded |= flags & ~band_flags[name]
    stranded &= ~whole_flags
    if stranded:
        warnings.warn(
            f"{_flag_names(stranded)} do not vary with a band subset, so naming them for a channel "
            f"group produces no column, and the stats mapping has no whole-image entry asking for "
            f"them — nothing computes them. Add None: <flags> to the mapping to measure them over "
            f"the image and its boxes.",
            UserWarning,
            stacklevel=2,
        )

    # A group producing no column is dropped whatever the reason, so the workers do not
    # slice it off every datum and scan the slice for a range nothing will read. Only the
    # groups that *asked* for something are worth a word: one handed ``ImageStats.NONE``
    # said exactly what it wanted, and telling it to request PIXEL, VISUAL or HASH would
    # answer a question it did not ask. `_check_stats_mapping` makes that the only spelling
    # of "define this group but do not measure it", so it has to be able to stay quiet.
    dead = [name for name, group_calculators in band_calculators.items() if not group_calculators]
    barren = sorted(name for name in dead if group_flags[name])
    if barren:
        warnings.warn(
            f"channel groups {{{_names(barren)}}} were named but none of the requested statistics "
            f"vary with a band subset, so no band columns will be returned for them. Geometry does "
            f"not narrow when channels are dropped; request ImageStats.PIXEL, VISUAL or HASH for a "
            f"group.",
            UserWarning,
            stacklevel=2,
        )
    for name in dead:
        del band_calculators[name]
        del channel_groups[name]

    # The background reduces over a masked region, which only some statistics survive;
    # the rest stay available to the image and its boxes, where they still mean something.
    background_calculators: list[tuple[type[Any], Flag]] = []
    background_band_calculators: dict[str, list[tuple[type[Any], Flag]]] = {}
    if per_background:
        background_calculators = _calculators_for_view(calculators, ViewKind.MASK)
        background_band_calculators = {
            name: masked
            for name, group_calculators in band_calculators.items()
            if (masked := _calculators_for_view(group_calculators, ViewKind.MASK))
        }
        if not background_calculators and not background_band_calculators:
            warnings.warn(
                f"per_background=True but none of the requested statistics apply to a background region, "
                f"so only '{BACKGROUND_PREFIX}fraction' will be returned. A background is a region with "
                f"part of it masked out, which hash statistics are not stable under and which dimension "
                f"statistics do not describe; request ImageStats.PIXEL or ImageStats.VISUAL for it.",
                UserWarning,
                stacklevel=2,
            )

    # Said once per datum that trips it, aggregated with the rest. Only the unnamed view is
    # at risk, and only for the statistics no group answers in its place — naming bands
    # settles the question for a statistic some group also measures, but a mapping can name
    # a group and still ask the whole cube for a brightness nothing reads per band. Keyed on
    # what the groups produce rather than on whether any group exists, which was the same
    # test only while every group was handed the same flags.
    # Rendered here rather than in the worker: the answer is the same for every datum.
    covered = reduce(or_, band_flags.values(), ImageStats(0))
    wide_band_names = _flag_names(whole_flags & (ImageStats.VISUAL | ImageStats.HASH) & ~covered)

    # Statistics that go NaN for want of an interval, named per view: the unnamed one is
    # measured against the whole datum's range, each group against its own.
    unmeasurable_names = _flag_names(_unmeasurable_flags(whole_flags, value_range, normalize_pixel_values))
    band_unmeasurable_names: dict[str, str] = {}
    for name, group in channel_groups.items():
        at_risk = _unmeasurable_flags(group_flags[name], group.value_range or value_range, normalize_pixel_values)
        # Narrowed to the band view so the message names the columns the group produces
        # rather than every statistic that would need an interval somewhere.
        group_names = _flag_names(at_risk & band_flags[name])
        if group_names:
            band_unmeasurable_names[name] = group_names

    # One object per group, assembled where every part of it is known. The mappings above
    # are locals with deliberately different key sets — `band_flags` still carries the
    # groups that were dropped, since the stranded check reads them, and the masked
    # calculators cover only the groups a mask does not destroy — so reconciling them here
    # rather than in the worker is what lets `_compute_batch` walk one mapping and read
    # fields instead of testing membership of four.
    band_plans = {
        name: BandPlan(
            indices=group.indices,
            value_range=group.value_range,
            calculators=band_calculators[name],
            background_calculators=background_band_calculators.get(name, ()),
            unmeasurable_names=band_unmeasurable_names.get(name, ""),
        )
        for name, group in channel_groups.items()
    }

    # Log the individual flags that will be computed
    resolved_names = _flag_list(requested)
    _logger.info(
        "Starting compute_stats: %d stats [%s], per_image=%s, per_target=%s, per_background=%s",
        len(resolved_names),
        ", ".join(resolved_names),
        per_image,
        per_target,
        per_background,
    )

    total_images = len(data) if isinstance(data, Sized) else None

    # Boxes have to be read off the dataset whenever anything is defined by them — the
    # per-box rows, the background mask, or both. Gated on their availability separately
    # from whether they were asked for: `unzip_dataset` validates the dataset as
    # object-detection shaped when asked for targets, so requesting a background over a
    # plain image dataset must not send it looking for boxes that cannot be there.
    needs_boxes = (per_target or per_background) and has_boxes
    images, boxes = (
        (data, boxes)
        if not isinstance(data, Dataset)
        else (unzip_dataset(data, per_target=False)[0], boxes)
        if boxes is not None
        else unzip_dataset(data, per_target=needs_boxes)
    )

    # Build description for progress bar
    calculator_names = [c[0].__name__.removesuffix("Calculator") for c in calculators]
    _logger.debug("Using calculators: %s", calculator_names)

    with PoolWrapper(processes=get_max_processes()) as p:
        for result in p.imap_unordered(
            partial(
                _compute_batch,
                plan=BatchPlan(
                    calculators=calculators,
                    per_image=per_image,
                    per_target=per_target,
                    normalize_pixel_values=normalize_pixel_values,
                    per_background=per_background,
                    background_calculators=background_calculators,
                    declared_range=value_range,
                    band_plans=band_plans,
                    wide_band_names=wide_band_names,
                    unmeasurable_names=unmeasurable_names,
                ),
            ),
            _enumerate_datum(images, boxes),
        ):
            _aggregate_batch(result, source_indices, aggregated_stats, object_count, invalid_box_count, warning_list)
            image_count += 1

            if progress_callback:
                progress_callback(image_count, total=total_images)

    # Aggregate warnings by message type, collecting indices per type
    grouped_warnings: dict[str, list[str]] = {}
    for w in warning_list:
        idx, _, msg = w.partition(": ")
        grouped_warnings.setdefault(msg, []).append(idx)
    for msg, indices in grouped_warnings.items():
        _logger.warning("%s — indices: %s", msg, ", ".join(indices))

    _logger.debug("Sorting %d source indices and %d stats", len(source_indices), len(aggregated_stats))
    sorted_source_indices, sorted_aggregated_stats = _sort(source_indices, aggregated_stats)

    total_boxes = sum(object_count.values())
    total_invalid = sum(invalid_box_count.values())
    _logger.info(
        "compute_stats complete: %d images processed, %d total boxes (%d invalid), %d stats computed",
        image_count,
        total_boxes,
        total_invalid,
        len(sorted_aggregated_stats),
    )

    return StatsResult(
        source_index=sorted_source_indices,
        object_count=[object_count.get(i, 0) for i in range(image_count)],
        invalid_box_count=[invalid_box_count.get(i, 0) for i in range(image_count)],
        image_count=image_count,
        stats=sorted_aggregated_stats,
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
    results: StatsResult | Sequence[StatsResult],
) -> tuple[StatsMap, list[SourceIndex], list[int]]:
    """Combine one or more StatsResults into unified stats, source_index, and dataset_steps.

    For a single StatsResult, returns its stats and source_index directly
    with empty dataset_steps.

    For multiple results, concatenates stats arrays by key, applies cumulative
    item offsets to source_index entries (making item indices globally unique
    across datasets), and computes cumulative dataset_steps boundaries.

    Parameters
    ----------
    results : StatsResult or Sequence[StatsResult]
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
        If an empty sequence is provided.
    """
    if isinstance(results, dict):
        return results["stats"], list(results["source_index"]), []

    if len(results) == 0:
        raise TypeError("Cannot combine empty sequence of stats.")

    if len(results) == 1:
        return results[0]["stats"], list(results[0]["source_index"]), []

    combined_stats: StatsMap | None = None
    combined_source_index: list[SourceIndex] = []
    dataset_steps: list[int] = []
    offset = 0

    for position, r in enumerate(results):
        stats = r["stats"]
        if not r["source_index"]:
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
        combined_source_index.extend(s._replace(item=s.item + offset) for s in r["source_index"])
        offset += len(r["source_index"])
        dataset_steps.append(offset)

    # `None` only where every result was empty, which is an empty result rather than an
    # absent one — the caller asked for a combination and gets the shape of one.
    return combined_stats if combined_stats is not None else {}, combined_source_index, dataset_steps
