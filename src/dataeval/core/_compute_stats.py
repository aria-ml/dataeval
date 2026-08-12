__all__ = []

import logging
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence, Sized
from dataclasses import dataclass
from enum import Flag
from functools import partial
from itertools import zip_longest
from typing import Any, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

# Import calculators to trigger auto-registration
import dataeval.core._calculators._register  # noqa: F401
from dataeval.config import get_max_processes
from dataeval.core._calculators._base import Calculator
from dataeval.core._calculators._cache import CalculatorCache
from dataeval.core._calculators._registry import CalculatorRegistry
from dataeval.data import unzip_dataset
from dataeval.flags import ImageStats
from dataeval.protocols import ArrayLike, Dataset, ObjectDetectionTarget, ProgressCallback
from dataeval.types import SourceIndex, StatsMap
from dataeval.utils._internal import PoolWrapper
from dataeval.utils.preprocessing import BitDepth, BoundingBox, BoxLike, boxes_to_mask, get_bitdepth, to_bounding_box

_logger = logging.getLogger(__name__)

SOURCE_INDEX = "source_index"

BACKGROUND_PREFIX = "background_"
"""Name prefix distinguishing a background statistic from the whole-image one beside it.

Background values share a row with the full image's — both describe the same item and
carry the same :class:`~dataeval.types.SourceIndex` — so they are told apart by name
rather than by position. Adding a row instead is not available: a source index addresses
exactly one item-level row per item, and naming it twice is rejected outright.
"""


def _maskable_calculators(
    calculators: Sequence[tuple[type[Calculator[Any]], Flag]],
) -> list[tuple[type[Calculator[Any]], Flag]]:
    """Keep only the calculators whose statistics survive part of the region being masked.

    Which those are is each calculator's own answer, via its ``supports_exclusion``,
    rather than a list of flags maintained here: a new calculator would otherwise have to
    be remembered in a module that knows nothing else about it, and would silently default
    to being run over masked data. Queried the same way the registry queries
    ``get_applicable_flags``.
    """
    maskable: list[tuple[type[Calculator[Any]], Flag]] = []
    for calculator_cls, flags in calculators:
        # __new__ rather than a real instance: the answer is a property of the class and
        # the constructor wants a datum. Same trick, and the same reason, as the registry.
        if calculator_cls.__new__(calculator_cls).supports_exclusion():
            maskable.append((calculator_cls, flags))
    return maskable


class StatsResult(TypedDict):
    """
    Type definition for calculation output.

    Attributes
    ----------
    source_index : Sequence[SourceIndex]
        Sequence of SourceIndex objects with image/box/channel info.
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
    per_channel: bool,
    normalize_pixel_values: bool = False,
    exclude: NDArray[np.bool_] | None = None,
    prefix: str = "",
    bitdepth: BitDepth | None = None,
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
    bitdepth : BitDepth or None, default None
        The datum's bit depth, already found. Every row of one datum shares it, so it is
        read once per datum and handed down rather than rediscovered per row — finding it
        costs a scan of the whole datum, which a per-row cache would repeat once per box.

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
        per_channel,
        normalize_pixel_values=normalize_pixel_values,
        exclude=exclude,
        bitdepth=bitdepth,
    )
    for calculator_cls, flags in calculators:
        calculator = calculator_cls(datum, processor, per_channel)
        stats_list.append({f"{prefix}{name}": values for name, values in calculator.compute(flags).items()})
        # Collect empty values from this calculator
        empty_values_map.update({f"{prefix}{name}": value for name, value in calculator.get_empty_values().items()})
        # Collect warnings from this calculator
        if hasattr(calculator, "warnings"):
            warnings.extend(calculator.warnings)
        del calculator
    return stats_list, empty_values_map, warnings


def _determine_channel_indices(calculator_output: list[dict[str, list[Any]]], num_channels: int) -> list[int | None]:
    """Determine what channel indices are needed based on processor outputs."""
    channel_indices_needed: set[int | None] = set()

    for output in calculator_output:
        first_stat_values = next(iter(output.values()))
        num_elements = len(first_stat_values)

        if num_elements == 1:
            # Single value per image/box - uses channel=None
            channel_indices_needed.add(None)
        elif num_elements == num_channels:
            # Per-channel values - uses channel=0,1,2,...
            channel_indices_needed.update(range(num_channels))
        else:
            # Unexpected case
            raise ValueError(
                f"Processor produced {num_elements} values but image has {num_channels} channels. "
                f"Expected either 1 (image-level) or {num_channels} (per-channel) values.",
            )

    # Return ordered list of channel indices (None first, then 0,1,2,...)
    return sorted(channel_indices_needed, key=lambda x: -1 if x is None else x)


def _reconcile_stats(  # noqa: C901
    calculator_output: list[dict[str, list[Any]]],
    sorted_channels: list[int | None],
    empty_values_map: dict[str, Any],
) -> dict[str, list[Any]]:
    """
    Reconcile stats from different processors into a unified structure.

    Uses empty values from empty_values_map for stats that don't apply to certain channels.
    Defaults to np.nan if a stat is not in the empty_values_map.
    """
    num_entries = len(sorted_channels)
    reconciled_stats: dict[str, list[Any]] = {}

    for output in calculator_output:
        first_stat_values = next(iter(output.values()))
        num_elements = len(first_stat_values)

        for stat_name, stat_values in output.items():
            if stat_name not in reconciled_stats:
                # Use the appropriate empty value for this stat (default to np.nan)
                empty_value = empty_values_map.get(stat_name, np.nan)
                reconciled_stats[stat_name] = [empty_value] * num_entries

            if num_elements == 1:
                # Single value goes to channel=None position
                none_idx = sorted_channels.index(None)
                reconciled_stats[stat_name][none_idx] = stat_values[0]
            else:
                # Per-channel values go to their respective positions
                for ch_idx, value in enumerate(stat_values):
                    ch_pos = sorted_channels.index(ch_idx)
                    reconciled_stats[stat_name][ch_pos] = value

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
    calculators: Iterable[tuple[type[Any], Flag]],
    per_image: bool,
    per_target: bool,
    per_channel: bool,
    normalize_pixel_values: bool = False,
    per_background: bool = False,
    background_calculators: Iterable[tuple[type[Any], Flag]] = (),
) -> DatumBatchResult:
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
    # view of it shares the value, and finding it costs a scan of the whole array.
    bitdepth = get_bitdepth(datum)

    # The foreground mask: every annotated box painted into one array, so overlapping
    # boxes count once. Built per datum rather than per row — the item-level row is the
    # only one that uses it, but building it here keeps it out of the row loop.
    exclude = boxes_to_mask(datum.shape, boxes or ()) if per_background else None
    # `.all()` is vacuously True for a datum with no pixels, which is an absence of
    # background rather than a full one; `.mean()` on it is NaN and warns. Neither is
    # worth saying, so an empty datum takes neither path.
    if exclude is not None and exclude.size and exclude.all():
        warnings_list.append(f"{i}: Boxes cover the entire datum, leaving no background to measure")

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
                per_channel,
                normalize_pixel_values=normalize_pixel_values,
                bitdepth=bitdepth,
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
                per_channel,
                normalize_pixel_values=normalize_pixel_values,
                exclude=exclude,
                prefix=BACKGROUND_PREFIX,
                bitdepth=bitdepth,
            )
            calculator_stats.extend(stats)
            empty_values_map.update(empties)
            calc_warnings.extend(warns)
            # Emitted whatever was requested, because every other background value is
            # only as trustworthy as the share of the image it was measured over. A datum
            # with no pixels has no share to report, so it reports NaN rather than 0/0.
            uncovered = float(1.0 - exclude.mean()) if exclude.size else float("nan")
            calculator_stats.append({f"{BACKGROUND_PREFIX}fraction": [uncovered]})

        # Thread calculator warnings with index context
        for w in calc_warnings:
            source = f"{i}" if box is None else f"{i}[{i_b}]"
            warnings_list.append(f"{source}: {w}")

        # Determine what channel indices are needed
        sorted_channels = _determine_channel_indices(calculator_stats, num_channels)

        # Reconcile stats into unified structure
        reconciled_stats = _reconcile_stats(calculator_stats, sorted_channels, empty_values_map)
        datum_empty_values.update(empty_values_map)

        # Build index lists
        channel_indices = sorted_channels
        source = [SourceIndex(i, i_b if box is not None else None, c) for c in channel_indices]

        results.append(DatumResult(source_indices=source, stats=reconciled_stats))

    # Background statistics exist on the item-level row alone, so the box rows need
    # placeholders for them before the rows are flattened into per-name arrays.
    if per_background:
        _pad_missing_stats(results, datum_empty_values)

    return DatumBatchResult(results, box_count, invalid_box_count, warnings_list)


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
    """Sort results by (item_index, box_index, channel_index) with None < 0 and convert to numpy arrays."""
    sort_indices = sorted(
        range(len(source_indices)),
        key=lambda i: (
            source_indices[i].item,
            -1 if source_indices[i].target is None else source_indices[i].target,
            -1 if source_indices[i].channel is None else source_indices[i].channel,
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
    stats: Flag = ImageStats.ALL,
    per_image: bool = True,
    per_target: bool = True,
    per_background: bool = False,
    per_channel: bool = False,
    normalize_pixel_values: bool = _UNSET,  # type: ignore
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
    stats : ImageStats, default ImageStats.ALL
        Flags indicating which statistics to compute. Can combine multiple flags
        using bitwise OR (|). Dependencies are resolved automatically for calculation,
        but intermediate/dependency statistics are not included in the output by
        default unless explicitly requested.
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

        Has no effect when boxes is None.

        .. versionadded:: 1.2
    per_channel : bool, default False
        If True, compute per-channel statistics. If False, statistics are
        aggregated across all channels.
    normalize_pixel_values : bool, default False
        If True, pixel values are normalized to [0, 1] based on each image's
        inferred bit depth before any statistic is computed. This makes results
        comparable across images with different bit depths (8-bit, 16-bit, etc.).
        If False, statistics are computed on raw pixel values.

        .. deprecated::
            The default changed to False in v1.1. Pass explicitly to silence
            the deprecation warning. This warning will be removed in v1.2.
    progress_callback : ProgressCallback or None, default None
        Callback to report progress during calculation. Called after each image is processed
        with the current image count and total number of images (if known).

    Returns
    -------
    StatsResult
        Mapping containing computed statistics and metadata:

        - source_index: Sequence[SourceIndex] - SourceIndex objects with image/box/channel info
        - object_count: Sequence[int] - Object counts per image
        - invalid_box_count: Sequence[int] - Invalid box counts per image
        - image_count: int - Total number of images processed
        - stats: Mapping[str, Sequence[Any]] - Mapping of statistic names to sequences of computed values

        Output is sorted by (item_index, box_index, channel_index) ascending,
        with None values appearing before 0.

    Notes
    -----
    .. versionchanged:: 1.1
        Statistics computed as intermediate dependencies (such as `PIXEL_HISTOGRAM` for
        `PIXEL_ENTROPY` or `VISUAL_PERCENTILES` for `VISUAL_BRIGHTNESS`) are cached at
        runtime and discarded afterwards. They are no longer returned in the final output
        by default unless they are explicitly requested.

    .. versionchanged:: 1.2
        Bit depth is now inferred from the whole image rather than from each region
        separately, so every row of one image is scaled and binned against one range.
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

    .. versionchanged:: 1.2
        ``entropy`` and ``histogram`` return NaN for a region that is entirely NaN — an
        out-of-bounds box, or an image its boxes cover completely — where they previously
        returned ``0.0`` and 256 zero bins. Every other statistic already answered NaN
        for such a region; ``0.0`` entropy is indistinguishable from a genuinely flat
        image and was reported as an outlier rather than skipped.

    Examples
    --------
    Compute all statistics:

    >>> from dataeval.flags import ImageStats
    >>> stats = compute_stats(images, boxes=boxes)

    Compute specific statistics:

    >>> stats = compute_stats(images, boxes=boxes, stats=ImageStats.PIXEL_MEAN | ImageStats.VISUAL_BRIGHTNESS)

    Use convenience groups:

    >>> stats = compute_stats(images, boxes=boxes, stats=ImageStats.PIXEL | ImageStats.VISUAL)
    >>> stats = compute_stats(images, boxes=boxes, stats=ImageStats.PIXEL_BASIC, per_channel=True)

    Compute statistics only for bounding boxes (not full images):

    >>> stats = compute_stats(images, boxes=boxes, per_image=False, per_target=True)

    Compute statistics for full images only (ignore boxes):

    >>> stats = compute_stats(images, boxes=boxes, per_image=True, per_target=False)

    Compute statistics for both full images and boxes with per-channel breakdown:

    >>> stats = compute_stats(images, boxes=boxes, per_image=True, per_target=True, per_channel=True)

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
            "This warning will be removed in v1.2.",
            FutureWarning,
            stacklevel=2,
        )
        normalize_pixel_values = False

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

    # `per_target` and `per_background` are True only if boxes are provided or data is
    # an ObjectDetectionDataset — both are defined by the boxes.
    has_boxes = is_object_detection_dataset or boxes is not None
    per_target = per_target and has_boxes
    per_background = per_background and has_boxes

    # Validate parameters
    if not per_image and not per_target and not per_background:
        raise ValueError("At least one of 'per_image', 'per_target' or 'per_background' must be True")

    # Get calculators from registry based on flags
    calculators = CalculatorRegistry.get_calculators(stats)

    # The background reduces over a masked region, which only some statistics survive;
    # the rest stay available to the image and its boxes, where they still mean something.
    background_calculators: list[tuple[type[Any], Flag]] = []
    if per_background:
        background_calculators = _maskable_calculators(calculators)
        if not background_calculators:
            warnings.warn(
                f"per_background=True but none of the requested statistics apply to a background region, "
                f"so only '{BACKGROUND_PREFIX}fraction' will be returned. A background is a region with "
                f"part of it masked out, which hash statistics are not stable under and which dimension "
                f"statistics do not describe; request ImageStats.PIXEL or ImageStats.VISUAL for it.",
                UserWarning,
                stacklevel=2,
            )

    # Log the individual flags that will be computed
    resolved_names = [
        f.name for f in type(stats) if f in stats and f.name and f.value and (f.value & (f.value - 1)) == 0
    ]
    _logger.info(
        "Starting compute_stats: %d stats [%s], per_image=%s, per_target=%s, per_background=%s, per_channel=%s",
        len(resolved_names),
        ", ".join(resolved_names),
        per_image,
        per_target,
        per_background,
        per_channel,
    )

    total_images = len(data) if isinstance(data, Sized) else None

    # Boxes have to be read off the dataset whenever anything is defined by them — the
    # per-box rows, the background mask, or both.
    needs_boxes = per_target or per_background
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
                calculators=calculators,
                per_image=per_image,
                per_target=per_target,
                per_channel=per_channel,
                normalize_pixel_values=normalize_pixel_values,
                per_background=per_background,
                background_calculators=background_calculators,
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

    combined_stats: StatsMap = {}
    combined_source_index: list[SourceIndex] = []
    dataset_steps: list[int] = []
    offset = 0

    for r in results:
        stats = r["stats"]
        if not combined_stats:
            combined_stats = stats
        else:
            combined_stats = {k: np.concatenate([combined_stats[k], stats[k]]) for k in combined_stats if k in stats}

        combined_source_index.extend(
            SourceIndex(item=s.item + offset, target=s.target, channel=s.channel) for s in r["source_index"]
        )
        offset += len(r["source_index"])
        dataset_steps.append(offset)

    return combined_stats, combined_source_index, dataset_steps
