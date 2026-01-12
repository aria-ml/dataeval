__all__ = []

import logging
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence, Sized
from dataclasses import dataclass
from enum import Flag
from functools import cached_property, partial
from itertools import zip_longest
from multiprocessing import Pool
from os import cpu_count
from typing import Any, TypedDict, TypeVar, cast

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

# Import calculators to trigger auto-registration
import dataeval.core._calculators._register  # noqa: F401
from dataeval.config import get_max_processes
from dataeval.core._calculators._registry import CalculatorRegistry
from dataeval.flags import ImageStats, resolve_dependencies
from dataeval.protocols import (
    ArrayLike,
    Dataset,
    ObjectDetectionTarget,
    ProgressCallback,
)
from dataeval.types import SourceIndex
from dataeval.utils.data import unzip_dataset
from dataeval.utils.preprocessing import (
    BoundingBox,
    BoxLike,
    clip_and_pad,
    normalize_image_shape,
    rescale,
    to_bounding_box,
)

_S = TypeVar("_S")
_T = TypeVar("_T")


_logger = logging.getLogger(__name__)

SOURCE_INDEX = "source_index"


class CalculationResult(TypedDict):
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
class CalculatorResult:
    """Result from processing a single image/box combination."""

    source_indices: list[SourceIndex]
    stats: dict[str, list[Any]]


@dataclass
class CalculatorOutput:
    """Output from processing multiple images."""

    results: list[CalculatorResult]
    object_count: int
    invalid_box_count: int
    warnings_list: list[str]


class CalculatorCache:
    """
    A calculator cache for a single datum (image, text, etc.).

    Provides preprocessing and cached transformations of the raw datum.
    This class adapts based on the data type passed in.
    """

    def __init__(self, datum: Any, box: BoundingBox | None = None, per_channel: bool = False) -> None:
        is_spatial = len(datum.shape) >= 2
        self.raw = datum
        # Assume image data for now (will be generic in future)
        self.width: int = datum.shape[-1] if is_spatial else 0
        self.height: int = datum.shape[-2] if is_spatial else 0
        self.shape: tuple[int, ...] = datum.shape
        self.per_channel_mode = per_channel
        self.has_box = box is not None

        # Ensure bounding box
        self.box = BoundingBox(0, 0, self.width, self.height, image_shape=datum.shape) if box is None else box

    @cached_property
    def image(self) -> NDArray[Any]:
        # Only normalize image shape if we have bounding boxes, since only image/video data
        # will have bounding box concepts. Otherwise, we cannot assume dimensionality >= 3.
        if self.has_box:
            return clip_and_pad(normalize_image_shape(self.raw), self.box.xyxy_int)
        # For non-image data or data without boxes, return as-is after ensuring minimum shape
        if self.raw.ndim >= 3:
            return clip_and_pad(normalize_image_shape(self.raw), self.box.xyxy_int)
        # For data with < 3 dimensions, don't normalize or clip
        return self.raw

    @cached_property
    def scaled(self) -> NDArray[Any]:
        return rescale(self.image)

    @cached_property
    def per_channel(self) -> NDArray[Any]:
        # For data with >= 3 dimensions, reshape as (channels, -1)
        # For data with < 3 dimensions, treat as single channel
        if self.image.ndim >= 3:
            return self.scaled.reshape(self.image.shape[0], -1)
        # For lower-dimensional data, add a channel dimension
        return self.scaled.reshape(1, -1)


class PoolWrapper:
    """
    Wraps `multiprocessing.Pool` to allow for easy switching between
    multiprocessing and single-threaded execution.

    This helps with debugging and profiling, as well as usage with Jupyter notebooks
    in VS Code, which does not support subprocess debugging.
    """

    def __init__(self, processes: int | None) -> None:
        procs = 1 if processes is None else max(1, (cpu_count() or 1) + processes + 1) if processes < 0 else processes
        self.pool = Pool(procs) if procs > 1 else None

    def imap_unordered(self, func: Callable[[_S], _T], iterable: Iterable[_S]) -> Iterator[_T]:
        return map(func, iterable) if self.pool is None else self.pool.imap_unordered(func, iterable)

    def __enter__(self, *args: Any, **kwargs: Any) -> Self:
        return self

    def __exit__(self, *args: Any) -> None:
        if self.pool is not None:
            self.pool.close()
            self.pool.join()


def _collect_calculator_stats(
    calculators: Iterable[tuple[type[Any], Flag]],
    datum: NDArray[Any],
    box: BoundingBox | None,
    per_channel: bool,
) -> tuple[list[dict[str, list[Any]]], dict[str, Any]]:
    """
    Collect stats from all calculators.

    Returns
    -------
    tuple[list[dict[str, list[Any]]], dict[str, Any]]
        A tuple of (stats_list, empty_values_map) where:
        - stats_list: List of computed stats from each calculator
        - empty_values_map: Mapping of stat names to their empty values (defaults to np.nan)
    """
    stats_list = []
    empty_values_map: dict[str, Any] = {}
    processor = CalculatorCache(datum, box, per_channel)
    for calculator_cls, flags in calculators:
        calculator = calculator_cls(datum, processor, per_channel)
        stats_list.append(calculator.compute(flags))
        # Collect empty values from this calculator
        empty_values_map.update(calculator.get_empty_values())
        del calculator
    return stats_list, empty_values_map


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
                f"Expected either 1 (image-level) or {num_channels} (per-channel) values."
            )

    # Return ordered list of channel indices (None first, then 0,1,2,...)
    return sorted(channel_indices_needed, key=lambda x: (-1 if x is None else x))


def _reconcile_stats(
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
) -> list[tuple[int | None, BoundingBox | None]]:
    """Determine what to process based on per_image and per_target flags."""
    process_items: list[tuple[int | None, BoundingBox | None]] = []

    if boxes is None or len(boxes) == 0:
        # No boxes provided - only process full image if per_image is True
        if per_image:
            process_items.append((None, None))
    else:
        # Boxes are provided
        if per_image:
            # Add full image processing
            process_items.append((None, None))

        if per_target:
            # Add per-box processing
            process_items.extend((i_b, box) for i_b, box in enumerate(boxes))

    return process_items


def _calculate_datum(
    i: int,
    datum: NDArray[Any],
    boxes: list[BoundingBox] | None,
    calculators: Iterable[tuple[type[Any], Flag]],
    per_image: bool,
    per_target: bool,
    per_channel: bool,
) -> CalculatorOutput:
    results: list[CalculatorResult] = []
    box_count = 0
    invalid_box_count = 0
    warnings_list: list[str] = []

    # Determine the number of channels from the datum shape
    num_channels = datum.shape[-3] if len(datum.shape) >= 3 else 1

    # Determine what to process based on per_image and per_target flags
    items = _get_items(boxes, per_image, per_target)

    # Process each item (full image and/or boxes)
    for i_b, box in items:
        if box is not None:
            box_count += 1
            if not box.is_clippable():
                invalid_box_count += 1
                warnings_list.append(f"Bounding box [{i}][{i_b}]: {box} for datum shape {datum.shape} is invalid.")

        # Collect stats from all calculators
        calculator_stats, empty_values_map = _collect_calculator_stats(calculators, datum, box, per_channel)

        # Determine what channel indices are needed
        sorted_channels = _determine_channel_indices(calculator_stats, num_channels)

        # Reconcile stats into unified structure
        reconciled_stats = _reconcile_stats(calculator_stats, sorted_channels, empty_values_map)

        # Build index lists
        channel_indices = sorted_channels
        source = [SourceIndex(i, i_b if box is not None else None, c) for c in channel_indices]

        results.append(CalculatorResult(source_indices=source, stats=reconciled_stats))

    return CalculatorOutput(results, box_count, invalid_box_count, warnings_list)


def _unpack(
    args: tuple[int, NDArray[Any], list[BoundingBox] | None],
    calculators: Iterable[tuple[type[Any], Flag]],
    per_image: bool,
    per_target: bool,
    per_channel: bool,
) -> CalculatorOutput:
    return _calculate_datum(*args, calculators, per_image, per_target, per_channel)


def _enumerate(
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
        # If the values are floats, convert to dtype float16 for efficiency
        if np.issubdtype(np_array.dtype, np.floating):
            np_array = np_array.astype(np.float16)
        sorted_aggregated_stats[stat_name] = np_array

    return sorted_source_indices, sorted_aggregated_stats


def _aggregate(
    result: CalculatorOutput,
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


def calculate(
    data: Iterable[ArrayLike] | Dataset[ArrayLike] | Dataset[tuple[ArrayLike, Any, Any]],
    boxes: Iterable[Iterable[BoxLike] | None] | None = None,
    stats: Flag = ImageStats.ALL,
    *,
    per_image: bool = True,
    per_target: bool = True,
    per_channel: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> CalculationResult:
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
        using bitwise OR (|). Dependencies are resolved automatically.
    per_image : bool, default True
        If True, compute statistics for entire images. When boxes are provided
        and per_image=True, statistics are computed for both the full image and
        each box (if per_target=True).
    per_target : bool, default True
        If True and boxes are provided, compute statistics for each bounding box.
        Has no effect when boxes is None. At least one of per_image or per_target
        must be True.
    per_channel : bool, default False
        If True, compute per-channel statistics. If False, statistics are
        aggregated across all channels.
    progress_callback : ProgressCallback or None, default None
        Callback to report progress during calculation. Called after each image is processed
        with the current image count and total number of images (if known).

    Returns
    -------
    CalculationResult
        Mapping containing computed statistics and metadata:

        - source_index: Sequence[SourceIndex] - SourceIndex objects with image/box/channel info
        - object_count: Sequence[int] - Object counts per image
        - invalid_box_count: Sequence[int] - Invalid box counts per image
        - image_count: int - Total number of images processed
        - stats: Mapping[str, Sequence[Any]] - Mapping of statistic names to sequences of computed values

        Output is sorted by (item_index, box_index, channel_index) ascending,
        with None values appearing before 0.

    Examples
    --------
    Compute all statistics:

    >>> from dataeval.flags import ImageStats
    >>> stats = calculate(images, boxes)

    Compute specific statistics:

    >>> stats = calculate(images, boxes, stats=ImageStats.PIXEL_MEAN | ImageStats.VISUAL_BRIGHTNESS)

    Use convenience groups:

    >>> stats = calculate(images, boxes, stats=ImageStats.PIXEL | ImageStats.VISUAL)
    >>> stats = calculate(images, boxes, stats=ImageStats.PIXEL_BASIC, per_channel=True)

    Compute statistics only for bounding boxes (not full images):

    >>> stats = calculate(images, boxes, per_image=False, per_target=True)

    Compute statistics for full images only (ignore boxes):

    >>> stats = calculate(images, boxes, per_image=True, per_target=False)

    Compute statistics for both full images and boxes with per-channel breakdown:

    >>> stats = calculate(images, boxes, per_image=True, per_target=True, per_channel=True)
    """
    source_indices: list[SourceIndex] = []
    aggregated_stats: dict[str, list[Any]] = {}
    object_count: dict[int, int] = {}
    invalid_box_count: dict[int, int] = {}
    image_count: int = 0
    warning_list: list[str] = []

    isObjectDetectionDataset: bool = False

    if isinstance(data, Dataset) and len(data) > 0 and isinstance(data[0], tuple):
        datum = cast(tuple, data[0])
        if len(datum) == 3:
            isObjectDetectionDataset = isinstance(datum[1], ObjectDetectionTarget)

    # `per_target` is True only if boxes are provided or data is an ObjectDetectionDataset
    per_target = per_target and (isObjectDetectionDataset or boxes is not None)

    # Validate parameters
    if not per_image and not per_target:
        raise ValueError("At least one of 'per_image' or 'per_target' must be True")

    # Resolve dependencies
    stats = resolve_dependencies(stats)

    # Get calculators from registry based on flags
    calculators = CalculatorRegistry.get_calculators(stats)

    _logger.info(
        "Starting calculate with per_image=%s, per_target=%s, per_channel=%s",
        per_image,
        per_target,
        per_channel,
    )

    total_images = len(data) if isinstance(data, Sized) else None

    images, boxes = (
        (data, boxes)
        if not isinstance(data, Dataset)
        else (unzip_dataset(data, per_target=False)[0], boxes)
        if boxes is not None
        else unzip_dataset(data, per_target=per_target)
    )

    # Build description for progress bar
    calculator_names = [c[0].__name__.removesuffix("Calculator") for c in calculators]
    _logger.debug("Using calculators: %s", calculator_names)

    with PoolWrapper(processes=get_max_processes()) as p:
        for result in p.imap_unordered(
            partial(
                _unpack,
                calculators=calculators,
                per_image=per_image,
                per_target=per_target,
                per_channel=per_channel,
            ),
            _enumerate(images, boxes),
        ):
            _aggregate(result, source_indices, aggregated_stats, object_count, invalid_box_count, warning_list)
            image_count += 1

            if progress_callback:
                progress_callback(image_count, total=total_images)

    for w in warning_list:
        _logger.warning(w)

    _logger.debug("Sorting %d source indices and %d stats", len(source_indices), len(aggregated_stats))
    sorted_source_indices, sorted_aggregated_stats = _sort(source_indices, aggregated_stats)

    total_boxes = sum(object_count.values())
    total_invalid = sum(invalid_box_count.values())
    _logger.info(
        "Calculate complete: %d images processed, %d total boxes (%d invalid), %d stats computed",
        image_count,
        total_boxes,
        total_invalid,
        len(sorted_aggregated_stats),
    )

    return CalculationResult(
        source_index=sorted_source_indices,
        object_count=[object_count.get(i, 0) for i in range(image_count)],
        invalid_box_count=[invalid_box_count.get(i, 0) for i in range(image_count)],
        image_count=image_count,
        stats=sorted_aggregated_stats,
    )
