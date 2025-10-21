from __future__ import annotations

__all__ = []

import warnings
from collections.abc import Callable, Iterable, Iterator, Sized
from dataclasses import dataclass
from enum import Flag
from functools import cached_property, partial
from itertools import zip_longest
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Import calculators to trigger auto-registration
import dataeval.core._calculators._imagestats  # noqa: F401
from dataeval.config import get_max_processes
from dataeval.core._calculators._registry import CalculatorRegistry
from dataeval.core.flags import ImageStats, resolve_dependencies
from dataeval.outputs._stats import SourceIndex
from dataeval.protocols import ArrayLike
from dataeval.utils._boundingbox import BoundingBox, BoxLike
from dataeval.utils._image import clip_and_pad, normalize_image_shape, rescale
from dataeval.utils._multiprocessing import PoolWrapper
from dataeval.utils._tqdm import tqdm


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
        self.raw = datum
        # Assume image data for now (will be generic in future)
        self.width: int = datum.shape[-1]
        self.height: int = datum.shape[-2]
        self.shape: tuple[int, ...] = datum.shape
        self.per_channel_mode = per_channel

        # Ensure bounding box
        self.box = BoundingBox(0, 0, self.width, self.height, image_shape=datum.shape) if box is None else box

    @cached_property
    def image(self) -> NDArray[Any]:
        return clip_and_pad(normalize_image_shape(self.raw), self.box.xyxy_int)

    @cached_property
    def scaled(self) -> NDArray[Any]:
        return rescale(self.image)

    @cached_property
    def per_channel(self) -> NDArray[Any]:
        return self.scaled.reshape(self.image.shape[0], -1)


def _collect_calculator_stats(
    calculators: Iterable[tuple[type[Any], Flag]],
    datum: NDArray[Any],
    box: BoundingBox | None,
    per_channel: bool,
) -> list[dict[str, list[Any]]]:
    """Collect stats from all calculators."""
    stats_list = []
    processor = CalculatorCache(datum, box, per_channel)
    for calculator_cls, flags in calculators:
        calculator = calculator_cls(datum, processor, per_channel)
        stats_list.append(calculator.compute(flags))
        del calculator
    return stats_list


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
    calculator_output: list[dict[str, list[Any]]], sorted_channels: list[int | None]
) -> dict[str, list[Any]]:
    """Reconcile stats from different processors into a unified structure."""
    num_entries = len(sorted_channels)
    reconciled_stats: dict[str, list[Any]] = {}

    for output in calculator_output:
        first_stat_values = next(iter(output.values()))
        num_elements = len(first_stat_values)

        for stat_name, stat_values in output.items():
            if stat_name not in reconciled_stats:
                reconciled_stats[stat_name] = [None] * num_entries

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
    per_box: bool,
) -> list[tuple[int | None, BoundingBox | None]]:
    """Determine what to process based on per_image and per_box flags."""
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

        if per_box:
            # Add per-box processing
            process_items.extend((i_b, box) for i_b, box in enumerate(boxes))

    return process_items


def _calculate_datum(
    i: int,
    datum: NDArray[Any],
    boxes: list[BoundingBox] | None,
    calculators: Iterable[tuple[type[Any], Flag]],
    per_image: bool,
    per_box: bool,
    per_channel: bool,
) -> CalculatorOutput:
    results: list[CalculatorResult] = []
    box_count = 0
    invalid_box_count = 0
    warnings_list: list[str] = []

    # Determine the number of channels from the datum shape
    num_channels = datum.shape[-3] if len(datum.shape) >= 3 else 1

    # Determine what to process based on per_image and per_box flags
    items = _get_items(boxes, per_image, per_box)

    # Process each item (full image and/or boxes)
    for i_b, box in items:
        if box is not None:
            box_count += 1
            if not box.is_clippable():
                invalid_box_count += 1
                warnings_list.append(f"Bounding box [{i}][{i_b}]: {box} for datum shape {datum.shape} is invalid.")

        # Collect stats from all calculators
        calculator_stats = _collect_calculator_stats(calculators, datum, box, per_channel)

        # Determine what channel indices are needed
        sorted_channels = _determine_channel_indices(calculator_stats, num_channels)

        # Reconcile stats into unified structure
        reconciled_stats = _reconcile_stats(calculator_stats, sorted_channels)

        # Build index lists
        channel_indices = sorted_channels
        source = [SourceIndex(i, i_b if box is not None else None, c) for c in channel_indices]

        results.append(CalculatorResult(source_indices=source, stats=reconciled_stats))

    return CalculatorOutput(results, box_count, invalid_box_count, warnings_list)


def _unpack(
    args: tuple[int, NDArray[Any], list[BoundingBox] | None],
    calculators: Iterable[tuple[type[Any], Flag]],
    per_image: bool,
    per_box: bool,
    per_channel: bool,
) -> CalculatorOutput:
    return _calculate_datum(*args, calculators, per_image, per_box, per_channel)


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
            bboxes = [BoundingBox.from_boxlike(b, image_shape=np_image.shape) for b in box or ()]
            yield i, np_image, bboxes


def _sort(
    source_indices: list[SourceIndex],
    aggregated_stats: dict[str, list[Any]],
) -> tuple[list[SourceIndex], dict[str, list[Any]]]:
    """Sort results by (image_index, box_index, channel_index) with None < 0."""
    sort_indices = sorted(
        range(len(source_indices)),
        key=lambda i: (
            source_indices[i].image,
            -1 if source_indices[i].box is None else source_indices[i].box,
            -1 if source_indices[i].channel is None else source_indices[i].channel,
        ),
    )

    sorted_source_indices: list[SourceIndex] = [source_indices[i] for i in sort_indices]
    sorted_aggregated_stats: dict[str, list[Any]] = {}
    for stat_name, stat_values in aggregated_stats.items():
        sorted_aggregated_stats[stat_name] = [stat_values[i] for i in sort_indices]

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
        img_idx = result.results[0].source_indices[0].image
        object_count[img_idx] = result.object_count
        invalid_box_count[img_idx] = result.invalid_box_count

    warning_list.extend(result.warnings_list)


def calculate(
    images: Iterable[ArrayLike],
    boxes: Iterable[Iterable[BoxLike] | None] | None,
    stats: Flag = ImageStats.ALL,
    *,
    per_image: bool = True,
    per_box: bool = True,
    per_channel: bool = False,
    progress_callback: Callable[[int, int | None], None] | None = None,
) -> dict[str, Any]:
    """
    Compute specified statistics on a set of images.

    Parameters
    ----------
    images : Iterable[ArrayLike]
        An iterable of images to compute statistics on.
    boxes : Iterable[Iterable[BoxLike] | None] | None
        Optional bounding boxes for each image. If None, processes entire images.
    stats : ImageStats, default ImageStats.ALL
        Flags indicating which statistics to compute. Can combine multiple flags
        using bitwise OR (|). Dependencies are resolved automatically.
    per_image : bool, default True
        If True, compute statistics for entire images. When boxes are provided
        and per_image=True, statistics are computed for both the full image and
        each box (if per_box=True).
    per_box : bool, default True
        If True and boxes are provided, compute statistics for each bounding box.
        Has no effect when boxes is None. At least one of per_image or per_box
        must be True.
    per_channel : bool, default False
        If True, compute per-channel statistics. If False, statistics are
        aggregated across all channels.
    progress_callback : Callable[[int, int | None], None] | None
        Optional callback for progress updates.

    Returns
    -------
    dict[str, Any]
        Dictionary containing computed statistics and metadata including:
        - Individual statistics as computed by processors
        - 'source_index': List of SourceIndex objects with image/box/channel info
        - 'object_count': List of object counts per image
        - 'invalid_box_count': List of invalid box counts per image
        - 'image_count': Total number of images processed

        Output is sorted by (image_index, box_index, channel_index) ascending,
        with None values appearing before 0.

    Examples
    --------
    Compute all statistics:

    >>> from dataeval.core.flags import ImageStats
    >>> stats = calculate(images, boxes)

    Compute specific statistics:

    >>> stats = calculate(images, boxes, stats=ImageStats.PIXEL_MEAN | ImageStats.VISUAL_BRIGHTNESS)

    Use convenience groups:

    >>> stats = calculate(images, boxes, stats=ImageStats.PIXEL | ImageStats.VISUAL)
    >>> stats = calculate(images, boxes, stats=ImageStats.PIXEL_BASIC, per_channel=True)

    Compute statistics only for bounding boxes (not full images):

    >>> stats = calculate(images, boxes, per_image=False, per_box=True)

    Compute statistics for full images only (ignore boxes):

    >>> stats = calculate(images, boxes, per_image=True, per_box=False)

    Compute statistics for both full images and boxes with per-channel breakdown:

    >>> stats = calculate(images, boxes, per_image=True, per_box=True, per_channel=True)
    """
    source_indices: list[SourceIndex] = []
    aggregated_stats: dict[str, list[Any]] = {}
    object_count: dict[int, int] = {}
    invalid_box_count: dict[int, int] = {}
    image_count: int = 0
    warning_list: list[str] = []

    # `per_box` is True only if boxes are provided
    per_box = per_box and boxes is not None

    # Validate parameters
    if not per_image and not per_box:
        raise ValueError("At least one of 'per_image' or 'per_box' must be True")

    # Resolve dependencies
    stats = resolve_dependencies(stats)

    # Get calculators from registry based on flags
    calculators = CalculatorRegistry.get_calculators(stats)

    total_images = len(images) if isinstance(images, Sized) else None

    # Build description for progress bar
    calculator_names = [c[0].__name__.removesuffix("Calculator") for c in calculators]
    desc = f"Processing images for {', '.join(calculator_names)}"

    with PoolWrapper(processes=get_max_processes()) as p:
        for result in tqdm(
            p.imap_unordered(
                partial(
                    _unpack,
                    calculators=calculators,
                    per_image=per_image,
                    per_box=per_box,
                    per_channel=per_channel,
                ),
                _enumerate(images, boxes),
            ),
            total=total_images,
            desc=desc,
        ):
            _aggregate(result, source_indices, aggregated_stats, object_count, invalid_box_count, warning_list)
            image_count += 1

            if progress_callback:
                progress_callback(image_count, total_images)

    for w in warning_list:
        warnings.warn(w, UserWarning)

    sorted_source_indices, sorted_aggregated_stats = _sort(source_indices, aggregated_stats)

    return sorted_aggregated_stats | {
        "source_index": sorted_source_indices,
        "object_count": [object_count.get(i, 0) for i in range(image_count)],
        "invalid_box_count": [invalid_box_count.get(i, 0) for i in range(image_count)],
        "image_count": image_count,
    }
