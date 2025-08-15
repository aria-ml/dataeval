from __future__ import annotations

__all__ = []

import warnings
from abc import abstractmethod
from collections.abc import Iterable, Iterator, Sized
from dataclasses import dataclass
from functools import cached_property, partial
from itertools import zip_longest
from typing import Any

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from dataeval.config import get_max_processes
from dataeval.outputs._stats import SourceIndex
from dataeval.typing import ArrayLike
from dataeval.utils._boundingbox import BoundingBox, BoxLike
from dataeval.utils._image import clip_and_pad, normalize_image_shape, rescale
from dataeval.utils._multiprocessing import PoolWrapper


@dataclass
class ProcessorResult:
    """Result from processing a single image/box combination."""

    source_indices: list[SourceIndex]
    stats: dict[str, list[Any]]


@dataclass
class ProcessorOutput:
    """Output from processing multiple images."""

    results: list[ProcessorResult]
    object_count: int
    invalid_box_count: int
    warnings_list: list[str]


class BaseProcessor:
    def __init__(self, image: NDArray[Any], box: BoundingBox | None) -> None:
        self.raw = image
        self.width: int = image.shape[-1]
        self.height: int = image.shape[-2]
        self.shape: tuple[int, ...] = image.shape

        # Ensure bounding box
        self.box = BoundingBox(0, 0, self.width, self.height, image_shape=image.shape) if box is None else box

    @abstractmethod
    def process(self) -> dict[str, list[Any]]: ...

    @cached_property
    def image(self) -> NDArray[Any]:
        return clip_and_pad(normalize_image_shape(self.raw), self.box.xyxy_int)

    @cached_property
    def scaled(self) -> NDArray[Any]:
        return rescale(self.image)

    @cached_property
    def per_channel(self) -> NDArray[Any]:
        return self.scaled.reshape(self.image.shape[0], -1)


def _collect_processor_stats(
    processors: Iterable[type[BaseProcessor]], image: NDArray[Any], box: BoundingBox | None
) -> list[dict[str, list[Any]]]:
    """Collect stats from all processors."""
    stats_list = []
    for p in processors:
        processor = p(image, box)
        stats_list.append(processor.process())
        del processor
    return stats_list


def _determine_channel_indices(processor_stats: list[dict[str, list[Any]]], num_channels: int) -> list[int | None]:
    """Determine what channel indices are needed based on processor outputs."""
    channel_indices_needed: set[int | None] = set()

    for stats in processor_stats:
        first_stat_values = next(iter(stats.values()))
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
    processor_stats: list[dict[str, list[Any]]], sorted_channels: list[int | None]
) -> dict[str, list[Any]]:
    """Reconcile stats from different processors into a unified structure."""
    num_entries = len(sorted_channels)
    reconciled_stats: dict[str, list[Any]] = {}

    for stats in processor_stats:
        first_stat_values = next(iter(stats.values()))
        num_elements = len(first_stat_values)

        for stat_name, stat_values in stats.items():
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


def _process_single(
    i: int,
    image: NDArray[Any],
    boxes: list[BoundingBox] | None,
    processors: Iterable[type[BaseProcessor]],
) -> ProcessorOutput:
    results: list[ProcessorResult] = []
    box_count = 0
    invalid_box_count = 0
    warnings_list: list[str] = []

    # Determine the number of channels from the image shape
    num_channels = image.shape[-3] if len(image.shape) >= 3 else 1

    for i_b, box in [(None, None)] if boxes is None else enumerate(boxes):
        if box is not None:
            box_count += 1
            if not box.is_clippable():
                invalid_box_count += 1
                warnings_list.append(f"Bounding box [{i}][{i_b}]: {box} for image shape {image.shape} is invalid.")

        # Collect stats from all processors
        processor_stats = _collect_processor_stats(processors, image, box)

        # Determine what channel indices are needed
        sorted_channels = _determine_channel_indices(processor_stats, num_channels)

        # Reconcile stats into unified structure
        reconciled_stats = _reconcile_stats(processor_stats, sorted_channels)

        # Build index lists
        channel_indices = sorted_channels
        source = [SourceIndex(i, i_b if box is not None else None, c) for c in channel_indices]

        results.append(ProcessorResult(source_indices=source, stats=reconciled_stats))

    return ProcessorOutput(results, box_count, invalid_box_count, warnings_list)


def _unpack(
    args: tuple[int, NDArray[Any], list[BoundingBox] | None],
    processors: Iterable[type[BaseProcessor]],
) -> ProcessorOutput:
    return _process_single(*args, processors)


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
    result: ProcessorOutput,
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


def process(
    images: Iterable[ArrayLike],
    boxes: Iterable[Iterable[BoxLike] | None] | None,
    processors: type[BaseProcessor] | Iterable[type[BaseProcessor]],
) -> dict[str, Any]:
    """
    Compute specified statistics on a set of images.

    Parameters
    ----------
    images : Iterable[NDArray[Any]]
        An iterable of images to compute statistics on.
    boxes : Iterable[Iterable[BoxLike] | None] | None
        Optional bounding boxes for each image. If None, processes entire images.
    processors : Iterable[type[BaseProcessor]]
        An iterable of processor classes that calculate statistics.

    Returns
    -------
    dict[str, Any]
        Dictionary containing computed statistics and metadata including:
        - Individual statistics as computed by processors
        - 'image_index': List of image indices
        - 'box_index': List of box indices (None for full images)
        - 'channel_index': List of channel indices (None for single-channel stats)
        - 'object_count': List of object counts per image
        - 'invalid_box_count': List of invalid box counts per image
        - 'image_count': Total number of images processed

        Output is sorted by (image_index, box_index, channel_index) ascending,
        with None values appearing before 0.
    """
    source_indices: list[SourceIndex] = []
    aggregated_stats: dict[str, list[Any]] = {}
    object_count: dict[int, int] = {}
    invalid_box_count: dict[int, int] = {}
    image_count: int = 0
    warning_list: list[str] = []

    processors = processors if isinstance(processors, Iterable) else (processors,)

    with PoolWrapper(processes=get_max_processes()) as p:
        for result in tqdm(
            p.imap_unordered(
                partial(_unpack, processors=processors),
                _enumerate(images, boxes),
            ),
            total=len(images) if isinstance(images, Sized) else None,
            desc=f"Processing images for {', '.join([p.__name__.removesuffix('Processor') for p in processors])}",
        ):
            _aggregate(result, source_indices, aggregated_stats, object_count, invalid_box_count, warning_list)
            image_count += 1

    for w in warning_list:
        warnings.warn(w, UserWarning)

    sorted_source_indices, sorted_aggregated_stats = _sort(source_indices, aggregated_stats)

    return sorted_aggregated_stats | {
        "source_index": sorted_source_indices,
        "object_count": [object_count.get(i, 0) for i in range(image_count)],
        "invalid_box_count": [invalid_box_count.get(i, 0) for i in range(image_count)],
        "image_count": image_count,
    }
