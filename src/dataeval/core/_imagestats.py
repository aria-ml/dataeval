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
from scipy.stats import entropy, kurtosis, skew
from tqdm.auto import tqdm

from dataeval.config import EPSILON, get_max_processes
from dataeval.core._hash import pchash, xxhash
from dataeval.outputs._stats import SourceIndex
from dataeval.typing import ArrayLike
from dataeval.utils._boundingbox import BoundingBox, BoxLike
from dataeval.utils._image import (
    clip_and_pad,
    edge_filter,
    get_bitdepth,
    normalize_image_shape,
    rescale,
)
from dataeval.utils._multiprocessing import PoolWrapper

QUARTILES = (0, 25, 50, 75, 100)


@dataclass
class ProcessorResult:
    """Result from processing a single image/box combination."""

    stats: dict[str, list[Any]]
    image_indices: list[int]
    box_indices: list[int | None]
    channel_indices: list[int | None]
    warning: str | None = None


@dataclass
class StatsProcessorOutput:
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


class PixelStatsProcessor(BaseProcessor):
    @cached_property
    def histogram(self) -> NDArray[np.float64]:
        return np.histogram(self.scaled, bins=256, range=(0, 1))[0]

    def process(self) -> dict[str, list[Any]]:
        return {
            "mean": [float(np.nanmean(self.scaled))],
            "std": [float(np.nanstd(self.scaled))],
            "var": [float(np.nanvar(self.scaled))],
            "skew": [float(skew(self.scaled.ravel(), nan_policy="omit"))],
            "kurtosis": [float(kurtosis(self.scaled.ravel(), nan_policy="omit"))],
            "entropy": [float(entropy(self.histogram))],
            "missing": [float(np.count_nonzero(np.isnan(self.image)) / np.prod(self.shape[-2:]))],
            "zeros": [float(np.count_nonzero(np.sum(self.image, axis=0) == 0) / np.prod(self.shape[-2:]))],
            "histogram": [self.histogram.tolist()],
        }


class VisualStatsProcessor(BaseProcessor):
    @cached_property
    def percentiles(self) -> NDArray[np.float64]:
        return np.nanpercentile(self.scaled, q=QUARTILES).astype(np.float64)

    def process(self) -> dict[str, list[Any]]:
        return {
            "brightness": [float(self.percentiles[1])],
            "contrast": [
                float(np.max(self.percentiles) - np.min(self.percentiles)) / float(np.mean(self.percentiles) + EPSILON)
            ],
            "darkness": [float(self.percentiles[-2])],
            "sharpness": [float(np.nanstd(edge_filter(np.mean(self.image, axis=0))))],
            "percentiles": [self.percentiles.tolist()],
        }


class PixelPerChannelStatsProcessor(BaseProcessor):
    @cached_property
    def histogram(self) -> NDArray[np.float64]:
        return np.apply_along_axis(lambda y: np.histogram(y, bins=256, range=(0, 1))[0], 1, self.per_channel)

    def process(self) -> dict[str, list[Any]]:
        return {
            "mean": np.nanmean(self.per_channel, axis=1).tolist(),
            "std": np.nanstd(self.per_channel, axis=1).tolist(),
            "var": np.nanvar(self.per_channel, axis=1).tolist(),
            "skew": skew(self.per_channel, axis=1, nan_policy="omit").tolist(),
            "kurtosis": kurtosis(self.per_channel, axis=1, nan_policy="omit").tolist(),
            "entropy": np.asarray(entropy(self.histogram, axis=1)).tolist(),
            "missing": (np.count_nonzero(np.isnan(self.image), axis=(1, 2)) / np.prod(self.shape[-2:])).tolist(),
            "zeros": (np.count_nonzero(self.image == 0, axis=(1, 2)) / np.prod(self.shape[-2:])).tolist(),
            "histogram": self.histogram.tolist(),
        }


class VisualPerChannelStatsProcessor(BaseProcessor):
    @cached_property
    def percentiles(self) -> NDArray[np.float64]:
        return np.nanpercentile(self.per_channel, q=QUARTILES, axis=1).T.astype(np.float64)

    def process(self) -> dict[str, list[Any]]:
        return {
            "brightness": self.percentiles[:, 1].tolist(),
            "contrast": (
                (np.max(self.percentiles, axis=1) - np.min(self.percentiles, axis=1))
                / (np.mean(self.percentiles, axis=1) + EPSILON)
            ).tolist(),
            "darkness": self.percentiles[:, -2].tolist(),
            "missing": (np.count_nonzero(np.isnan(self.image), axis=(1, 2)) / np.prod(self.shape[-2:])).tolist(),
            "sharpness": np.nanstd(
                np.vectorize(edge_filter, signature="(m,n)->(m,n)")(self.image), axis=(1, 2)
            ).tolist(),
            "percentiles": self.percentiles.tolist(),
        }


class DimensionStatsProcessor(BaseProcessor):
    def process(self) -> dict[str, list[Any]]:
        return {
            "offset_x": [self.box.x0],
            "offset_y": [self.box.y0],
            "width": [self.box.width],
            "height": [self.box.height],
            "channels": [self.shape[-3]],
            "size": [self.box.width * self.box.height],
            "aspect_ratio": [0.0 if self.box.height == 0 else self.box.width / self.box.height],
            "depth": [get_bitdepth(self.raw).depth],
            "center": [[(self.box.x0 + self.box.x1) / 2, (self.box.y0 + self.box.y1) / 2]],
            "distance_center": [
                float(
                    np.sqrt(
                        np.square(((self.box.x0 + self.box.x1) / 2) - (self.raw.shape[-1] / 2))
                        + np.square(((self.box.y0 + self.box.y1) / 2) - (self.raw.shape[-2] / 2))
                    )
                )
            ],
            "distance_edge": [
                float(
                    np.min(
                        [
                            np.abs(self.box.x0),
                            np.abs(self.box.y0),
                            np.abs(self.box.x1 - self.raw.shape[-1]),
                            np.abs(self.box.y1 - self.raw.shape[-2]),
                        ]
                    )
                )
            ],
            "invalid_box": [not self.box.is_valid()],
        }


class HashStatsProcessor(BaseProcessor):
    def process(self) -> dict[str, list[Any]]:
        return {
            "xxhash": [xxhash(self.raw)],
            "pchash": [pchash(self.raw)],
        }


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
) -> StatsProcessorOutput:
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
        num_entries = len(sorted_channels)
        image_indices = [i] * num_entries
        box_indices = [i_b if box is not None else None] * num_entries
        channel_indices = sorted_channels

        results.append(
            ProcessorResult(
                stats=reconciled_stats,
                image_indices=image_indices,
                box_indices=box_indices,
                channel_indices=channel_indices,
            )
        )

    return StatsProcessorOutput(results, box_count, invalid_box_count, warnings_list)


def _unpack(
    args: tuple[int, NDArray[Any], list[BoundingBox] | None],
    processors: Iterable[type[BaseProcessor]],
) -> StatsProcessorOutput:
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


def _aggregate(
    results: list[ProcessorResult],
) -> tuple[list[int], list[int | None], list[int | None], dict[str, list[Any]]]:
    """Extract indices and aggregate results from ProcessorResult objects."""
    image_indices: list[int] = []
    box_indices: list[int | None] = []
    channel_indices: list[int | None] = []
    aggregated_stats: dict[str, list[Any]] = {}

    for result in results:
        image_indices.extend(result.image_indices)
        box_indices.extend(result.box_indices)
        channel_indices.extend(result.channel_indices)

        for stat_name, stat_values in result.stats.items():
            aggregated_stats.setdefault(stat_name, []).extend(stat_values)

    return image_indices, box_indices, channel_indices, aggregated_stats


def _sort_results(
    image_indices: list[int],
    box_indices: list[int | None],
    channel_indices: list[int | None],
    aggregated_stats: dict[str, list[Any]],
) -> tuple[list[SourceIndex], dict[str, list[Any]]]:
    """Sort results by (image_index, box_index, channel_index) with None < 0."""
    sort_indices = sorted(
        range(len(image_indices)),
        key=lambda i: (
            image_indices[i],
            -1 if box_indices[i] is None else box_indices[i],
            -1 if channel_indices[i] is None else channel_indices[i],
        ),
    )

    # Apply sorting to all parallel arrays
    sorted_image_indices = [image_indices[i] for i in sort_indices]
    sorted_box_indices = [box_indices[i] for i in sort_indices]
    sorted_channel_indices = [channel_indices[i] for i in sort_indices]

    sorted_aggregated_stats: dict[str, list[Any]] = {}
    for stat_name, stat_values in aggregated_stats.items():
        sorted_aggregated_stats[stat_name] = [stat_values[i] for i in sort_indices]

    sorted_source_indices = [
        SourceIndex(ii, bi, ci) for ii, bi, ci in zip(sorted_image_indices, sorted_box_indices, sorted_channel_indices)
    ]

    return sorted_source_indices, sorted_aggregated_stats


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
    image_indices: list[int] = []
    box_indices: list[int | None] = []
    channel_indices: list[int | None] = []
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
            aggregated_results = _aggregate(result.results)

            image_indices.extend(aggregated_results[0])
            box_indices.extend(aggregated_results[1])
            channel_indices.extend(aggregated_results[2])
            for stat_name, stat_values in aggregated_results[3].items():
                aggregated_stats.setdefault(stat_name, []).extend(stat_values)

            if result.results:
                img_idx = result.results[0].image_indices[0]
                object_count[img_idx] = result.object_count
                invalid_box_count[img_idx] = result.invalid_box_count
                warning_list.extend(result.warnings_list)

            image_count += 1

    for w in warning_list:
        warnings.warn(w, UserWarning)

    sorted_source_indices, sorted_aggregated_stats = _sort_results(
        image_indices, box_indices, channel_indices, aggregated_stats
    )

    return sorted_aggregated_stats | {
        "source_index": sorted_source_indices,
        "object_count": [object_count.get(i, 0) for i in range(image_count)],
        "invalid_box_count": [invalid_box_count.get(i, 0) for i in range(image_count)],
        "image_count": image_count,
    }
