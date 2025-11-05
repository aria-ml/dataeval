from __future__ import annotations

__all__ = []

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from dataeval.config import EPSILON
from dataeval.core._calculate import CalculationResult
from dataeval.types import SourceIndex

SOURCE_INDEX_KEY = "source_index"

OverrideFunctionMap: TypeAlias = Mapping[
    str, Callable[[Mapping[str, NDArray[Any]], Mapping[str, NDArray[Any]]], NDArray[Any]]
]
"""
Mapping of stat names to custom ratio calculation functions.
Each function takes (box_stats_dict, img_stats_dict) and returns calculated ratio for that stat.

Example
-------
override_map = {
    "offset_x": lambda box, img: box["offset_x"] / (img["width"] + EPSILON),
    "channels": lambda box, img: box["channels"],
}
"""


def _default_ratio_map() -> OverrideFunctionMap:
    """
    Default override mappings for specific statistics that need special ratio calculations.

    Returns
    -------
    dict[str, Callable]
        Mapping of stat names to custom ratio calculation functions.
        Each function takes (box_stats_dict, img_stats_dict) and returns calculated ratio.
    """
    return {
        # Normalize offsets by image dimensions
        "offset_x": lambda box, img: box["offset_x"] / (img["width"] + EPSILON),
        "offset_y": lambda box, img: box["offset_y"] / (img["height"] + EPSILON),
        # Keep these values unchanged from box stats
        "channels": lambda box, img: box["channels"],
        "depth": lambda box, img: box["depth"],
        # Hash stats should be kept as-is (they're strings, not numeric)
        "xxhash": lambda box, img: box["xxhash"],
        "pchash": lambda box, img: box["pchash"],
        # Normalize distance to center by half-diagonal of image
        "distance_center": lambda box, img: box["distance_center"]
        / (np.sqrt(np.square(img["width"]) + np.square(img["height"])) / 2 + EPSILON),
        # Normalize distance to edge by the relevant dimension (width or height)
        "distance_edge": lambda box, img: box["distance_edge"]
        / (
            (
                img["width"]
                if np.min([np.abs(box["offset_x"]), np.abs((box["width"] + box["offset_x"]) - img["width"])])
                < np.min([np.abs(box["offset_y"]), np.abs((box["height"] + box["offset_y"]) - img["height"])])
                else img["height"]
            )
            + EPSILON
        ),
    }


def _build_image_lookup(source_indices: Sequence[SourceIndex]) -> dict[tuple[int, int | None], int]:
    """
    Build a lookup table mapping (image_index, channel_index) to array index.

    Parameters
    ----------
    source_indices : Sequence[SourceIndex]
        Sequence of source indices from calculate() output
    calculation_result : CalculationResult
        Calculation result containing stats dictionary

    Returns
    -------
    dict[tuple[int, int | None], int]
        Lookup table where key is (image_index, channel_index) and value is the array index
    """
    lookup: dict[tuple[int, int | None], int] = {}

    for idx, source_idx in enumerate(source_indices):
        # Only process image-level entries (box=None)
        if source_idx.box is None:
            key = (source_idx.image, source_idx.channel)
            lookup[key] = idx

    return lookup


def _calculate_ratio_for_stat(
    stat_name: str,
    box_value: Any,
    img_value: Any,
    override_map: OverrideFunctionMap,
    box_stats_context: dict[str, NDArray[Any]],
    img_stats_context: dict[str, NDArray[Any]],
) -> Any:
    """
    Calculate ratio for a single statistic value.

    Parameters
    ----------
    stat_name : str
        Name of the statistic being calculated
    box_value : Any
        The box-level statistic value (scalar from NDArray)
    img_value : Any
        The corresponding image-level statistic value (scalar from NDArray)
    override_map : dict
        Custom ratio calculation functions
    box_stats_context : dict
        Full box stats dict (for override functions that need context)
    img_stats_context : dict
        Full image stats dict (for override functions that need context)

    Returns
    -------
    Any
        The calculated ratio value
    """
    if stat_name in override_map:
        # Use custom calculation
        return override_map[stat_name](box_stats_context, img_stats_context)
    # Default: simple division with error handling for non-numeric types
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            box_arr = np.asarray(box_value)
            img_arr = np.asarray(img_value)

            # Check if values are numeric
            if not np.issubdtype(box_arr.dtype, np.number) or not np.issubdtype(img_arr.dtype, np.number):
                # For non-numeric types, just return box value
                return box_value

            result = box_arr / (img_arr + EPSILON)
            # Convert numpy scalars to native Python types for consistency
            if isinstance(result, np.ndarray) and result.ndim == 0:
                return result.item()
            return result
    except (TypeError, ValueError):
        # If division fails for any reason, return box value as-is
        return box_value


def _validate_separate_inputs(
    stats_output: CalculationResult,
    box_stats_output: CalculationResult,
) -> tuple[Sequence[SourceIndex], Sequence[SourceIndex]]:
    """
    Validate that separate image and box stats outputs are compatible.

    Returns
    -------
    tuple[Sequence[SourceIndex], Sequence[SourceIndex]]
        Image source indices and box source indices
    """
    # Validate compatibility
    if stats_output["image_count"] != box_stats_output["image_count"]:
        raise ValueError(
            f"Image count mismatch: stats_output has {stats_output['image_count']} images, "
            f"but box_stats_output has {box_stats_output['image_count']} images."
        )

    # Validate that stats_output has only image entries
    img_source_indices: Sequence[SourceIndex] = stats_output[SOURCE_INDEX_KEY]
    if any(si.box is not None for si in img_source_indices):
        raise ValueError(
            "When using box_stats_output parameter, stats_output should contain only "
            "image-level statistics (per_image=True, per_box=False). "
            f"Found {sum(1 for si in img_source_indices if si.box is not None)} box entries."
        )

    # Validate that box_stats_output has only box entries
    box_source_indices: Sequence[SourceIndex] = box_stats_output[SOURCE_INDEX_KEY]
    if any(si.box is None for si in box_source_indices):
        raise ValueError(
            "When using box_stats_output parameter, it should contain only "
            "box-level statistics (per_image=False, per_box=True). "
            f"Found {sum(1 for si in box_source_indices if si.box is None)} image entries."
        )

    # Validate channel compatibility
    img_has_channels = any(si.channel is not None for si in img_source_indices)
    box_has_channels = any(si.channel is not None for si in box_source_indices)
    if img_has_channels != box_has_channels:
        raise ValueError(
            "Channel mismatch: Both stats_output and box_stats_output must have matching "
            "per_channel settings (both True or both False)."
        )

    # Validate that stats dictionaries have overlapping keys
    img_stats_keys = set(stats_output["stats"].keys())
    box_stats_keys = set(box_stats_output["stats"].keys())
    overlapping_keys = img_stats_keys & box_stats_keys

    if not overlapping_keys:
        raise ValueError(
            "No overlapping statistics found between stats_output and box_stats_output. "
            f"stats_output has keys: {sorted(img_stats_keys)}, "
            f"box_stats_output has keys: {sorted(box_stats_keys)}. "
            "Ensure both outputs were computed with the same statistics flags."
        )

    return img_source_indices, box_source_indices


def _validate_unified_input(source_indices: Sequence[SourceIndex]) -> None:
    """Validate that unified stats output contains both image and box entries."""
    has_image_entries = any(si.box is None for si in source_indices)
    has_box_entries = any(si.box is not None for si in source_indices)

    if not has_image_entries:
        raise ValueError(
            "stats_output must contain image-level statistics (entries with box=None). "
            "Ensure per_image=True when calling calculate(), or provide box_stats_output parameter."
        )

    if not has_box_entries:
        raise ValueError(
            "stats_output must contain box-level statistics (entries with box!=None). "
            "Ensure per_box=True and boxes are provided when calling calculate(), "
            "or provide box_stats_output parameter."
        )


def calculate_ratios(
    stats_output: CalculationResult,
    *,
    box_stats_output: CalculationResult | None = None,
    override_map: OverrideFunctionMap | None = None,
) -> CalculationResult:
    """
    Calculate box-to-image ratios from calculate() output.

    This function supports two usage patterns:

    1. **Unified input**: Pass a single stats_output containing both
    image and box statistics (from calculate() with per_image=True, per_box=True).

    2. **Separate inputs**: Pass image stats as stats_output and box
    stats as box_stats_output (useful when migrating from boxratiostats()).

    Parameters
    ----------
    stats_output : CalculationResult
        Either:
        - Output from calculate() with both per_image=True and per_box=True (unified), OR
        - Output from calculate() with per_image=True, per_box=False (if box_stats_output provided)
    box_stats_output : CalculationResult | None, optional
        Output from calculate() with per_image=False and per_box=True.
        When provided, stats_output is treated as image-only stats.
        Default is None (use unified input from stats_output).
    override_map : OverrideFunctionMap | None, optional
        Optional custom ratio calculations for specific stat keys.
        Function signature: (box_stats_dict, img_stats_dict) -> ratio_value
        If None, uses default override map for common statistics.

    Returns
    -------
    CalculationResult
        Dictionary with same structure as calculate() output, including:
        - source_index: Sequence of SourceIndex objects for box-level entries only
        - object_count: Sequence of object counts per image
        - invalid_box_count: Sequence of invalid box counts per image
        - image_count: Total number of images processed
        - stats: Mapping of statistic names to ratio values (box stats as ratios of image stats)

    Raises
    ------
    ValueError
        If inputs don't contain the required image and box statistics, or if
        the two inputs are incompatible (different image counts, mismatched channels).
    KeyError
        If stats_output doesn't contain required 'source_index' key.

    Notes
    -----
    - Only processes entries where source_index.box is not None
    - For each box, finds its corresponding image stats (box=None, same image and channel index)
    - Applies custom calculations from override_map or defaults to simple division
    - Handles per-channel stats automatically via channel_index matching
    - BASE_ATTRS (source_index, object_count, etc.) are preserved for box entries only

    Examples
    --------
    **Pattern 1: Unified input (recommended)**

    >>> from dataeval.core import calculate, calculate_ratios
    >>> from dataeval.core.flags import ImageStats
    >>>
    >>> # Single call gets both image and box stats
    >>> stats = calculate(images, boxes, stats=ImageStats.DIMENSION, per_image=True, per_box=True)
    >>> ratios = calculate_ratios(stats)
    >>> width_ratios = ratios["stats"]["width"]
    >>> print(["%.2f" % ratio for ratio in width_ratios])  # Box widths as fraction of image width
    ['0.15', '0.08', '0.12', '0.12', '0.17', '0.05', '0.08', '0.36', '0.20', '0.08', '0.03', '0.07', '0.09', '0.12', '0.18']

    **Pattern 2: Separate inputs (backward compatibility)**

    >>> # Separate calls for image and box stats
    >>> img_stats = calculate(images, boxes, stats=ImageStats.DIMENSION, per_image=True, per_box=False)
    >>> box_stats = calculate(images, boxes, stats=ImageStats.DIMENSION, per_image=False, per_box=True)
    >>> ratios = calculate_ratios(img_stats, box_stats_output=box_stats)

    **Custom override map:**

    >>> custom_overrides = {
    ...     "mean": lambda box, img: (box["mean"] - img["mean"]) / (img["std"] + 1e-10),
    ... }
    >>> ratios = calculate_ratios(stats, override_map=custom_overrides)

    **Per-channel statistics:**

    >>> stats = calculate(images, boxes, stats=ImageStats.PIXEL, per_image=True, per_box=True, per_channel=True)
    >>> ratios = calculate_ratios(stats)
    >>> # Ratios are calculated per-channel automatically
    """  # noqa: E501
    # Validate input
    if SOURCE_INDEX_KEY not in stats_output:
        raise KeyError(f"stats_output must contain '{SOURCE_INDEX_KEY}' key from calculate() output")

    # Determine which pattern we're using and validate
    if box_stats_output is not None:
        # Pattern 2: Separate image and box stats
        if SOURCE_INDEX_KEY not in box_stats_output:
            raise KeyError(f"box_stats_output must contain '{SOURCE_INDEX_KEY}' key from calculate() output")

        img_source_indices, box_source_indices = _validate_separate_inputs(stats_output, box_stats_output)
        source_indices_for_lookup = img_source_indices
        source_indices_for_boxes = box_source_indices
        img_calc_result = stats_output
        box_calc_result = box_stats_output
    else:
        # Pattern 1: Unified input
        source_indices: Sequence[SourceIndex] = stats_output[SOURCE_INDEX_KEY]
        _validate_unified_input(source_indices)
        source_indices_for_lookup = source_indices
        source_indices_for_boxes = source_indices
        img_calc_result = stats_output
        box_calc_result = stats_output

    # Use default override map if none provided
    if override_map is None:
        override_map = _default_ratio_map()

    # Build lookup table for image stats (maps (image_idx, channel_idx) -> array index)
    img_lookup = _build_image_lookup(source_indices_for_lookup)

    # Calculate overlapping stats keys for ratio calculation
    overlapping_keys = set(img_calc_result["stats"]) & set(box_calc_result["stats"])

    # Find all box indices and their corresponding image indices
    box_indices: list[int] = []
    img_indices: list[int] = []
    ratio_source_indices: list[SourceIndex] = []

    for box_idx, source_idx in enumerate(source_indices_for_boxes):
        # Only process box entries
        if source_idx.box is None:
            continue

        # Find corresponding image entry
        img_key = (source_idx.image, source_idx.channel)
        if img_key not in img_lookup:
            raise ValueError(
                f"Cannot find image-level stats for box at image={source_idx.image}, "
                f"channel={source_idx.channel}. Ensure both stats_output and box_stats_output "
                f"were computed on the same dataset with matching per_channel settings."
            )

        img_idx = img_lookup[img_key]
        box_indices.append(box_idx)
        img_indices.append(img_idx)
        ratio_source_indices.append(source_idx)

    # Calculate ratios for each statistic using the overlapping keys
    ratio_stats: dict[str, NDArray[Any]] = {}
    for stat_name in overlapping_keys:
        box_stat_values = box_calc_result["stats"][stat_name]
        img_stat_values = img_calc_result["stats"][stat_name]

        ratio_values: list[Any] = []
        for box_idx, img_idx in zip(box_indices, img_indices):
            # Build context dicts for custom override functions
            box_stats: dict[str, NDArray[Any]] = {k: v[box_idx] for k, v in box_calc_result["stats"].items()}
            img_stats: dict[str, NDArray[Any]] = {k: v[img_idx] for k, v in img_calc_result["stats"].items()}

            ratio_value = _calculate_ratio_for_stat(
                stat_name,
                box_stat_values[box_idx],
                img_stat_values[img_idx],
                override_map,
                box_stats,
                img_stats,
            )
            ratio_values.append(ratio_value)

        # Convert ratio values to numpy array - let numpy infer the appropriate dtype
        ratio_stats[stat_name] = np.array(ratio_values)

    # Build CalculationResult dict with proper structure
    result: CalculationResult = {
        "source_index": ratio_source_indices,
        "object_count": box_calc_result["object_count"],
        "invalid_box_count": box_calc_result["invalid_box_count"],
        "image_count": box_calc_result["image_count"],
        "stats": ratio_stats,
    }

    return result
