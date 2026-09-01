__all__ = []

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from dataeval._log import get_logger
from dataeval.core._compute_stats import BACKGROUND_PREFIX, StatsResult, require_same_stat_names
from dataeval.types import FactorLevel, SourceIndex
from dataeval.utils._internal import EPSILON

_logger = get_logger(__name__)

SOURCE_INDEX_KEY = "source_index"

OverrideFunctionMap: TypeAlias = Mapping[
    str,
    Callable[[Mapping[str, NDArray[Any]], Mapping[str, NDArray[Any]]], NDArray[Any]],
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
        "aspect_ratio": lambda box, _img: box["aspect_ratio"],
        "channels": lambda box, _img: box["channels"],
        "depth": lambda box, _img: box["depth"],
        # Hash stats should be kept as-is (they're strings, not numeric)
        "xxhash": lambda box, _img: box["xxhash"],
        "phash": lambda box, _img: box["phash"],
        "dhash": lambda box, _img: box["dhash"],
        # Normalize distance to center by half-diagonal of image
        "distance_center": lambda box, img: (
            box["distance_center"] / (np.sqrt(np.square(img["width"]) + np.square(img["height"])) / 2 + EPSILON)
        ),
        # Normalize distance to edge by the relevant dimension (width or height)
        "distance_edge": lambda box, img: (
            box["distance_edge"]
            / (
                (
                    img["width"]
                    if np.min([np.abs(box["offset_x"]), np.abs((box["width"] + box["offset_x"]) - img["width"])])
                    < np.min([np.abs(box["offset_y"]), np.abs((box["height"] + box["offset_y"]) - img["height"])])
                    else img["height"]
                )
                + EPSILON
            )
        ),
    }


# What each side of a ratio may state. An unkeyed address names an item's own row, which
# is ``sequence`` on a tracking task and ``unit`` on an image one; a keyed address names a
# label, which is ``instance`` on every task. Any other pairing either contradicts the key
# — a keyed ``unit`` is a video frame, not a box — or is malformed, such as an ``instance``
# with nothing to say which instance. Ordered as the canonical hierarchy orders them, so a
# message reads coarsest-first without a second pass to recover an order a set threw away.
_ITEM_LEVELS: tuple[FactorLevel, ...] = ("sequence", "unit")
_LABEL_LEVELS: tuple[FactorLevel, ...] = ("instance",)


def _names_the_item(source_index: SourceIndex) -> bool:
    """Whether an address names an item's own row rather than one of its labels.

    Read off the key rather than inferred from a missing field. An unkeyed address is the
    item level and a keyed one is the label level, which are the two levels a ratio is
    defined between; `_reject_disagreeing_levels` has already turned away any address whose
    stated level contradicts that reading.
    """
    return source_index.key is None


def _reject_disagreeing_levels(source_indices: Sequence[SourceIndex], argument: str) -> None:
    """Reject addresses whose stated level contradicts what their key says.

    A ratio divides a box's value by the whole image's — the label level over the item
    level — and `_names_the_item` reads which of the two an address is by whether it
    carries a key. A stated level is accepted when it agrees with that reading, so the
    fully explicit spelling of an ordinary result goes through unchanged.

    It is the disagreements that have to be caught rather than ignored. A keyed ``unit``
    address is a video frame, and left to fall through it would be taken for a box and
    divided by whatever image row shared its item — a wrong number rather than an error.

    One side may only be spelled one way, for the same reason. ``'sequence'`` and
    ``'unit'`` both name an item's own row, so an input carrying two spellings of it
    carries two item rows per item, and `_build_image_lookup` keeps whichever came last —
    every box of that item then divided by a denominator nothing chose.

    Raises
    ------
    ValueError
        When any address states a level its key contradicts, or when one side of the ratio
        is addressed at more than one level.
    """
    offenders = sorted({
        (si.level, si.key is not None)
        for si in source_indices
        if si.level is not None and si.level not in (_LABEL_LEVELS if si.key is not None else _ITEM_LEVELS)
    })
    if offenders:
        named = ", ".join(f"{level!r}{' with a key' if keyed else ' with no key'}" for level, keyed in offenders)
        raise ValueError(
            f"{argument} contains addresses at {named}, and a ratio is only defined between an item "
            f"and its targets. An item's own row is addressed unkeyed at {_join(_ITEM_LEVELS)}, and one "
            f"of its targets by a key at {_join(_LABEL_LEVELS)}.",
        )

    for keyed, role in ((False, "an item's own row"), (True, "one of its targets")):
        spellings = {si.level for si in source_indices if (si.key is not None) is keyed}
        if len(spellings) > 1:
            named = ", ".join(sorted(repr(level) for level in spellings))
            raise ValueError(
                f"{argument} addresses {role} at {named}, and a ratio has one row per item on each "
                "side. Two spellings of one address are two rows, and only one of them can be the "
                "one divided by. Address each side at a single level, or leave the level unstated.",
            )


def _join(levels: Sequence[FactorLevel]) -> str:
    """Render the levels one side of a ratio may state, for a message."""
    return " or ".join(repr(level) for level in levels)


def _build_image_lookup(source_indices: Sequence[SourceIndex]) -> dict[int, int]:
    """
    Build a lookup table mapping item_index to array index.

    Parameters
    ----------
    source_indices : Sequence[SourceIndex]
        Sequence of source indices from compute_stats() output

    Returns
    -------
    dict[int, int]
        Lookup table where key is item_index and value is the array index
    """
    lookup: dict[int, int] = {}

    for idx, source_idx in enumerate(source_indices):
        # Only process image-level entries (box=None)
        if _names_the_item(source_idx):
            lookup[source_idx.item] = idx

    return lookup


def _resolve_override(stat_name: str, override_map: OverrideFunctionMap) -> Any:
    """Find the override for a statistic, band-group prefix and all.

    The map is keyed by bare statistic name, but a named channel group produces
    ``<group>_<statistic>`` — so a lookup on the literal column name misses every band
    column and falls through to plain division. For `depth` that is silently meaningless:
    the band range is anchored on the whole datum, so image and box report the same depth
    and the ratio comes back as 1.0 under a name that reads like a measurement.

    Matched on the suffix after the first underscore rather than by stripping a known
    prefix, since the group names are a call-site argument this function never sees. A
    group cannot be named after a statistic — `_resolve_channel_groups` rejects that at the
    call — so the split is unambiguous.
    """
    if stat_name in override_map:
        return override_map[stat_name]
    _, _, suffix = stat_name.partition("_")
    while suffix:
        if suffix in override_map:
            return override_map[suffix]
        _, _, suffix = suffix.partition("_")
    return None


def _calculate_ratio_for_stat(  # noqa: C901
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
    override = _resolve_override(stat_name, override_map)
    if override is not None:
        # Use custom calculation
        return override(box_stats_context, img_stats_context)
    # Default: simple division with error handling for non-numeric types
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            box_arr = np.asarray(box_value)
            img_arr = np.asarray(img_value)

            # Check if values are numeric
            if not np.issubdtype(box_arr.dtype, np.number) or not np.issubdtype(img_arr.dtype, np.number):
                # For non-numeric types, just return box value
                return box_value

            # Upscale to float64 for calculation to avoid precision issues
            result = box_arr.astype(np.float64) / (img_arr.astype(np.float64) + EPSILON)
            if np.issubdtype(result.dtype, np.floating):
                result = result.astype(np.float32)

            # Convert numpy scalars to native Python types for consistency
            if isinstance(result, np.ndarray) and result.ndim == 0:
                return result.item()
            return result
    except (TypeError, ValueError):
        # If division fails for any reason, return box value as-is
        return box_value


def _validate_separate_inputs(
    stats_output: StatsResult,
    target_stats_output: StatsResult,
) -> tuple[Sequence[SourceIndex], Sequence[SourceIndex]]:
    """
    Validate that separate image and box stats outputs are compatible.

    Returns
    -------
    tuple[Sequence[SourceIndex], Sequence[SourceIndex]]
        Image source indices and box source indices
    """
    # Validate compatibility
    if stats_output["image_count"] != target_stats_output["image_count"]:
        raise ValueError(
            f"Image count mismatch: stats_output has {stats_output['image_count']} images, "
            f"but target_stats_output has {target_stats_output['image_count']} images.",
        )

    # Validate that stats_output has only image entries
    img_source_indices: Sequence[SourceIndex] = stats_output[SOURCE_INDEX_KEY]
    box_source_indices: Sequence[SourceIndex] = target_stats_output[SOURCE_INDEX_KEY]
    _reject_disagreeing_levels(img_source_indices, "stats_output")
    _reject_disagreeing_levels(box_source_indices, "target_stats_output")

    if any(not _names_the_item(si) for si in img_source_indices):
        raise ValueError(
            "When using target_stats_output parameter, stats_output should contain only "
            "image-level statistics (per_image=True, per_target=False). "
            f"Found {sum(1 for si in img_source_indices if not _names_the_item(si))} box entries.",
        )

    # Validate that target_stats_output has only box entries
    if any(_names_the_item(si) for si in box_source_indices):
        raise ValueError(
            "When using target_stats_output parameter, it should contain only "
            "box-level statistics (per_image=False, per_target=True). "
            f"Found {sum(1 for si in box_source_indices if _names_the_item(si))} image entries.",
        )

    # Validate that the two inputs measure the same things. A ratio pairs a box column
    # against the image column of the same name, so a name on one side and not the other
    # has nothing to divide by — and intersecting silently, as this once did, drops it
    # with no warning at all. Two runs given different `channels=` mappings are what makes
    # that reachable: both sides carry the always-on unnamed view, so the intersection is
    # never empty and the band columns simply vanish from the result.
    #
    # Compared over `_ratio_keys` rather than the raw names, since background columns
    # legitimately exist on the image side alone.
    img_ratio_keys = set(_ratio_keys(stats_output["stats"]))
    box_ratio_keys = set(_ratio_keys(target_stats_output["stats"]))

    require_same_stat_names(
        img_ratio_keys,
        box_ratio_keys,
        left_label="stats_output",
        right_label="target_stats_output",
        summary="Statistic mismatch between stats_output and target_stats_output",
        remedy=(
            "A ratio pairs columns of the same name, so these have no partner to divide "
            "by. Compute both with the same `stats` flags and the same `channels` mapping."
        ),
    )

    if not img_ratio_keys:
        raise ValueError(
            "No statistics to take a ratio of. Both stats_output and target_stats_output "
            f"hold only background statistics: {sorted(stats_output['stats'])}. "
            "Background statistics exist on the image rows alone, so a box has no "
            "counterpart to divide against.",
        )

    return img_source_indices, box_source_indices


def _ratio_keys(stats: Mapping[str, NDArray[Any]]) -> list[str]:
    """Return the statistic names a ratio can be taken of.

    Background statistics are excluded. They exist only on the image rows — a box has no
    background of its own — so pairing one against "its" box value would divide a real
    number by the null standing in for a measurement that was never made, and produce a
    column of NaN under a name that reads like a result.
    """
    return [name for name in stats if not name.startswith(BACKGROUND_PREFIX)]


def _measured_anywhere(values: NDArray[Any], positions: NDArray[np.intp]) -> bool:
    """Whether any of ``positions`` holds a real value rather than a placeholder."""
    if len(positions) == 0:
        return False
    taken = np.asarray(values)[positions]
    if taken.dtype.kind in "fc":
        return bool(np.isfinite(taken).any())
    if taken.dtype.kind == "O":
        return any(not (isinstance(v, float) and np.isnan(v)) for v in taken.ravel())
    # Integer, boolean and fixed-width string arrays have no placeholder to hold.
    return True


def _validate_unified_input(source_indices: Sequence[SourceIndex], stats: Mapping[str, NDArray[Any]]) -> None:
    """Validate that unified stats output contains both image and box entries."""
    _reject_disagreeing_levels(source_indices, "stats_output")
    has_image_entries = any(_names_the_item(si) for si in source_indices)
    has_target_entries = any(not _names_the_item(si) for si in source_indices)

    # `per_background=True` puts an item-level row on every image whether or not
    # `per_image` asked for one, because that row is where a background value lives. Such
    # a row carries background statistics and a null for every other name, so counting
    # rows alone would read a background-only run as having image-level statistics and
    # return a table of NaN instead of saying what was missing. Ask what the rows hold.
    image_positions = np.array([i for i, si in enumerate(source_indices) if _names_the_item(si)], dtype=np.intp)
    ratio_keys = _ratio_keys(stats)
    if has_image_entries and not any(_measured_anywhere(stats[name], image_positions) for name in ratio_keys):
        raise ValueError(
            "stats_output has image-level entries, but every statistic on them is null — the rows "
            "carry background values only. This is what compute_stats(per_image=False, "
            "per_background=True) produces: the background needs an image-level row to live on, but "
            "no image-level statistics were computed to fill it. A ratio needs an image-level value "
            "to divide each box's value by, so pass per_image=True as well.",
        )

    if not has_image_entries:
        raise ValueError(
            "stats_output must contain image-level statistics (entries with box=None). "
            "Ensure per_image=True when calling compute_stats(), or provide target_stats_output parameter.",
        )

    if not has_target_entries:
        raise ValueError(
            "stats_output must contain box-level statistics (entries with box!=None). "
            "Ensure per_target=True and boxes are provided when calling compute_stats(), "
            "or provide target_stats_output parameter.",
        )


def compute_ratios(  # noqa: C901
    stats_output: StatsResult,
    *,
    target_stats_output: StatsResult | None = None,
    override_map: OverrideFunctionMap | None = None,
) -> StatsResult:
    """
    Compute box-to-image ratios from compute_stats() output.

    This function supports two usage patterns:

    1. **Unified input**: Pass a single stats_output containing both
    image and box statistics (from compute_stats() with per_image=True, per_target=True).

    2. **Separate inputs**: Pass image stats as stats_output and box
    stats as target_stats_output (useful when migrating from boxratiostats()).

    Parameters
    ----------
    stats_output : StatsResult
        Either:

        - Output from compute_stats() with both per_image=True and per_target=True (unified), OR
        - Output from compute_stats() with per_image=True, per_target=False (if target_stats_output provided)
    target_stats_output : StatsResult | None, optional
        Output from compute_stats() with per_image=False and per_target=True.
        When provided, stats_output is treated as image-only stats.
        Default is None (use unified input from stats_output).
    override_map : OverrideFunctionMap | None, optional
        Optional custom ratio calculations for specific stat keys.

        Function signature: `(box_stats_dict, img_stats_dict) -> ratio_value`

        If None, uses default override map for common statistics.

    Returns
    -------
    StatsResult
        Dictionary with same structure as compute_stats() output, including:

        - source_index: Sequence[SourceIndex] - SourceIndex objects with image/box info
        - object_count: Sequence[int] - Object counts per image
        - invalid_box_count: Sequence[int] - Invalid box counts per image
        - image_count: int - Total number of images processed
        - stats: Mapping[str, Sequence[Any]] - Mapping of statistic names to sequences of computed values

    Raises
    ------
    ValueError
        If inputs don't contain the required image and box statistics, or if the two
        inputs are incompatible: different image counts, or a different set of statistics
        on each side. The last covers mismatched ``channels`` mappings, whose band columns
        would otherwise be dropped from the result without warning.
    KeyError
        If stats_output doesn't contain required 'source_index' key.

    Notes
    -----
    - Only processes entries where source_index.box is not None
    - For each box, finds its corresponding image stats (box=None, same image)
    - Applies custom calculations from override_map or defaults to simple division
    - Divides each band group column against the image column of the same name, so a
      ``channels=`` mapping yields ``<group>_<statistic>`` ratios beside whichever unprefixed
      ones the run produced — a ``stats`` mapping naming no ``None`` view produces none
    - BASE_ATTRS (source_index, object_count, etc.) are preserved for box entries only

    Examples
    --------
    **Pattern 1: Unified input (recommended)**

    >>> from dataeval.core import compute_stats, compute_ratios
    >>> from dataeval.flags import ImageStats
    >>>
    >>> # Single call gets both image and target stats
    >>> stats = compute_stats(images, boxes=boxes, stats=ImageStats.DIMENSION, per_image=True, per_target=True)
    >>> ratios = compute_ratios(stats)
    >>> ratios["stats"]["width"][:12]
    array([0.25 , 0.203, 0.328, 0.266, 0.234, 0.297, 0.25 , 0.359, 0.297,
           0.234, 0.359, 0.234], dtype=float32)

    **Pattern 2: Separate inputs (backward compatibility)**

    >>> # Separate calls for image and box stats
    >>> img_stats = compute_stats(images, boxes=boxes, stats=ImageStats.DIMENSION, per_image=True, per_target=False)
    >>> tgt_stats = compute_stats(images, boxes=boxes, stats=ImageStats.DIMENSION, per_image=False, per_target=True)
    >>> ratios = compute_ratios(img_stats, target_stats_output=tgt_stats)

    **Custom override map:**

    >>> custom_overrides = {
    ...     "mean": lambda box, img: (box["mean"] - img["mean"]) / (img["std"] + 1e-10),
    ... }
    >>> ratios = compute_ratios(stats, override_map=custom_overrides)

    **Band group statistics:**

    >>> stats = compute_stats(
    ...     images,
    ...     boxes=boxes,
    ...     stats=ImageStats.PIXEL_MEAN,
    ...     per_image=True,
    ...     per_target=True,
    ...     channels={"r": 0, "g": 1, "b": 2},
    ...     normalize_pixel_values=False,
    ... )
    >>> ratios = compute_ratios(stats)
    >>> sorted(ratios["stats"])
    ['b_mean', 'g_mean', 'mean', 'r_mean']
    """
    _logger.info(
        "Starting compute_ratios with %s input pattern",
        "separate" if target_stats_output is not None else "unified",
    )

    # Validate input
    if SOURCE_INDEX_KEY not in stats_output:
        raise KeyError(f"stats_output must contain '{SOURCE_INDEX_KEY}' key from compute_stats() output")

    # Determine which pattern we're using and validate
    if target_stats_output is not None:
        # Pattern 2: Separate image and box stats
        if SOURCE_INDEX_KEY not in target_stats_output:
            raise KeyError(f"target_stats_output must contain '{SOURCE_INDEX_KEY}' key from compute_stats() output")

        img_source_indices, box_source_indices = _validate_separate_inputs(stats_output, target_stats_output)
        _logger.debug(
            "Using separate inputs: %d image entries, %d box entries",
            len(img_source_indices),
            len(box_source_indices),
        )
        source_indices_for_lookup = img_source_indices
        source_indices_for_boxes = box_source_indices
        img_calc_result = stats_output
        box_calc_result = target_stats_output
    else:
        # Pattern 1: Unified input
        source_indices: Sequence[SourceIndex] = stats_output[SOURCE_INDEX_KEY]
        _validate_unified_input(source_indices, stats_output["stats"])
        _logger.debug("Using unified input: %d total entries", len(source_indices))
        source_indices_for_lookup = source_indices
        source_indices_for_boxes = source_indices
        img_calc_result = stats_output
        box_calc_result = stats_output

    ratio_map = dict(_default_ratio_map())
    ratio_map.update(override_map or {})

    # Build lookup table for image stats (maps image_idx -> array index)
    img_lookup = _build_image_lookup(source_indices_for_lookup)

    # Calculate overlapping stats keys for ratio calculation
    # Background names are dropped here rather than divided: see _ratio_keys.
    overlapping_keys = set(_ratio_keys(img_calc_result["stats"])) & set(_ratio_keys(box_calc_result["stats"]))
    _logger.debug("Computing ratios for %d overlapping stats: %s", len(overlapping_keys), sorted(overlapping_keys))

    # Find all box indices and their corresponding image indices
    box_indices: list[int] = []
    img_indices: list[int] = []
    ratio_source_indices: list[SourceIndex] = []

    for box_idx, source_idx in enumerate(source_indices_for_boxes):
        # Only process box entries
        if _names_the_item(source_idx):
            continue

        # Find corresponding image entry
        if source_idx.item not in img_lookup:
            raise ValueError(
                f"Cannot find image-level stats for box at image={source_idx.item}. Ensure both "
                f"stats_output and target_stats_output were computed on the same dataset.",
            )

        img_idx = img_lookup[source_idx.item]
        box_indices.append(box_idx)
        img_indices.append(img_idx)
        ratio_source_indices.append(source_idx)

    # Calculate ratios for each statistic using the overlapping keys
    ratio_stats: dict[str, NDArray[Any]] = {}
    for stat_name in overlapping_keys:
        box_stat_values = box_calc_result["stats"][stat_name]
        img_stat_values = img_calc_result["stats"][stat_name]

        ratio_values: list[Any] = []
        for box_idx, img_idx in zip(box_indices, img_indices, strict=False):
            # Build context dicts for custom override functions
            box_stats: dict[str, NDArray[Any]] = {k: v[box_idx] for k, v in box_calc_result["stats"].items()}
            img_stats: dict[str, NDArray[Any]] = {k: v[img_idx] for k, v in img_calc_result["stats"].items()}

            ratio_value = _calculate_ratio_for_stat(
                stat_name,
                box_stat_values[box_idx],
                img_stat_values[img_idx],
                ratio_map,
                box_stats,
                img_stats,
            )
            ratio_values.append(ratio_value)

        # Convert ratio values to numpy array - let numpy infer the appropriate dtype
        ratio_stats[stat_name] = np.array(ratio_values)

    # Build StatsResult dict with proper structure
    result: StatsResult = {
        "source_index": ratio_source_indices,
        "object_count": box_calc_result["object_count"],
        "invalid_box_count": box_calc_result["invalid_box_count"],
        "image_count": box_calc_result["image_count"],
        "stats": ratio_stats,
    }

    _logger.info(
        "compute_ratios complete: %d ratio entries calculated for %d images",
        len(ratio_source_indices),
        box_calc_result["image_count"],
    )

    return result
