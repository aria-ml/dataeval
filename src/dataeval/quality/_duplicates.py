"""Duplicate detection for images using hashing and clustering."""

__all__ = []

import warnings
from collections.abc import Mapping, Sequence, Sized
from itertools import combinations
from typing import Any, Generic, Literal, NamedTuple, TypeAlias, TypeVar, cast, overload

import numpy as np
import polars as pl
from numpy.typing import NDArray
from typing_extensions import Self

from dataeval import Embeddings
from dataeval._log import get_logger
from dataeval.core import (
    ClusterResult,
    SegmentMatchResult,
    StatsResult,
    align_subsequence,
    cluster,
    combine_stats_results,
    hash_groups,
    hash_neighbors,
    match_segments,
    pack_hashes,
    redundant_runs,
    sequence_containment,
    sequence_fingerprint,
)
from dataeval.data import AllFrames, FrameRate, FrameSelector, SequenceFrames, Stride
from dataeval.flags import ImageStats
from dataeval.protocols import (
    ArrayLike,
    Dataset,
    FeatureExtractor,
    MultiobjectTrackingDataset,
    MultiobjectTrackingTarget,
    _is_protocol_instance,
)
from dataeval.quality._shared import (
    LABEL_KIND,
    checked_compute_stats,
    drop_null_index_columns,
    get_dataset_step_from_idx,
    reported_level,
)
from dataeval.types import (
    ClusterConfigMixin,
    DataFrameOutput,
    Evaluator,
    EvaluatorConfig,
    FactorLevel,
    SourceIndex,
    StatsMap,
    set_metadata,
)
from dataeval.utils._internal import flatten_samples, iter_images, to_numpy

_logger = get_logger(__name__)

DEFAULT_DUPLICATES_FLAGS = ImageStats.HASH_DUPLICATES_BASIC
DEFAULT_DUPLICATES_CLUSTER_DISTANCE_FACTOR: float | None = None
DEFAULT_DUPLICATES_MERGE_NEAR_DUPLICATES = True
DEFAULT_DUPLICATES_HASH_RADIUS = 0
DEFAULT_DUPLICATES_REDUNDANCY_RADIUS = 4
DEFAULT_DUPLICATES_MIN_SEGMENT_FRAMES = 30
DEFAULT_DUPLICATES_MAX_SEGMENT_GAP = 5
DEFAULT_DUPLICATES_SEGMENT_OFFSET_TOLERANCE = 0
DEFAULT_DUPLICATES_VERIFY_ALIGNMENT: int | None = None
DEFAULT_DUPLICATES_MIN_TRACK_FRAMES = 5

# Cells one sequence pair's warping may build. Two 4 000-frame runs sit just inside it; past that
# the pair costs more than the relation is worth and is logged rather than waited on.
_ALIGNMENT_CELLS = 16_000_000
FrameSample: TypeAlias = "int | float | FrameSelector | None"
"""How much of a video to look at: a frame stride, a target frame rate, or a selector.

``int`` and ``float`` are not interchangeable here and the union is not redundant: an integer is a
*stride in frames* and a float is a *rate in frames per second*, which is a distinction
:func:`_resolve_selector` makes at runtime.
"""

DEFAULT_DUPLICATES_FRAME_SAMPLE: FrameSample = None


_BASIC_HASH_METHODS = frozenset({"phash", "dhash"})
_D4_HASH_METHODS = frozenset({"phash_d4", "dhash_d4"})

# Type alias for raw detection output: (exact_groups, near_method_groups)
# method_groups are (indices, method_name) tuples before merging
MethodGroups = list[tuple[Sequence[Any], str]]

# A redundant run carries a third field the other groups have no equivalent of: how close its
# frames actually were. Without it a run reads as "these repeat" with no figure saying how nearly.
RedundantGroup: TypeAlias = tuple[Sequence[Any], str, float]

# A track group names a representative frame of each member -- which is what says the sequence --
# beside the track ids themselves, since a track index is not a frame index and cannot be derived
# from one.
TrackGroup: TypeAlias = tuple[Sequence[Any], str, list[int]]


class SegmentMatch(NamedTuple):
    """One sequence pair's shared stretches.

    Attributes
    ----------
    query, candidate : int
        A representative frame of each sequence, as a combined item index.
    segments : SegmentMatchResult
        The shared stretches, in positions among the frames measured from each sequence.
    containment : tuple[float, float]
        How much of the query the candidate accounts for, and the reverse.
    method : str
        The hash the frames were matched by.
    query_frames, candidate_frames : NDArray[np.intp]
        Each measured frame's position in its source video, so a segment's bounds can be reported
        in the coordinates a reader can find the frames again by.
    keys : tuple[int, int]
        Which two runs this is a match between. Carried because ``query`` and ``candidate`` name a
        *frame*, which identifies a sequence but not a track -- two tracks share a frame whenever
        two objects are visible at once.
    tracks : tuple[int, int] or None
        The two track ids, when this is a match between tracks rather than between whole videos.
        None for a sequence-level match.
    """

    query: int
    candidate: int
    segments: SegmentMatchResult
    containment: tuple[float, float]
    method: str
    query_frames: NDArray[np.intp]
    candidate_frames: NDArray[np.intp]
    keys: tuple[int, int] = (-1, -1)
    tracks: tuple[int, int] | None = None


class AlignmentMatch(NamedTuple):
    """One sequence pair that runs together without keeping step -- a speed edit, or a stretch.

    What :class:`SegmentMatch` cannot express. A segment is a run of frames at one *constant*
    offset, so a copy played back at another rate arrives as a scatter of fragments too short to
    report. Warping the two against each other finds the same relation with the slope absorbed.

    Attributes
    ----------
    query, candidate : int
        A representative frame of each sequence, as a combined item index. The query is the
        shorter of the two, since it is aligned in its entirety.
    query_span, candidate_span : tuple[int, int]
        First and last frame of each side's aligned stretch, in source-video positions.
    containment : tuple[float, float]
        How much of the query the candidate accounts for, and the reverse. The query is aligned
        whole, so its own figure is 1.0 by construction and the candidate's is the informative one.
    normalized_cost : float
        Mean bits differing per aligned frame pair, on the same scale as ``hash_radius``.
    method : str
        The hash the frames were aligned by.
    """

    query: int
    candidate: int
    query_span: tuple[int, int]
    candidate_span: tuple[int, int]
    containment: tuple[float, float]
    normalized_cost: float
    method: str


class _SegmentPolicy(NamedTuple):
    """Everything that decides what counts as a shared stretch, carried as one argument.

    Bundled rather than threaded: three separate call paths compute these relations and every one
    of them has to pass the same policy, so a knob added to one signature and forgotten in another
    is the recurring defect in this module.
    """

    radius: int
    min_length: int
    max_gap: int
    offset_tolerance: int
    verify_alignment: int | None
    track_length: int


class _SharedStretches(NamedTuple):
    """What two sequences have in common, at a fixed offset and under warping."""

    segments: list[SegmentMatch]
    alignments: list[AlignmentMatch]


SingleExactDuplicatesGroup = Sequence[Sequence[int]]
SingleExactTargetDuplicatesGroup = Sequence[Sequence[SourceIndex]]
SingleNearDuplicatesGroup = Sequence[tuple[Sequence[int], Sequence[str]]]
SingleNearTargetDuplicatesGroup = Sequence[tuple[Sequence[SourceIndex], Sequence[str]]]

MultiExactDuplicatesGroup = Mapping[int, Sequence[Sequence[int]]]
MultiExactTargetDuplicatesGroup = Mapping[int, Sequence[Sequence[SourceIndex]]]
MultiNearDuplicatesGroup = Mapping[int, Sequence[tuple[Sequence[int], Sequence[str]]]]
MultiNearTargetDuplicatesGroup = Mapping[int, Sequence[tuple[Sequence[SourceIndex], Sequence[str]]]]

ExactDuplicatesGroup = (
    SingleExactDuplicatesGroup
    | SingleExactTargetDuplicatesGroup
    | MultiExactDuplicatesGroup
    | MultiExactTargetDuplicatesGroup
)
NearDuplicatesGroup = (
    SingleNearDuplicatesGroup
    | SingleNearTargetDuplicatesGroup
    | MultiNearDuplicatesGroup
    | MultiNearTargetDuplicatesGroup
)

TExactDuplicatesGroup = TypeVar(
    "TExactDuplicatesGroup",
    SingleExactDuplicatesGroup,
    SingleExactTargetDuplicatesGroup,
    MultiExactDuplicatesGroup,
    MultiExactTargetDuplicatesGroup,
)
TNearDuplicatesGroup = TypeVar(
    "TNearDuplicatesGroup",
    SingleNearDuplicatesGroup,
    SingleNearTargetDuplicatesGroup,
    MultiNearDuplicatesGroup,
    MultiNearTargetDuplicatesGroup,
)

_EMPTY_DUPS_SCHEMA: dict[str, pl.DataType | type] = {
    "group_id": pl.Int64,
    "level": pl.Utf8,
    "dup_type": pl.Utf8,
    "item_indices": pl.List(pl.Int64),
    "unit_indices": pl.List(pl.Int64),
    "track_indices": pl.List(pl.Int64),
    "target_indices": pl.List(pl.Int64),
    "address_levels": pl.List(pl.Utf8),
    "methods": pl.List(pl.Utf8),
    "orientation": pl.Utf8,
    "span_start": pl.List(pl.Int64),
    "span_end": pl.List(pl.Int64),
    "containment": pl.List(pl.Float64),
    "mean_distance": pl.Float64,
}

# Level names by task. Image tasks keep the spellings that shipped; a tracking dataset uses the
# names Metadata already gives its levels, because "item" for a video and "target" for a
# detection inside a frame name neither the thing measured nor the level it sits at.
_IMAGE_LEVELS = {"item": "item", "target": "target"}
_TRACKING_LEVELS = {"item": "unit", "target": "instance", "sequence": "sequence", "track": "track"}


def _resolve_selector(frame_sample: FrameSample) -> FrameSelector:
    """Turn the `frame_sample` policy into the mechanism that carries it out."""
    if frame_sample is None:
        return AllFrames()
    if isinstance(frame_sample, FrameSelector):
        return frame_sample
    if isinstance(frame_sample, bool):
        raise TypeError(f"frame_sample must be a frame stride, a target fps, or a FrameSelector; got {frame_sample!r}.")
    return Stride(int(frame_sample)) if isinstance(frame_sample, int) else FrameRate(float(frame_sample))


# ---------------------------------------------------------------------------
# Module-level helper functions (extracted from Duplicates class for reuse by
# DuplicatesOutput._redetect without needing an evaluator instance)
# ---------------------------------------------------------------------------


def _get_orientation(methods: frozenset[str]) -> Literal["rotated", "same"]:
    """Determine orientation based on which methods detected the group."""
    has_basic = bool(methods & _BASIC_HASH_METHODS)
    has_d4 = bool(methods & _D4_HASH_METHODS)
    if has_d4 and not has_basic:
        return "rotated"
    return "same"


def _merge_near_groups(  # noqa: C901
    method_groups: Sequence[tuple[Sequence[Any], str]],
    available_stats: set[str],
    merge: bool,
) -> list[tuple[tuple[Any, ...], frozenset[str], str | None]]:
    """Merge overlapping near-duplicate groups and compute orientation.

    Parameters
    ----------
    method_groups : Sequence[tuple[Sequence[Any], str]]
        List of (indices, method_name) tuples from each detection method.
    available_stats : set[str]
        Set of hash types that were computed (e.g., {"phash", "dhash", "phash_d4"}).
    merge : bool
        Whether to merge overlapping groups from different methods.

    Returns
    -------
    list[tuple[tuple[Any, ...], frozenset[str], str | None]]
        Each element is (sorted_indices, methods, orientation).
    """
    if not method_groups:
        return []

    # Determine if we can compute orientation (need both basic and D4 hashes)
    has_basic_stats = bool(available_stats & _BASIC_HASH_METHODS)
    has_d4_stats = bool(available_stats & _D4_HASH_METHODS)
    is_unknown = not (has_basic_stats and has_d4_stats)

    if not merge:
        # Keep groups separate - each group has a single method
        groups = [
            (
                tuple(sorted(group, key=_address_order)),
                frozenset({method}),
                None if is_unknown else _get_orientation(frozenset({method})),
            )
            for group, method in method_groups
        ]
        return sorted(groups, key=lambda g: _group_order(g[0]))

    # Merge overlapping groups and union their methods
    # Each entry: (set of indices, set of methods)
    merged: list[tuple[set[Any], set[str]]] = []

    for group, method in method_groups:
        group_set = set(group)
        overlapping_indices: list[int] = []

        for i, (existing_set, _) in enumerate(merged):
            if existing_set & group_set:  # Any overlap
                overlapping_indices.append(i)

        if not overlapping_indices:
            # No overlap - add as new group
            merged.append((group_set, {method}))
        else:
            # Merge with all overlapping groups
            new_indices = group_set.copy()
            new_methods = {method}
            for i in sorted(overlapping_indices, reverse=True):
                existing_indices, existing_methods = merged.pop(i)
                new_indices |= existing_indices
                new_methods |= existing_methods
            merged.append((new_indices, new_methods))

    result = [
        (
            tuple(sorted(indices, key=_address_order)),
            frozenset(methods),
            None if is_unknown else _get_orientation(frozenset(methods)),
        )
        for indices, methods in merged
        if len(indices) > 1
    ]
    return sorted(result, key=lambda g: _group_order(g[0]))


def _find_cluster_duplicates(
    mst: NDArray[np.float32],
    clusters: NDArray[np.intp],
    cluster_sensitivity: float = 1.0,
) -> list[list[int]]:
    """Find duplicate data based on cluster average distance.

    Parameters
    ----------
    mst : NDArray[np.float32]
        Minimum spanning tree from cluster() output.
    clusters : NDArray[np.intp]
        Cluster labels from cluster() output.
    cluster_sensitivity : float, default 1.0
        Controls how aggressively points are considered duplicates by
        scaling the cluster's standard deviation. Lower values are
        stricter (fewer duplicates). Typical range: 0.1 – 3.0.

    Returns
    -------
    list[list[int]]
        Duplicates as lists of related indices.
    """
    from dataeval.core._fast_hdbscan._mst import compare_links_to_cluster_std

    indices = compare_links_to_cluster_std(mst, clusters, cluster_sensitivity)
    dupes = _sorted_union_find(indices)

    return [[int(ii) for ii in il] for il in dupes]


def _sorted_union_find(index_groups: Any) -> list[list[Any]]:
    """Merge and sort groups of indices that share any common index."""
    from dataeval.core._fast_hdbscan._disjoint_set import ds_find, ds_rank_create, ds_union_by_rank

    groups: list[list[np.int32]] = [[np.int32(x) for x in range(0)] for y in range(0)]
    uniques, inverse = np.unique(index_groups, return_inverse=True)
    inverse = inverse.flatten()
    disjoint_set = ds_rank_create(np.int64(uniques.size))
    cluster_points = np.empty(uniques.size, dtype=np.uint32)
    for i in range(index_groups.shape[0]):
        point, nbr = np.intp(inverse[i * 2]), np.intp(inverse[i * 2 + 1])
        ds_union_by_rank(disjoint_set, point, nbr)
    for i in range(uniques.size):
        cluster_points[i] = ds_find(disjoint_set, np.intp(i))
    for i in range(uniques.size):
        dups = np.nonzero(cluster_points == i)[0]
        if dups.size > 0:
            groups.append(uniques[dups].tolist())
    return sorted(groups)


def _is_between_the_ends(members: Sequence[Any]) -> bool:
    """Whether a group's members address a level between an item and one of its labels.

    Every member of a group shares a kind by construction — `_detect_hash_duplicates`
    buckets on it before anything is compared — so the first member answers for all.
    """
    first = next((m for m in members if isinstance(m, SourceIndex)), None)
    return first is not None and first.kind not in (None, LABEL_KIND)


def _selected_targets(groups: Sequence[Any], per_target: bool, near: bool = False) -> list[Any]:
    """Return the label-role groups `per_target` selects, plus every group between the ends.

    `per_image` and `per_target` name the two ends of the level graph — an item's own row,
    and one of its labels — and a group's kind says which of those it is, canonically, so
    the fully explicit spelling of a result is gated exactly as the minimal spelling is. A
    group addressing a level *between* the ends, such as a video frame, is neither, so
    neither flag has a say over it.
    """
    if per_target:
        return list(groups)
    return [g for g in groups if _is_between_the_ends(g[0] if near else g)]


def _address_order(index: "SourceIndex | int") -> tuple[int, int, str]:
    """Sort key for a member, whether it is an address or a bare item index.

    An item's own row is reported as a bare index rather than as an address, so a group's
    members are one or the other and both have to order against the same key.
    :attr:`~dataeval.types.SourceIndex.sort_key` is that key for an address, and a bare
    index is the unkeyed address it stands for.
    """
    return index.sort_key if isinstance(index, SourceIndex) else (index, -1, "")


def _group_order(members: Sequence[Any]) -> list[tuple[int, int, str]]:
    """Sort key for a whole group, ordering it the way its members order.

    The one spelling of "compare these groups", so that every sort over groups agrees and
    none of them falls back to comparing addresses as the raw tuples they are.
    """
    return [_address_order(member) for member in members]


def _member_levels(row: Mapping[str, Any]) -> Sequence[FactorLevel | None]:
    """Return the level each member's address states, or nulls when the row carries none.

    The column holds whatever the addresses stated, and an address cannot hold a value that
    is not a level — :class:`~dataeval.types.SourceIndex` rejects one on construction — so
    what comes back out is a level or nothing.
    """
    levels = row.get("address_levels")
    if levels is None:
        return [None] * len(row["item_indices"])
    return cast("Sequence[FactorLevel | None]", levels)


def _extract_members(row: Mapping[str, Any], has_targets: bool) -> list[Any]:
    """Extract member indices from a single DataFrame row.

    A frame view's rows carry ``unit_indices`` as well, because a member there is a frame *within*
    a sequence and the sequence index alone does not name it -- three groups drawn from the same
    pair of videos would otherwise read identically.
    :class:`~dataeval.types.SourceIndex` addresses an item and a target with no level in between,
    so a frame is reported as a plain ``(sequence, frame)`` pair.
    """
    units = row.get("unit_indices")
    if units is not None:
        if has_targets:
            return [
                (item, unit, target)
                for item, unit, target in zip(row["item_indices"], units, row["target_indices"], strict=True)
            ]
        return [(item, unit) for item, unit in zip(row["item_indices"], units, strict=True)]
    if has_targets:
        return [
            SourceIndex(item=item, key=target, level=level)
            for item, target, level in zip(row["item_indices"], row["target_indices"], _member_levels(row), strict=True)
        ]
    return row["item_indices"]


def _group_by_dataset(row: Mapping[str, Any], has_targets: bool) -> dict[int, list[Any]]:
    """Group a row's members by the dataset each came from."""
    by_ds: dict[int, list[Any]] = {}
    for member, ds in zip(_extract_members(row, has_targets), row["dataset_indices"], strict=True):
        by_ds.setdefault(ds, []).append(member)
    return by_ds


def _get_groups_single(filtered: pl.DataFrame, has_targets: bool, is_near: bool) -> list[Any]:
    """Extract duplicate groups for single-dataset results."""
    groups: list[Any] = []
    for row in filtered.iter_rows(named=True):
        members = _extract_members(row, has_targets)
        if is_near:
            groups.append((members, row["methods"]))
        else:
            groups.append(members)
    return groups


def _get_groups_cross(filtered: pl.DataFrame, has_targets: bool, is_near: bool) -> dict[int, list[Any]]:
    """Extract duplicate groups for cross-dataset results, keyed by dataset index."""
    result: dict[int, list[Any]] = {}
    for row in filtered.iter_rows(named=True):
        by_ds = _group_by_dataset(row, has_targets)
        for ds, members in by_ds.items():
            if is_near:
                result.setdefault(ds, []).append((members, row["methods"]))
            else:
                result.setdefault(ds, []).append(members)
    return result


def _indices_to_row_fields(
    indices: Sequence[Any],
    dataset_steps: Sequence[int] | None,
) -> tuple[list[int], list[int | None], list[str | None], list[int] | None]:
    """Destructure a group's members into the row's columns.

    The level is carried out alongside the item and the key rather than dropped, so that
    :func:`_extract_members` can rebuild the address that went in. Without it a frame comes
    back as ``SourceIndex(0, 3)``, which names detection 3 of item 0 — a different row.
    """
    item_indices: list[int] = []
    target_indices: list[int | None] = []
    address_levels: list[str | None] = []
    dataset_ids: list[int] = [] if dataset_steps is not None else None  # type: ignore[assignment]

    for idx in indices:
        if isinstance(idx, SourceIndex):
            item_idx = idx.item
            target_idx = idx.key
            level = reported_level(idx)
        else:
            item_idx = idx
            target_idx = None
            level = None

        if dataset_steps is not None:
            ds_idx, item_idx = get_dataset_step_from_idx(item_idx, dataset_steps)
            dataset_ids.append(ds_idx)

        item_indices.append(item_idx)
        target_indices.append(target_idx)
        address_levels.append(level)

    return item_indices, target_indices, address_levels, dataset_ids


def _make_row(
    indices: Sequence[Any],
    group_id: int,
    level: str,
    dup_type: str,
    methods: Sequence[str],
    orientation: str | None,
    dataset_steps: Sequence[int] | None,
    frame_map: NDArray[np.intp] | None = None,
    spans: tuple[list[int], list[int]] | None = None,
    containment: list[float] | None = None,
    mean_distance: float | None = None,
    track_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Build a single DataFrame row dict from a duplicate group."""
    item_ids, target_ids, address_levels, ds_ids = _indices_to_row_fields(indices, dataset_steps)
    unit_ids: list[int] | None = None
    if frame_map is not None:
        # A frame view's item index is a position in the flattened walk. Split it back into the
        # sequence it came from and its position inside that sequence, which is what a reader
        # needs to find the frame again in the source video.
        #
        # Looked up by the *combined* index rather than the per-dataset one `item_ids` already
        # holds: the map spans every dataset in the call, so a local index would read the wrong
        # dataset's rows for every dataset after the first.
        combined = [index.item if isinstance(index, SourceIndex) else index for index in indices]
        located = frame_map[np.asarray(combined, dtype=np.intp)]
        item_ids, unit_ids = located[:, 0].tolist(), located[:, 1].tolist()
    row: dict[str, Any] = {
        "group_id": group_id,
        "level": level,
        "dup_type": dup_type,
        "item_indices": item_ids,
        "unit_indices": unit_ids,
        "track_indices": track_ids,
        "target_indices": target_ids,
        "address_levels": address_levels,
    }
    if ds_ids is not None:
        row["dataset_indices"] = ds_ids
    row["methods"] = methods
    row["orientation"] = orientation
    row["span_start"], row["span_end"] = spans if spans is not None else (None, None)
    row["containment"] = containment
    row["mean_distance"] = mean_distance
    return row


def _build_duplicates_dataframe(  # noqa: C901
    item_exact: Sequence[Sequence[int]] | None,
    item_near_method_groups: Sequence[tuple[Sequence[Any], str]],
    target_exact: Sequence[Sequence[SourceIndex]] | None,
    target_near_method_groups: Sequence[tuple[Sequence[Any], str]],
    available_stats: set[str],
    merge: bool,
    dataset_steps: Sequence[int] | None = None,
    frame_map: NDArray[np.intp] | None = None,
    redundant_groups: Sequence[RedundantGroup] | None = None,
    sequence_exact: MethodGroups | None = None,
    segment_matches: Sequence[SegmentMatch] | None = None,
    alignment_matches: Sequence[AlignmentMatch] | None = None,
    track_exact: Sequence[TrackGroup] | None = None,
) -> pl.DataFrame:
    """Build a unified DataFrame of duplicate groups from raw detection data.

    Handles near-group merging internally via ``_merge_near_groups``.
    Each row represents one duplicate group with columns defined by ``_EMPTY_DUPS_SCHEMA``.

    ``frame_map`` marks the results as coming from a frame view of a tracking dataset: it splits
    each flattened frame position back into ``(sequence, frame)`` and renames the levels to the
    ones :class:`~dataeval.Metadata` gives a tracking dataset.
    """
    rows: list[dict[str, Any]] = []
    group_id = 0
    names = _IMAGE_LEVELS if frame_map is None else _TRACKING_LEVELS

    for indices, method in sequence_exact or []:
        rows.append(_make_row(indices, group_id, names["sequence"], "exact", [method], None, dataset_steps, frame_map))
        rows[-1]["unit_indices"] = None
        group_id += 1

    for indices, method, track_ids in track_exact or []:
        rows.append(
            _make_row(
                indices,
                group_id,
                names["track"],
                "exact",
                [method],
                None,
                dataset_steps,
                frame_map,
                track_ids=track_ids,
            )
        )
        rows[-1]["unit_indices"] = None
        group_id += 1

    for match in segment_matches or []:
        segments = match.segments
        level = names["sequence"] if match.tracks is None else names["track"]
        for index in range(len(segments["offset"])):
            rows.append(
                _make_row(
                    [match.query, match.candidate],
                    group_id,
                    level,
                    "segment",
                    [match.method],
                    None,
                    dataset_steps,
                    frame_map,
                    spans=(
                        [
                            int(match.query_frames[segments["query_start"][index]]),
                            int(match.candidate_frames[segments["candidate_start"][index]]),
                        ],
                        [
                            int(match.query_frames[segments["query_end"][index]]),
                            int(match.candidate_frames[segments["candidate_end"][index]]),
                        ],
                    ),
                    containment=[float(match.containment[0]), float(match.containment[1])],
                    mean_distance=float(segments["mean_distance"][index]),
                    track_ids=None if match.tracks is None else list(match.tracks),
                )
            )
            rows[-1]["unit_indices"] = None
            group_id += 1

    for match in alignment_matches or []:
        rows.append(
            _make_row(
                [match.query, match.candidate],
                group_id,
                names["sequence"],
                "aligned",
                [match.method],
                None,
                dataset_steps,
                frame_map,
                spans=(
                    [match.query_span[0], match.candidate_span[0]],
                    [match.query_span[1], match.candidate_span[1]],
                ),
                containment=[float(match.containment[0]), float(match.containment[1])],
                mean_distance=match.normalized_cost,
            )
        )
        rows[-1]["unit_indices"] = None
        group_id += 1

    for indices, method, distance in redundant_groups or []:
        rows.append(
            _make_row(
                indices,
                group_id,
                names["item"],
                "redundant",
                [method],
                None,
                dataset_steps,
                frame_map,
                mean_distance=distance,
            )
        )
        group_id += 1

    for level, exact_groups, near_method_groups in (
        ("item", item_exact, item_near_method_groups),
        ("target", target_exact, target_near_method_groups),
    ):
        if exact_groups:
            ordered = [sorted(g, key=_address_order) for g in exact_groups]
            for group in sorted(ordered, key=lambda g: _address_order(g[0])):
                rows.append(
                    _make_row(group, group_id, names[level], "exact", ["xxhash"], None, dataset_steps, frame_map)
                )
                group_id += 1

        if near_method_groups:
            for indices, methods, orientation in _merge_near_groups(near_method_groups, available_stats, merge):
                rows.append(
                    _make_row(
                        indices, group_id, names[level], "near", sorted(methods), orientation, dataset_steps, frame_map
                    )
                )
                group_id += 1

    # Orientation is only meaningful when both basic and D4 hashes were computed
    has_basic_stats = bool(available_stats & _BASIC_HASH_METHODS)
    has_d4_stats = bool(available_stats & _D4_HASH_METHODS)
    include_orientation = has_basic_stats and has_d4_stats

    # Build schema explicitly so polars does not infer dtypes from a 100-row sample.
    # Without this, ≥100 exact rows (orientation=None) followed by a near row with a
    # string orientation trigger a ComputeError when polars appends the string to a
    # Null-typed column.
    schema: dict[str, pl.DataType | type] = {}
    for key, dtype in _EMPTY_DUPS_SCHEMA.items():
        if key == "methods" and dataset_steps is not None:
            schema["dataset_indices"] = pl.List(pl.Int64)
        if key == "orientation" and not include_orientation:
            continue
        schema[key] = dtype

    if not rows:
        return pl.DataFrame(schema=schema)

    df = pl.DataFrame(rows, schema=schema)

    return drop_null_index_columns(
        df,
        [
            "target_indices",
            "address_levels",
            "unit_indices",
            "track_indices",
            "span_start",
            "span_end",
            "containment",
            "mean_distance",
        ],
    )


def _find_hash_groups(
    stats: StatsMap,
    hash_key: str,
    source_index: Sequence[SourceIndex],
    indices: Sequence[int],
    exact_groups: Sequence[Sequence[Any]],
    use_source_index: bool = False,
    hash_radius: int = DEFAULT_DUPLICATES_HASH_RADIUS,
) -> list[list[Any]]:
    """Find near duplicates for a specific hash type.

    When use_source_index is True, stores full SourceIndex objects (for targets).
    Otherwise stores item integers (for items).

    At ``hash_radius`` 0 a group is a set of *identical* digests, which is the strictest reading
    of "near": a re-saved PNG often reproduces its perceptual hash exactly, and a great deal of
    real redundancy is found this way. Above 0 the digests are searched by Hamming distance
    instead, so a re-encode that moved a few bits still groups with its source. Grouping is
    transitive at any radius — a chain of near-duplicates is one redundant set — which is the
    same reading the identical-digest path already had.

    Empty digests take no part either way. An empty digest is how a calculator reports a region
    it could not measure, and grouping those together would call every unmeasured region a
    duplicate of every other.
    """
    keys = [source_index[i] if use_source_index else source_index[i].item for i in indices]
    if hash_radius:
        digests = [stats[hash_key][i] for i in indices]
        codes, valid = pack_hashes(digests)
        # pack_hashes pads each digest out to a whole 64-bit word, so tell the search how many of
        # those bits the digest actually spans. Left to assume the padding is signal, it spends
        # bands on bits that are zero in every code, and every code lands in the same bucket.
        bits = next((len(digest) * 4 for digest in digests if digest), None)
        groups = hash_groups(codes, hash_radius, valid=valid, bits=bits)["groups"]
        candidates = [sorted(keys[position] for position in group) for group in groups]
    else:
        near_dict: dict[str, list[Any]] = {}
        for key, i in zip(keys, indices, strict=True):
            value = stats[hash_key][i]
            if value:  # Skip empty hashes
                near_dict.setdefault(value, []).append(key)
        candidates = [sorted(v) for v in near_dict.values() if len(v) > 1]

    return [group for group in candidates if not any(set(group).issubset(x) for x in exact_groups)]


def _find_exact_groups(
    stats: StatsMap,
    indices: list[int],
    key_fn: Any,
) -> list[list[Any]]:
    """Group indices by xxhash and return groups with more than one member.

    Empty hashes are skipped, as they are for near duplicates. An empty digest is how a
    calculator reports a region it could not measure — an out-of-bounds box, an image its
    boxes cover completely, or a band group the datum cannot supply — and grouping those
    together would call every unmeasured region an exact duplicate of every other.
    """
    if "xxhash" not in stats:
        return []
    d: dict[str, list[Any]] = {}
    for i in indices:
        value = stats["xxhash"][i]
        if value:
            d.setdefault(value, []).append(key_fn(i))
    return [sorted(v) for v in d.values() if len(v) > 1]


def _find_near_group_pairs(
    stats: StatsMap,
    source_index: Sequence[SourceIndex],
    indices: list[int],
    exact_groups: list[list[Any]],
    hash_methods: list[str],
    *,
    use_source_index: bool = False,
    hash_radius: int = DEFAULT_DUPLICATES_HASH_RADIUS,
) -> MethodGroups:
    """Find near-duplicate groups across all hash methods."""
    near: MethodGroups = []
    for method in hash_methods:
        if method in stats:
            near.extend(
                (g, method)
                for g in _find_hash_groups(
                    stats,
                    method,
                    source_index,
                    indices,
                    exact_groups,
                    use_source_index=use_source_index,
                    hash_radius=hash_radius,
                )
            )
    return near


def _detect_hash_duplicates(
    stats: StatsMap,
    source_index: Sequence[SourceIndex],
    hash_radius: int = DEFAULT_DUPLICATES_HASH_RADIUS,
) -> tuple[
    tuple[SingleExactDuplicatesGroup, MethodGroups],
    tuple[SingleExactTargetDuplicatesGroup, MethodGroups],
]:
    """Extract duplicate groups from hash statistics, separating items and targets.

    Returns
    -------
    tuple of ((item_exact, item_method_groups), (target_exact, target_method_groups))
        Raw detection results for item-level and target-level duplicates.

    Raises
    ------
    ValueError
        If ``hash_radius`` is negative.
    """
    # Checked here rather than where the grouping runs, because that only runs for flags that ask
    # for a perceptual hash: an xxhash-only configuration would otherwise accept a nonsense radius
    # and quietly return a result computed without it.
    if hash_radius < 0:
        raise ValueError(f"hash_radius must be non-negative; got {hash_radius}.")

    # Readings are bucketed by :attr:`~dataeval.types.SourceIndex.kind` before anything is
    # compared, so a reading of one kind of row is never hashed against a reading of another
    # — a video frame and a detection inside it are both keyed addresses, and pooling them
    # would call them duplicates of each other.
    buckets: dict[FactorLevel | None, list[int]] = {}
    for i, src_idx in enumerate(source_index):
        buckets.setdefault(src_idx.kind, []).append(i)

    hash_methods = ["phash", "dhash", "phash_d4", "dhash_d4"]

    item_exact: list[Any] = []
    item_near: MethodGroups = []
    target_exact: list[Any] = []
    target_near: MethodGroups = []
    for kind, indices in buckets.items():
        if kind is not None:
            exact = _find_exact_groups(stats, indices, lambda i: source_index[i])
            target_exact.extend(exact)
            target_near.extend(
                _find_near_group_pairs(
                    stats, source_index, indices, exact, hash_methods, use_source_index=True, hash_radius=hash_radius
                )
            )
        else:
            exact = _find_exact_groups(stats, indices, lambda i: source_index[i].item)
            item_exact.extend(exact)
            item_near.extend(
                _find_near_group_pairs(stats, source_index, indices, exact, hash_methods, hash_radius=hash_radius)
            )

    # Ordering is left to :func:`_build_duplicates_dataframe`, the sole consumer, which sorts
    # the members of every group and then the groups themselves.
    return (item_exact, item_near), (target_exact, target_near)


def _prepare_hash_inputs(
    calculation_results: StatsResult | Sequence[StatsResult],
) -> tuple[StatsMap, list[SourceIndex], set[str], Sequence[int] | None]:
    """Prepare unified stats and source_index from single or multi-dataset calculation results.

    Returns (stats, source_index, available_stats, dataset_steps).
    """
    if isinstance(calculation_results, dict):
        stats = calculation_results["stats"]
        return stats, list(calculation_results["source_index"]), set(stats.keys()), None

    combined_stats, combined_source_index, dataset_steps = combine_stats_results(calculation_results)
    return combined_stats, combined_source_index, set(combined_stats.keys()), dataset_steps


_REDUNDANCY_METHODS = ("phash", "dhash", "xxhash")
"""Hashes a redundancy scan will use, in preference order.

The D4 variants are deliberately absent: consecutive frames of one video are not rotations of one
another, so their invariance buys nothing here and costs eight times as much.
"""


def _run_keys(
    items: NDArray[np.intp],
    frame_map: NDArray[np.intp],
    dataset_steps: Sequence[int] | None,
) -> NDArray[np.intp]:
    """Label each measured frame with the sequence a temporal run may not leave.

    Sequence 0 of one dataset is not sequence 0 of the next, so for a several-dataset call the
    label has to name the dataset too -- the frame maps are laid end to end and their sequence
    numbering restarts at every boundary.
    """
    sequences = frame_map[items, 0]
    if dataset_steps is None:
        return sequences
    datasets = np.searchsorted(np.asarray(dataset_steps, dtype=np.intp), items, side="right")
    return datasets * (int(sequences.max(initial=0)) + 1) + sequences


class _HashedRuns(NamedTuple):
    """Measured rows grouped into ordered runs, which is what every temporal relation starts from.

    Level-neutral on purpose. A sequence's *frames* in temporal order are a run, and so are a
    *track's* detection crops ordered by frame -- so redundancy, whole-run matching and segment
    matching are one implementation asking for one shape, rather than two that drift apart.

    Attributes
    ----------
    method : str
        The hash the relations are read from, the first of :data:`_REDUNDANCY_METHODS` present.
    digests : list[str]
        Each row's hex digest, in walk order -- which within a run is temporal order.
    items : NDArray[np.intp]
        Each row's combined item index, which is what a duplicate group names. This is the
        *frame's* index at both levels: a detection is addressed by the frame holding it.
    keys : NDArray[np.intp]
        The run each row belongs to, unique across datasets -- a sequence for frames, a track for
        detections. See :func:`_run_keys`.
    codes, valid : NDArray
        ``digests`` packed for bitwise comparison. See :func:`~dataeval.core.pack_hashes`.
    bits : int or None
        Significant bits per digest, or None when nothing was measured anywhere. Passed to the
        searches so a digest padded to a word boundary does not spend bands on bits that are zero
        in every code -- a band lying wholly in the padding puts the whole corpus in one bucket.
    """

    method: str
    digests: list[str]
    items: NDArray[np.intp]
    keys: NDArray[np.intp]
    codes: NDArray[np.uint64]
    valid: NDArray[np.bool_]
    bits: int | None


def _sequence_frames(
    stats: StatsMap,
    source_index: Sequence[SourceIndex],
    frame_map: NDArray[np.intp],
    dataset_steps: Sequence[int] | None,
) -> _HashedRuns | None:
    """Gather the frame rows every sequence-level relation starts from, or None if there are none.

    The shared prologue of redundancy, whole-sequence matching and segment matching: which measured
    rows are whole frames, which sequence each belongs to, and their digests packed for comparison.
    Shared rather than repeated, so one call does not pack the whole corpus three times over.
    """
    method = next((name for name in _REDUNDANCY_METHODS if name in stats), None)
    if method is None:
        return None
    positions = np.array([i for i, index in enumerate(source_index) if index.key is None], dtype=np.intp)
    if not len(positions):
        return None
    items = np.array([source_index[i].item for i in positions], dtype=np.intp)
    digests = [str(stats[method][i]) for i in positions]
    codes, valid = pack_hashes(digests)
    return _HashedRuns(
        method=method,
        digests=digests,
        items=items,
        keys=_run_keys(items, frame_map, dataset_steps),
        codes=codes,
        valid=valid,
        bits=next((len(digest) * 4 for digest in digests if digest), None),
    )


def _find_redundant_runs(
    stats: StatsMap,
    source_index: Sequence[SourceIndex],
    frame_map: NDArray[np.intp] | None,
    radius: int,
    dataset_steps: Sequence[int] | None = None,
) -> list[RedundantGroup]:
    """Find each sequence's stretches of frames that carry nothing new over the one before.

    Runs are found *within* a sequence and never across one, because the relation is temporal and
    two sequences are not adjacent in time. Returns groups shaped like every other near-duplicate
    group -- the run's frames, and the method that found them -- so nothing downstream needs a
    second representation.

    A ``frame_map`` of None means image data, which has no temporal order for a run to span, and
    answers with no groups -- so every caller asks the same way whatever it was handed.
    """
    gathered = _sequence_frames(stats, source_index, frame_map, dataset_steps) if frame_map is not None else None
    if gathered is None:
        return []

    groups: list[RedundantGroup] = []
    for key in np.unique(gathered.keys):
        # Slice the sequence's own frames out, in the order they were walked, which is the order
        # they occur in. Their positions within the sequence come back from the frame map.
        where = np.flatnonzero(gathered.keys == key)
        runs = redundant_runs(gathered.codes[where], radius, valid=gathered.valid[where])
        for start, end, distance in zip(runs["start"], runs["end"], runs["mean_distance"], strict=True):
            members = [int(gathered.items[where[position]]) for position in range(start, end + 1)]
            groups.append((members, gathered.method, float(distance)))
    return groups


def _find_sequence_duplicates(
    stats: StatsMap,
    source_index: Sequence[SourceIndex],
    frame_map: NDArray[np.intp] | None,
    dataset_steps: Sequence[int] | None = None,
) -> MethodGroups:
    """Group sequences holding the same frames in the same order.

    The cheapest relation there is, and the one a frame-level scan cannot state: two videos whose
    frames all match pairwise might still be different edits of the same footage. Reported as
    groups of *representative frames*, one per sequence, so a caller can name the sequences the
    same way it names everything else, each beside the hash that found it.

    A sequence none of whose frames could be measured takes no part. Every unmeasured frame packs
    to the same empty digest, so such sequences all fingerprint alike -- and calling them copies of
    one another is a claim about frames there was no data for.
    """
    gathered = _sequence_frames(stats, source_index, frame_map, dataset_steps) if frame_map is not None else None
    if gathered is None:
        return []

    digests: dict[str, list[int]] = {}
    for key in np.unique(gathered.keys):
        # Frames arrive in walk order, which within a sequence is temporal order, so the
        # fingerprint is order-sensitive in the way it needs to be.
        where = np.flatnonzero(gathered.keys == key)
        if not gathered.valid[where].any():
            continue
        fingerprint = sequence_fingerprint([gathered.digests[index] for index in where])
        digests.setdefault(fingerprint["exact"], []).append(int(gathered.items[where[0]]))
    return [(sorted(members), gathered.method) for members in digests.values() if len(members) > 1]


def _sharing_runs(
    frames: _HashedRuns,
    radius: int,
) -> list[tuple[int, int]]:
    """Return the pairs of runs sharing any row at all -- the screen before the pairwise search.

    Grouping rather than pairing, so a corpus full of identical frames costs its own size rather
    than the square of it. Transitive grouping over-reports, which is the safe direction: a pair
    that shares nothing simply yields no segments.
    """
    candidates: set[tuple[int, int]] = set()
    groups = hash_groups(frames.codes, radius, valid=frames.valid, bits=frames.bits)["groups"]
    for group in groups:
        present = sorted({int(frames.keys[position]) for position in group})
        candidates.update((a, b) for index, a in enumerate(present) for b in present[index + 1 :])
    return sorted(candidates)


def _crossing_pairs(
    frames: _HashedRuns,
    left: NDArray[np.intp],
    right: NDArray[np.intp],
    radius: int,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Return the matched pairs crossing from one sequence to the other, in their own positions.

    A pair inside a single sequence is redundancy, which has its own relation, so only the
    crossing pairs are kept.

    The search still enumerates those within-sequence pairs before they are dropped, and they count
    against its own pair budget: two static-camera sequences hold quadratically many identical
    frames that were never going to cross. Segments are one relation among several, so an overrun
    costs this pair its segments and says so, rather than taking the whole result down with it.
    """
    joined = np.concatenate((left, right))
    try:
        neighbours = hash_neighbors(frames.codes[joined], radius, valid=frames.valid[joined], bits=frames.bits)
    except ValueError as err:
        _logger.info(
            "Duplicates: no segments for a %d and %d frame sequence pair; matching them exceeded "
            "the neighbour budget (%s). Their frames repeat themselves heavily enough that the "
            "within-sequence matches alone are quadratic -- redundant_runs reports those.",
            len(left),
            len(right),
            err,
        )
        return np.empty((0, 2), dtype=np.intp), np.empty(0, dtype=np.intp)
    pairs = neighbours["pairs"]
    if not len(pairs):
        return np.empty((0, 2), dtype=np.intp), np.empty(0, dtype=np.intp)
    first = pairs[:, 0] < len(left)
    crosses = first != (pairs[:, 1] < len(left))
    pairs, first = pairs[crosses], first[crosses]
    # Re-expressed as (position in left, position in right), whichever way each pair fell.
    query = np.where(first, pairs[:, 0], pairs[:, 1])
    candidate = np.where(first, pairs[:, 1], pairs[:, 0]) - len(left)
    return np.stack((query, candidate), axis=1).astype(np.intp), neighbours["distances"][crosses]


def _checked_segment_policy(policy: _SegmentPolicy) -> None:
    """Reject a nonsense segment policy.

    Checked before the matching rather than inside it, because the matching only runs for a
    tracking dataset: an image-only call would otherwise accept a nonsense policy and quietly
    return a result computed without it -- the same reason ``hash_radius`` is checked up front.
    """
    if policy.min_length < 1:
        raise ValueError(f"min_segment_frames must be at least 1; got {policy.min_length}.")
    if policy.max_gap < 0 or policy.offset_tolerance < 0:
        raise ValueError(
            "max_segment_gap and segment_offset_tolerance must be non-negative; "
            f"got {policy.max_gap} and {policy.offset_tolerance}."
        )
    if policy.verify_alignment is not None and policy.verify_alignment < 0:
        raise ValueError(f"verify_alignment must be non-negative or None; got {policy.verify_alignment}.")
    if policy.track_length < 1:
        raise ValueError(f"min_track_frames must be at least 1; got {policy.track_length}.")


def _aligned_pair(
    frames: _HashedRuns,
    left: NDArray[np.intp],
    right: NDArray[np.intp],
    frame_map: NDArray[np.intp],
    policy: _SegmentPolicy,
) -> AlignmentMatch | None:
    """Warp one sequence pair against each other, or None if they do not align closely enough.

    The shorter run is the query, because subsequence alignment places the whole of the query
    inside a window of the candidate and asking the reverse of a long run against a short one has
    no answer.
    """
    threshold = policy.verify_alignment
    if threshold is None:
        return None
    # Unmeasured frames all pack to the same empty code, so leaving them in aligns two sequences
    # by the frames neither of them has.
    left, right = left[frames.valid[left]], right[frames.valid[right]]
    if len(left) < policy.min_length or len(right) < policy.min_length:
        return None
    query, candidate = (left, right) if len(left) <= len(right) else (right, left)
    try:
        found = align_subsequence(
            frames.codes[query], frames.codes[candidate], metric="hamming", max_cells=_ALIGNMENT_CELLS
        )
    except ValueError as err:
        _logger.info(
            "Duplicates: no alignment for a %d and %d frame sequence pair; warping them is too large a problem (%s).",
            len(query),
            len(candidate),
            err,
        )
        return None
    window = found["end"] - found["start"] + 1
    # A warp with no band will collapse a whole sequence onto a handful of frames to save a few
    # bits and call the result a match. Holding the window to the same length bar a segment has to
    # clear is what keeps `aligned` a claim about a shared stretch rather than a shared moment.
    if found["normalized_cost"] > threshold or window < policy.min_length:
        return None
    query_frames = frame_map[frames.items[query], 1]
    candidate_frames = frame_map[frames.items[candidate], 1]
    return AlignmentMatch(
        query=int(frames.items[query[0]]),
        candidate=int(frames.items[candidate[0]]),
        query_span=(int(query_frames[0]), int(query_frames[-1])),
        candidate_span=(int(candidate_frames[found["start"]]), int(candidate_frames[found["end"]])),
        containment=(1.0, window / len(candidate)),
        normalized_cost=found["normalized_cost"],
        method=frames.method,
    )


_SEGMENT_OVERLAP = 0.5


def _overlap(first: tuple[int, int], second: tuple[int, int]) -> float:
    """Share of the shorter of two inclusive spans that the two have in common."""
    shared = min(first[1], second[1]) - max(first[0], second[0]) + 1
    shortest = min(first[1] - first[0], second[1] - second[0]) + 1
    return max(shared, 0) / shortest


def _dominant(segments: SegmentMatchResult) -> SegmentMatchResult:
    """Keep one segment per stretch, dropping the neighbouring diagonals that restate it.

    Consecutive video frames resemble each other, so frame *i* of one sequence matches not only
    frame *i + k* of the other but *i + k ± 1* as well. Each of those near-misses forms its own
    diagonal, and ``max_gap`` bridges their sparser matches into full-length runs -- so a single
    re-encode is reported four or five times, at offsets differing by a frame or two. They are all
    true and only one is the relation.

    Segments are taken best-first -- most matched rows, then closest -- and one is dropped only
    when a kept segment already covers most of *both* its spans. Requiring both is what keeps a
    stretch that genuinely repeats: the same minute of A appearing twice in B shares its query
    span with itself but sits in two different places in B, so both are reported.
    """
    order = sorted(
        range(len(segments["offset"])),
        key=lambda i: (-int(segments["n_matched"][i]), float(segments["mean_distance"][i])),
    )
    kept: list[int] = []
    for index in order:
        query = (int(segments["query_start"][index]), int(segments["query_end"][index]))
        candidate = (int(segments["candidate_start"][index]), int(segments["candidate_end"][index]))
        if any(
            _overlap(query, (int(segments["query_start"][other]), int(segments["query_end"][other])))
            >= _SEGMENT_OVERLAP
            and _overlap(candidate, (int(segments["candidate_start"][other]), int(segments["candidate_end"][other])))
            >= _SEGMENT_OVERLAP
            for other in kept
        ):
            continue
        kept.append(index)
    if len(kept) == len(order):
        return segments
    _logger.debug(
        "Duplicates: %d of %d segment(s) restated a stretch already reported",
        len(order) - len(kept),
        len(order),
    )
    keep = np.sort(np.array(kept, dtype=np.intp))
    return SegmentMatchResult(
        query_start=segments["query_start"][keep],
        query_end=segments["query_end"][keep],
        candidate_start=segments["candidate_start"][keep],
        candidate_end=segments["candidate_end"][keep],
        offset=segments["offset"][keep],
        n_matched=segments["n_matched"][keep],
        mean_distance=segments["mean_distance"][keep],
        density=segments["density"][keep],
    )


def _shared_stretches(runs: _HashedRuns, frame_map: NDArray[np.intp], policy: _SegmentPolicy) -> _SharedStretches:
    """Find what each pair of runs holds in common, at a fixed offset and under warping.

    A two-tier cascade, which is what keeps it affordable. A single grouping pass over *every* row
    says which runs share anything at all -- bounded by the row count, however duplicate-heavy the
    data. Only the pairs that survive it are searched pairwise for the matched rows a diagonal
    needs.

    With ``verify_alignment`` set, a pair the diagonal search could not explain is warped against
    its partner as well. Only such a pair: warping is quadratic where the diagonal search is
    near-linear, and a pair whose segments were already found needs no second account of itself.
    """
    members = {int(key): np.flatnonzero(runs.keys == key) for key in np.unique(runs.keys)}
    found: list[SegmentMatch] = []
    aligned: list[AlignmentMatch] = []
    for first, second in _sharing_runs(runs, policy.radius):
        left, right = members[first], members[second]
        crossed, distances = _crossing_pairs(runs, left, right, policy.radius)
        segments = (
            match_segments(
                crossed,
                distances,
                min_length=policy.min_length,
                max_gap=policy.max_gap,
                offset_tolerance=policy.offset_tolerance,
            )
            if len(crossed)
            else None
        )
        if segments is None or not len(segments["offset"]):
            match = _aligned_pair(runs, left, right, frame_map, policy)
            if match is not None:
                aligned.append(match)
            continue
        segments = _dominant(segments)
        found.append(
            SegmentMatch(
                query=int(runs.items[left[0]]),
                candidate=int(runs.items[right[0]]),
                segments=segments,
                containment=sequence_containment(crossed, len(left), len(right)),
                method=runs.method,
                # The matcher works in positions among the rows actually measured, which frame
                # sampling divorces from the frames of the source video. Carried back to source
                # positions here so a span reads in the same coordinates as `unit_indices`.
                query_frames=frame_map[runs.items[left], 1],
                candidate_frames=frame_map[runs.items[right], 1],
                keys=(first, second),
            )
        )
    return _SharedStretches(found, aligned)


def _find_shared_stretches(
    stats: StatsMap,
    source_index: Sequence[SourceIndex],
    frame_map: NDArray[np.intp] | None,
    policy: _SegmentPolicy,
    dataset_steps: Sequence[int] | None = None,
) -> _SharedStretches:
    """Find the stretches over which two sequences run together, and how much of each they cover.

    Raises
    ------
    ValueError
        If the segment policy is out of range. See :func:`_checked_segment_policy`.
    """
    _checked_segment_policy(policy)
    gathered = _sequence_frames(stats, source_index, frame_map, dataset_steps) if frame_map is not None else None
    if gathered is None or frame_map is None:
        return _SharedStretches([], [])
    return _shared_stretches(gathered, frame_map, policy)


class _TrackRuns(NamedTuple):
    """A dataset's detection crops gathered into the tracks they belong to.

    Attributes
    ----------
    runs : _HashedRuns
        One row per tracked detection, keyed by track. ``items`` names the *frame* each detection
        sits in, because a detection has no index of its own that a duplicate group could carry.
    tracks : NDArray[np.intp]
        Each row's own track id, as the annotation gives it -- not a dense re-numbering, so a
        reported track is the one a reader can look up in their own data.
    """

    runs: _HashedRuns
    tracks: NDArray[np.intp]


def _detection_tracks(
    source_index: Sequence[SourceIndex],
    track_map: NDArray[np.intp],
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Return which stats rows are tracked detections, and the track id of each.

    ``track_map`` runs in the order the detections were measured, so a detection's row in it is
    its frame's offset plus its index within that frame.
    """
    positions = np.array([i for i, index in enumerate(source_index) if index.key is not None], dtype=np.intp)
    if not len(positions):
        return positions, np.empty(0, dtype=np.intp)
    items = np.array([source_index[i].item for i in positions], dtype=np.intp)
    targets = np.array([source_index[i].key for i in positions], dtype=np.intp)
    offsets = np.searchsorted(track_map[:, 0], items, side="left")
    rows = offsets + targets
    inside = rows < len(track_map)
    ids = np.full(len(positions), -1, dtype=np.intp)
    ids[inside] = track_map[rows[inside], 1]
    return positions, ids


def _track_runs(
    stats: StatsMap,
    source_index: Sequence[SourceIndex],
    frame_map: NDArray[np.intp],
    track_map: NDArray[np.intp],
    dataset_steps: Sequence[int] | None,
) -> _TrackRuns | None:
    """Gather each track's detection crops, in frame order, or None if there are no tracks.

    An unlinked detection -- MAITE's ``track_id == -1`` -- has a frame but no track, and is left
    out rather than pooled. Pooling them makes one phantom track out of everything unassigned, and
    every unassigned detection then duplicates every other: the same failure as hashing an all-NaN
    region, and the rule the tracking structurer already applies when it builds track-level
    metadata.
    """
    method = next((name for name in _REDUNDANCY_METHODS if name in stats), None)
    if method is None or not len(track_map):
        return None
    positions, ids = _detection_tracks(source_index, track_map)
    keep = np.flatnonzero(ids >= 0)
    if not len(keep):
        return None
    positions, ids = positions[keep], ids[keep]
    items = np.array([source_index[i].item for i in positions], dtype=np.intp)
    # A track belongs to one sequence, so its identity is the sequence's own key paired with the
    # track id -- which is only unique within a sequence, and is reused freely across them.
    sequences = _run_keys(items, frame_map, dataset_steps)
    _, keys = np.unique(np.stack((sequences, ids), axis=1), axis=0, return_inverse=True)
    digests = [str(stats[method][i]) for i in positions]
    codes, valid = pack_hashes(digests)
    runs = _HashedRuns(
        method=method,
        digests=digests,
        items=items,
        keys=keys.astype(np.intp).reshape(-1),
        codes=codes,
        valid=valid,
        bits=next((len(digest) * 4 for digest in digests if digest), None),
    )
    _logger.debug("Duplicates: %d tracked detection(s) over %d track(s)", len(positions), len(np.unique(keys)))
    return _TrackRuns(runs=runs, tracks=ids)


def _find_track_duplicates(tracks: _TrackRuns) -> list[TrackGroup]:
    """Group tracks whose crops are the same, in the same order -- one object under two ids.

    A track none of whose crops could be measured takes no part, for the reason
    :func:`_find_sequence_duplicates` gives: every unmeasured crop packs to the same empty digest,
    so such tracks all fingerprint alike.
    """
    runs = tracks.runs
    digests: dict[str, list[tuple[int, int]]] = {}
    for key in np.unique(runs.keys):
        where = np.flatnonzero(runs.keys == key)
        if not runs.valid[where].any():
            continue
        fingerprint = sequence_fingerprint([runs.digests[index] for index in where])
        digests.setdefault(fingerprint["exact"], []).append((int(runs.items[where[0]]), int(tracks.tracks[where[0]])))
    groups: list[TrackGroup] = []
    for members in digests.values():
        if len(members) > 1:
            ordered = sorted(members)
            groups.append(([item for item, _ in ordered], runs.method, [track for _, track in ordered]))
    return groups


class _TrackRelations(NamedTuple):
    """What the tracks of a dataset have in common: whole duplicates, and shared stretches."""

    exact: list[TrackGroup]
    segments: list[SegmentMatch]


def _find_track_relations(
    stats: StatsMap,
    source_index: Sequence[SourceIndex],
    frame_map: NDArray[np.intp] | None,
    track_map: NDArray[np.intp] | None,
    policy: _SegmentPolicy,
    dataset_steps: Sequence[int] | None = None,
) -> _TrackRelations:
    """Find every track-level relation, or nothing where there are no tracked detections.

    One entry point for all three detection paths. They have drifted apart once per relation added
    to only one of them, and a single call is the cheapest guard against that.
    """
    if frame_map is None or track_map is None:
        return _TrackRelations([], [])
    tracks = _track_runs(stats, source_index, frame_map, track_map, dataset_steps)
    if tracks is None:
        return _TrackRelations([], [])
    return _TrackRelations(_find_track_duplicates(tracks), _find_track_stretches(tracks, frame_map, policy))


def _find_track_stretches(
    tracks: _TrackRuns,
    frame_map: NDArray[np.intp],
    policy: _SegmentPolicy,
) -> list[SegmentMatch]:
    """Find the stretches over which two tracks run together -- the same machinery one level down.

    Warping is deliberately not offered here. A speed edit is a property of a video, and a track
    is already confined to one; a warped track match would mean an object moving through the same
    poses at another rate, which is a resemblance rather than a duplicate.
    """
    runs = tracks.runs
    identity = {int(key): int(tracks.tracks[np.flatnonzero(runs.keys == key)[0]]) for key in np.unique(runs.keys)}
    stretches = _shared_stretches(runs, frame_map, policy._replace(verify_alignment=None))
    # Read off the run keys rather than the representative frames: a frame holding two tracked
    # objects belongs to two tracks, and would name whichever of them was gathered first.
    return [match._replace(tracks=(identity[match.keys[0]], identity[match.keys[1]])) for match in stretches.segments]


def _aligned_frame_map(
    source_index: Sequence[SourceIndex],
    frame_map: NDArray[np.intp] | None,
) -> NDArray[np.intp] | None:
    """Re-lay a several-dataset frame map so row *i* is the frame at combined item index *i*.

    The two are laid out on different counts. ``combine_stats_results`` offsets each dataset's item
    indices by the number of *stats rows* before it, which for a per-target run counts the
    detections too; the frame maps are simply concatenated, one row per frame. Reading the map by
    the item index therefore lands on another dataset's frames -- or off the end of the map
    entirely, which is how this surfaces.

    Rows that no measured frame claims are filled with ``-1`` and are never read. A single-dataset
    call has no offsets to reconcile and comes back unchanged.
    """
    if frame_map is None:
        return None
    items = np.unique(np.array([index.item for index in source_index if index.key is None], dtype=np.intp))
    if len(items) != len(frame_map) or (len(items) and int(items[-1]) == len(items) - 1):
        # Either nothing to align against, or the item indices already number the frames directly.
        return frame_map
    aligned = np.full((int(items[-1]) + 1, frame_map.shape[1]), -1, dtype=np.intp)
    aligned[items] = frame_map
    return aligned


class _FrameView(NamedTuple):
    """A dataset prepared for measurement, with the maps naming what each measured row is.

    Attributes
    ----------
    data : Any
        The dataset to measure -- a :class:`~dataeval.data.SequenceFrames` for tracking input,
        and the original object untouched for anything else.
    frame_map : NDArray[np.intp] or None
        Shape ``(F, 2)``, the ``(sequence, frame)`` behind each measured frame. None for images.
    track_map : NDArray[np.intp] or None
        Shape ``(D, 2)``, the ``(frame, track id)`` behind each measured detection. None for
        images, and unused unless detections were measured.
    """

    data: Any
    frame_map: NDArray[np.intp] | None
    track_map: NDArray[np.intp] | None


def _aligned_track_map(
    source_index: Sequence[SourceIndex],
    track_map: NDArray[np.intp] | None,
) -> NDArray[np.intp] | None:
    """Re-lay a track map's frame column into combined item indices, for the reason above.

    See :func:`_aligned_frame_map`: the map counts frames while the item indices count stats rows,
    so for a several-dataset per-target call the two disagree from the first dataset boundary on.
    """
    if track_map is None or not len(track_map):
        return track_map
    items = np.unique(np.array([index.item for index in source_index if index.key is None], dtype=np.intp))
    if not len(items) or int(items[-1]) == len(items) - 1:
        return track_map
    inside = track_map[:, 0] < len(items)
    aligned = track_map.copy()
    aligned[inside, 0] = items[track_map[inside, 0]]
    return aligned


#: Which level names each task understands. Image spellings are the ones that shipped; a tracking
#: dataset uses the names :class:`~dataeval.Metadata` gives its levels.
_IMAGE_LEVEL_NAMES = ("item", "target")
_TRACKING_LEVEL_NAMES = ("sequence", "unit", "track", "instance")

#: Levels whose relations are read from per-detection crop hashes rather than whole frames.
_TARGET_LEVELS = frozenset({"target", "instance", "track"})


class _DetectionPolicy(NamedTuple):
    """Everything that decides what a pass of hash-based detection looks for.

    Carried as one object because three call paths run that pass -- a first evaluation, a
    cross-dataset one, and a re-detection from stored statistics -- and every knob threaded
    separately is a knob one of them can be left without.
    """

    levels: frozenset[str]
    hash_radius: int
    redundancy_radius: int
    segment: _SegmentPolicy
    track: _SegmentPolicy


class _Relations(NamedTuple):
    """Every relation one pass of hash-based detection found, at whichever levels were asked for."""

    item_exact: list[Any]
    item_near: MethodGroups
    target_exact: list[Any]
    target_near: MethodGroups
    redundant: list[RedundantGroup]
    sequence_exact: MethodGroups
    stretches: _SharedStretches
    tracks: _TrackRelations

    @classmethod
    def empty(cls) -> "_Relations":
        """Return the nothing-found result, which is what a cluster-only run reports."""
        return cls([], [], [], [], [], [], _SharedStretches([], []), _TrackRelations([], []))


class _LevelPlan(NamedTuple):
    """Which levels to report, and the statistics that reporting them needs.

    ``levels`` and ``per_image``/``per_target`` are two spellings of one thing -- one names the
    answers wanted, the other the measurements taken -- so this resolves whichever was given into
    both. Passing both spellings at once is refused rather than reconciled.
    """

    per_image: bool
    per_target: bool
    levels: frozenset[str]


def _resolve_levels(
    levels: str | Sequence[str] | None,
    per_image: bool | None,
    per_target: bool | None,
    tracking: bool,
) -> _LevelPlan:
    """Resolve the requested levels and the statistics they need.

    Raises
    ------
    ValueError
        If both spellings are given, or a level name is not one this task has.
    """
    names = _TRACKING_LEVEL_NAMES if tracking else _IMAGE_LEVEL_NAMES
    if levels is not None and (per_image is not None or per_target is not None):
        raise ValueError(
            "Duplicates.evaluate: pass either `levels` or `per_image`/`per_target`, not both -- "
            "they are two spellings of one thing, and reconciling them silently would hide "
            f"whichever lost. `levels` for this dataset accepts {list(names)}."
        )
    if levels is None:
        return _implied_levels(bool(per_image if per_image is not None else True), bool(per_target), tracking)

    wanted = frozenset([levels] if isinstance(levels, str) else levels)
    unknown = sorted(wanted - set(names))
    if unknown:
        raise ValueError(
            f"Duplicates.evaluate: {unknown} {'is' if len(unknown) == 1 else 'are'} not a level of "
            f"{'a tracking' if tracking else 'an image'} dataset; it accepts {list(names)}."
        )
    if not wanted:
        raise ValueError("Duplicates.evaluate: `levels` must name at least one level.")
    # Frames are the substrate every tracking relation is built on -- a track is found among the
    # detections of frames, and the frame map that places them is built either way -- so they are
    # always measured. What `levels` decides there is which of them are reported.
    return _LevelPlan(
        per_image=True if tracking else bool(wanted - _TARGET_LEVELS),
        per_target=bool(wanted & _TARGET_LEVELS),
        levels=wanted,
    )


def _implied_levels(per_image: bool, per_target: bool, tracking: bool) -> _LevelPlan:
    """Resolve the older spelling, reproducing what each combination reports today."""
    if tracking:
        levels = {"sequence", "unit"} if per_image else set()
        if per_target:
            levels |= {"instance", "track"}
    else:
        levels = {"item"} if per_image else set()
        if per_target:
            levels |= {"target"}
    return _LevelPlan(per_image=per_image, per_target=per_target, levels=frozenset(levels))


def _find_relations(
    stats: StatsMap,
    source_index: Sequence[SourceIndex],
    frame_map: NDArray[np.intp] | None,
    track_map: NDArray[np.intp] | None,
    policy: _DetectionPolicy,
    dataset_steps: Sequence[int] | None = None,
) -> _Relations:
    """Find every relation the policy asks for, and none it does not.

    The single entry point all three detection paths share. They have drifted apart once per
    relation added to only one of them, so the level gating lives here rather than at each call
    site -- and a level that is not asked for is not searched for, which is also where the saving
    is: a caller wanting only whole-video relations pays for no frame-level grouping.
    """
    levels = policy.levels
    item, target = ("unit", "instance") if frame_map is not None else ("item", "target")
    found = _Relations.empty()

    if item in levels or target in levels:
        (item_exact, item_near), (target_exact, target_near) = _detect_hash_duplicates(
            stats, source_index, policy.hash_radius
        )
        found = found._replace(
            item_exact=list(item_exact) if item in levels else [],
            item_near=item_near if item in levels else [],
            target_exact=list(target_exact) if target in levels else [],
            target_near=target_near if target in levels else [],
        )
    if item in levels:
        found = found._replace(
            redundant=_find_redundant_runs(stats, source_index, frame_map, policy.redundancy_radius, dataset_steps)
        )
    if "sequence" in levels:
        found = found._replace(
            sequence_exact=_find_sequence_duplicates(stats, source_index, frame_map, dataset_steps),
            stretches=_find_shared_stretches(stats, source_index, frame_map, policy.segment, dataset_steps),
        )
    else:
        # Still checked, because a nonsense policy passed alongside a `levels` that happens to
        # exclude segments would otherwise be accepted in silence.
        _checked_segment_policy(policy.segment)
    if "track" in levels:
        found = found._replace(
            tracks=_find_track_relations(stats, source_index, frame_map, track_map, policy.track, dataset_steps)
        )
    return found


def _relations_frame(
    found: _Relations,
    available_stats: set[str],
    merge: bool,
    frame_map: NDArray[np.intp] | None,
    dataset_steps: Sequence[int] | None = None,
) -> pl.DataFrame:
    """Lay one pass of detection out as rows. Shared, so a new relation reaches every path at once."""
    return _build_duplicates_dataframe(
        found.item_exact or None,
        found.item_near,
        found.target_exact or None,
        found.target_near,
        available_stats,
        merge,
        dataset_steps=dataset_steps,
        frame_map=frame_map,
        redundant_groups=found.redundant,
        sequence_exact=found.sequence_exact,
        segment_matches=[*found.stretches.segments, *found.tracks.segments],
        alignment_matches=found.stretches.alignments,
        track_exact=found.tracks.exact,
    )


def _is_tracking(data: Any) -> bool:
    """Whether `data` is a tracking dataset, asked without raising on anything else.

    ``evaluate`` accepts a bare sequence of images as readily as a MAITE 3-tuple dataset, so this
    has to be a question rather than an assertion. The shape validators raise on anything that is
    not the kind they were asked about, which would reject the image-only inputs that are
    perfectly valid here.
    """
    if not isinstance(data, Sized) or len(data) == 0:
        return False
    datum = data[0] if isinstance(data, Dataset) else None
    return isinstance(datum, tuple) and len(datum) == 3 and _is_protocol_instance(datum[1], MultiobjectTrackingTarget)


def _warn_strict_radius(hash_radius: int, caller: str) -> None:
    """Say so when video meets the still-image default.

    A default carried over from still imagery is the most expensive mistake available here, and it
    is expensive precisely because it does not look like one: matching only bit-identical frames
    still finds a transcoded pair, it just reports the wrong stretch and a fraction of the real
    overlap. A leak measured at 60% of a test split when it is really 100% reads like a finding,
    not like a misconfiguration.

    Said out loud rather than corrected by dispatch, so what the call does stays predictable from
    what the call says. See :attr:`~dataeval.quality.Duplicates.hash_radius`.
    """
    if hash_radius:
        return
    _logger.warning(
        "%s: hash_radius=0 requires video frames to hash bit-identically, which survives almost "
        "no re-encode. Relations across transcoded copies are under-reported rather than missed "
        "-- spans and containment both read low. Pass hash_radius=6 for video.",
        caller,
    )


def _yields_frames(data: Any) -> bool:
    """Whether measuring `data` produces video frames, and so the tracking level names.

    True for a tracking dataset and for a frame view already built over one. The two arrive by
    different routes -- one gets wrapped here, the other was wrapped by the caller -- and asking
    only the first would give a hand-wrapped view the image level vocabulary, which then matches
    nothing.
    """
    return isinstance(data, SequenceFrames) or _is_tracking(data)


def _as_frames(
    data: Any,
    frame_sample: FrameSample,
    caller: str,
    hash_radius: int = DEFAULT_DUPLICATES_HASH_RADIUS,
) -> _FrameView:
    """Present a tracking dataset as its frames, leaving anything else alone.

    Returns the dataset to measure and, for a tracking dataset, the maps from each flattened frame
    position back to ``(sequence, frame)`` and from each detection to the track holding it. Image
    datasets are returned untouched with no maps, which is what keeps every existing call path
    byte-identical.
    """
    if isinstance(data, SequenceFrames):
        frames = data
        if frame_sample is not None:
            _logger.warning(
                "%s: frame_sample=%r is ignored because the dataset is already a SequenceFrames, "
                "whose own selector (%r) decides which frames take part.",
                caller,
                frame_sample,
                frames.selector,
            )
    elif not _is_tracking(data):
        return _FrameView(data, None, None)
    else:
        frames = SequenceFrames(data, _resolve_selector(frame_sample))

    _warn_strict_radius(hash_radius, caller)

    # Read before `n_dropped`, which a selector that cannot plan only learns by walking -- and
    # asking for the map is what walks it. Checked the other way round, the one warning that
    # matters is silent for exactly the selectors that drop the most.
    frame_map = frames.frame_map
    if frames.n_dropped:
        _logger.info(
            "%s: measuring %d of %d frame(s); %r dropped the rest. Every unweighted statistic "
            "downstream therefore describes the frames kept rather than the source stream -- "
            "each row's 'frames_represented' records what it stands for.",
            caller,
            len(frames),
            frames.n_source_frames,
            frames.selector,
        )
    return _FrameView(frames, frame_map, frames.track_map)


def _as_frames_multi(
    datasets: Sequence[Any],
    frame_sample: FrameSample,
    caller: str,
    hash_radius: int = DEFAULT_DUPLICATES_HASH_RADIUS,
) -> _FrameView:
    """Prepare several datasets together, refusing a mix of tracking and image data.

    The two cannot share one result: a frame view's item index names a frame and an image
    dataset's names an image, so the groups would address different things under one column.
    """
    prepared = [_as_frames(dataset, frame_sample, caller, hash_radius) for dataset in datasets]
    maps = [view.frame_map for view in prepared]
    if any(frame_map is None for frame_map in maps) != all(frame_map is None for frame_map in maps):
        raise ValueError(
            f"{caller}: cannot combine tracking datasets with image datasets in one call. A frame "
            "view's item index is a frame and an image dataset's is an image, so the resulting "
            "groups would address two different things under one column."
        )
    located = [frame_map for frame_map in maps if frame_map is not None]
    if not located:
        return _FrameView([view.data for view in prepared], None, None)
    # Each view numbers its own frames from zero, so a track map's frame column is shifted onto
    # the flat numbering the concatenated frame map uses before the two are read together.
    tracked: list[NDArray[np.intp]] = []
    offset = 0
    for view, frame_map in zip(prepared, maps, strict=True):
        if view.track_map is not None and len(view.track_map):
            shifted = view.track_map.copy()
            shifted[:, 0] += offset
            tracked.append(shifted)
        offset += len(frame_map) if frame_map is not None else 0
    return _FrameView(
        [view.data for view in prepared],
        np.concatenate(located),
        np.concatenate(tracked) if tracked else None,
    )


_PAIR_SCHEMA: dict[str, pl.DataType | type] = {
    "level": pl.Utf8,
    "dataset_a": pl.Int64,
    "dataset_b": pl.Int64,
    "item_a": pl.Int64,
    "item_b": pl.Int64,
    "track_a": pl.Int64,
    "track_b": pl.Int64,
    "relations": pl.List(pl.Utf8),
    "n_groups": pl.UInt32,
    "containment_a": pl.Float64,
    "containment_b": pl.Float64,
    "mean_distance": pl.Float64,
}

#: Pairs one ``aggregate_by_pair`` call may expand to. A group of n members implies n(n-1)/2, so a
#: single cluster of near-identical frames reaches this long before the row count does.
_MAX_PAIR_EXPANSION = 2_000_000


def _checked_levels(levels: str | Sequence[str] | None, frame: pl.DataFrame) -> list[str]:
    """Resolve the requested levels against the ones the result actually holds."""
    present = sorted(set(frame["level"].unique().to_list()))
    if levels is None:
        return present
    wanted = [levels] if isinstance(levels, str) else list(levels)
    unknown = [level for level in wanted if level not in present]
    if unknown:
        raise ValueError(
            f"aggregate_by_pair: {unknown} {'is' if len(unknown) == 1 else 'are'} not among the "
            f"levels these results hold ({present})."
        )
    return wanted


def _pair_members(row: Mapping[str, Any], cross: bool) -> list[tuple[int, int, int]]:
    """Return each member of a row as the ``(dataset, item, track)`` a pair is keyed by.

    A frame index is deliberately not part of the key: the useful summary of two videos sharing a
    hundred frames is one row saying so, not a hundred rows saying it once each.
    """
    items = row["item_indices"]
    datasets = row["dataset_indices"] if cross else [0] * len(items)
    tracks = row.get("track_indices") or [-1] * len(items)
    return [(int(dataset), int(item), int(track)) for dataset, item, track in zip(datasets, items, tracks, strict=True)]


def _pair_rows(rows: pl.DataFrame, cross: bool) -> pl.DataFrame:
    """Expand each group into the pairs it implies, then fold them onto one row per pair."""
    if rows.is_empty():
        return pl.DataFrame(schema=_PAIR_SCHEMA)
    sizes = rows["item_indices"].list.len().to_list()
    implied = sum(size * (size - 1) // 2 for size in sizes)
    if implied > _MAX_PAIR_EXPANSION:
        raise ValueError(
            f"aggregate_by_pair: these groups imply {implied} pairs, past the "
            f"{_MAX_PAIR_EXPANSION} this can return. Name a narrower `levels` -- frame-level "
            "groups over duplicate-heavy video are what grow quadratically."
        )

    folded: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows.iter_rows(named=True):
        _fold_row(folded, row, _pair_members(row, cross))
    return _pair_frame(folded, cross)


def _blank(level: str, a: tuple[int, int, int], b: tuple[int, int, int]) -> dict[str, Any]:
    """Return an unfilled row for one pair, before any group has been folded onto it."""
    return {
        "level": level,
        "dataset_a": a[0],
        "item_a": a[1],
        "track_a": a[2],
        "dataset_b": b[0],
        "item_b": b[1],
        "track_b": b[2],
        "relations": set(),
        "n_groups": 0,
        "containment_a": None,
        "containment_b": None,
        "mean_distance": None,
    }


def _fold_row(
    folded: dict[tuple[Any, ...], dict[str, Any]],
    row: Mapping[str, Any],
    members: Sequence[tuple[int, int, int]],
) -> None:
    """Fold every pair one group implies onto the running per-pair rows."""
    for left, right in combinations(range(len(members)), 2):
        first, second = members[left], members[right]
        if first == second:
            # A group holding two frames of one video relates it to itself, which the redundancy
            # relation already reports and which names no pair. Compared on the whole key rather
            # than the sequence alone: two *tracks* of one video are a pair, and the pair R7
            # exists to find.
            continue
        flip = second < first
        a, b = (second, first) if flip else (first, second)
        entry = folded.setdefault((row["level"], *a, *b), _blank(row["level"], a, b))
        entry["relations"].add(row["dup_type"])
        entry["n_groups"] += 1
        _carry(entry, row.get("containment"), left, right, flip, row.get("mean_distance"))


def _carry(
    entry: dict[str, Any],
    containment: Sequence[float] | None,
    left: int,
    right: int,
    flip: bool,
    distance: float | None,
) -> None:
    """Fold one group's directed figures onto the pair, following the canonical ordering."""
    if containment is not None and containment[left] is not None:
        first, second = float(containment[left]), float(containment[right])
        if flip:
            first, second = second, first
        entry["containment_a"] = first if entry["containment_a"] is None else max(entry["containment_a"], first)
        entry["containment_b"] = second if entry["containment_b"] is None else max(entry["containment_b"], second)
    if distance is not None:
        current = entry["mean_distance"]
        entry["mean_distance"] = float(distance) if current is None else min(current, float(distance))


def _pair_frame(folded: Mapping[tuple[Any, ...], dict[str, Any]], cross: bool) -> pl.DataFrame:
    """Assemble the folded pairs, dropping the columns this result has no use for."""
    if not folded:
        return pl.DataFrame(schema=_PAIR_SCHEMA)
    records = []
    for entry in folded.values():
        record = dict(entry)
        record["relations"] = sorted(record["relations"])
        records.append(record)
    frame = pl.DataFrame(records, schema=_PAIR_SCHEMA)
    drop = [] if cross else ["dataset_a", "dataset_b"]
    if frame["track_a"].eq(-1).all():
        drop += ["track_a", "track_b"]
    return frame.drop(drop).sort(["level", "n_groups", "item_a", "item_b"], descending=[False, True, False, False])


class DuplicatesOutput(DataFrameOutput, Generic[TExactDuplicatesGroup, TNearDuplicatesGroup]):
    """
    Output class for :class:`.Duplicates` detector.

    Wraps a Polars DataFrame of duplicate groups with aggregation helpers
    and threshold-based redetection for cluster duplicates.

    **Which question, which call.** The frame is organized by *group*, which answers some
    questions directly and others only after a reshape.

    .. list-table::
       :header-rows: 1
       :widths: 45 55

       * - What you want to know
         - Where to look
       * - Which items duplicate which, and how much of each?
         - :meth:`aggregate_by_pair`
       * - Is content shared across my splits?
         - ``result.crossing.aggregate_by_pair()``
       * - How much does each video repeat itself, or the corpus?
         - :meth:`aggregate_by_sequence`
       * - Which items should I look at first?
         - :meth:`aggregate_by_image`
       * - Which detector found this?
         - :meth:`aggregate_by_method`
       * - Just the whole-video / frame / track / detection relations
         - :attr:`sequences`, :attr:`frames`, :attr:`tracks`, :attr:`detections`
       * - The groups as plain indices, to act on
         - :attr:`exact`, :attr:`near`
       * - Everything, unreshaped
         - :meth:`data`

    None of these decide anything. A duplicate is evidence of redundancy, not an instruction to
    delete: ``containment`` and ``redundant_fraction`` are reported so the cutoff stays yours.

    DataFrame of duplicate groups with columns:

    - group_id: int - Auto-incrementing ID for each duplicate group
    - level: str - ``"item"`` or ``"target"``
    - dup_type: str - ``"exact"`` or ``"near"``
    - item_indices: list[int] - Item indices of members in the group
    - target_indices: list[int] - Target indices within items (only when target-level
      groups exist, positionally aligned with item_indices)
    - methods: list[str] - Detection method names (e.g., ``["phash", "dhash"]``)
    - orientation: str | None - ``"same"``, ``"rotated"``, or None (only present
      when both basic and D4 hashes were computed)
    - dataset_indices: list[int] - Dataset indices for cross-dataset results (only
      present for multi-dataset output, positionally aligned with item_indices)

    Attributes
    ----------
    calculation_results : StatsResult or Sequence[StatsResult] or None
        The original hash statistics. Used internally for redetection via
        :meth:`with_threshold`.
    cluster_result : ClusterResult or None
        The clustering result (MST + cluster assignments). Used internally
        for redetection via :meth:`with_threshold`.
    cluster_sensitivity : float or None
        Factor used for cluster-based near duplicate detection.
        Scales the cluster standard deviation to set the duplicate cutoff.
    merge_near_duplicates : bool
        Whether overlapping near duplicate groups were merged.
    flags : ImageStats
        The hash statistics flags used for detection.
    hash_radius : int
        Hamming radius, in bits, used for hash-based near duplicate detection.
    redundancy_radius : int
        Hamming radius, in bits, used for temporal redundancy between consecutive video frames.
    min_segment_frames, max_segment_gap, segment_offset_tolerance : int
        The shared-stretch policy these results were found under. Kept so a re-detection finds
        the same segments the original did.
    verify_alignment : int or None
        Mean bits per frame two sequences could differ by and still be reported as aligned, or
        None if warped matching was off.
    min_track_frames : int
        Shortest stretch two tracks may share and still be reported, in detections.
    levels : frozenset[str]
        The levels these results were asked for. Kept so a re-detection reports the same ones.
    frame_map : NDArray[np.intp] or None
        For results from a video dataset, the ``(sequence, frame)`` behind each measured frame.
        None for image datasets.
    track_map : NDArray[np.intp] or None
        For results from a video dataset, the ``(frame, track id)`` behind each measured
        detection. None for image datasets, and for a run that measured no detections.
    """

    def __init__(
        self,
        data: pl.DataFrame,
        *,
        calculation_results: StatsResult | Sequence[StatsResult] | None = None,
        cluster_result: ClusterResult | None = None,
        cluster_sensitivity: float | None = None,
        merge_near_duplicates: bool = True,
        flags: ImageStats = ImageStats.NONE,
        hash_radius: int = DEFAULT_DUPLICATES_HASH_RADIUS,
        redundancy_radius: int = DEFAULT_DUPLICATES_REDUNDANCY_RADIUS,
        min_segment_frames: int = DEFAULT_DUPLICATES_MIN_SEGMENT_FRAMES,
        max_segment_gap: int = DEFAULT_DUPLICATES_MAX_SEGMENT_GAP,
        segment_offset_tolerance: int = DEFAULT_DUPLICATES_SEGMENT_OFFSET_TOLERANCE,
        verify_alignment: int | None = DEFAULT_DUPLICATES_VERIFY_ALIGNMENT,
        min_track_frames: int = DEFAULT_DUPLICATES_MIN_TRACK_FRAMES,
        levels: frozenset[str] | None = None,
        frame_map: NDArray[np.intp] | None = None,
        track_map: NDArray[np.intp] | None = None,
    ) -> None:
        super().__init__(data)
        self.calculation_results = calculation_results
        self.cluster_result = cluster_result
        self.cluster_sensitivity = cluster_sensitivity
        self.merge_near_duplicates = merge_near_duplicates
        self.flags = flags
        self.hash_radius = hash_radius
        self.redundancy_radius = redundancy_radius
        self.min_segment_frames = min_segment_frames
        self.max_segment_gap = max_segment_gap
        self.segment_offset_tolerance = segment_offset_tolerance
        self.verify_alignment = verify_alignment
        self.min_track_frames = min_track_frames
        default = _TRACKING_LEVEL_NAMES if frame_map is not None else _IMAGE_LEVEL_NAMES
        self.levels: frozenset[str] = frozenset(default) if levels is None else levels
        self.frame_map = frame_map
        self.track_map = track_map

    def _segment_policy(self, hash_radius: int | None = None) -> _SegmentPolicy:
        """Rebuild the shared-stretch policy these results were found under."""
        return _SegmentPolicy(
            radius=self.hash_radius if hash_radius is None else hash_radius,
            min_length=self.min_segment_frames,
            max_gap=self.max_segment_gap,
            offset_tolerance=self.segment_offset_tolerance,
            verify_alignment=self.verify_alignment,
            track_length=self.min_track_frames,
        )

    def _detection_policy(self, hash_radius: int | None = None) -> _DetectionPolicy:
        """Rebuild the detection policy these results were found under."""
        return _DetectionPolicy(
            levels=self.levels,
            hash_radius=self.hash_radius if hash_radius is None else hash_radius,
            redundancy_radius=self.redundancy_radius,
            segment=self._segment_policy(hash_radius),
            track=self._track_policy(hash_radius),
        )

    def _track_policy(self, hash_radius: int | None = None) -> _SegmentPolicy:
        """Return the same policy one level down, where a shared stretch is measured in detections."""
        policy = self._segment_policy(hash_radius)
        return policy._replace(min_length=policy.track_length, verify_alignment=None)

    def __len__(self) -> int:
        """Return the number of duplicate groups."""
        return self.data().shape[0]

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @overload
    def _get_groups(
        self,
        dup_type: Literal["exact"],
    ) -> ExactDuplicatesGroup: ...

    @overload
    def _get_groups(
        self,
        dup_type: Literal["near"],
    ) -> NearDuplicatesGroup: ...

    def _get_groups(
        self,
        dup_type: Literal["exact", "near"],
    ) -> ExactDuplicatesGroup | NearDuplicatesGroup:
        """Return duplicate groups of the given type as simple data structures.

        For exact duplicates:

          - Single-dataset without targets: ``list[list[int]]``
          - Single-dataset with targets: ``list[list[SourceIndex]]``
          - Cross-dataset: wraps the above in a ``dict`` keyed by dataset index.

        For near duplicates, each group is a ``tuple[indices, methods]`` where
        ``methods`` is the ``list[str]`` of detection methods (reasons) that
        flagged the group:

          - Single-dataset without targets: ``list[tuple[list[int], list[str]]]``
          - Single-dataset with targets: ``list[tuple[list[SourceIndex], list[str]]]``
          - Cross-dataset: wraps the above in a ``dict`` keyed by dataset index.
        """
        is_cross = "dataset_indices" in self.data().columns
        has_targets = "target_indices" in self.data().columns
        is_near = dup_type == "near"

        filtered = self.data().filter(pl.col("dup_type") == dup_type)
        filtered = self._one_level(filtered)

        if is_cross:
            return _get_groups_cross(filtered, has_targets, is_near)
        return _get_groups_single(filtered, has_targets, is_near)

    def _one_level(self, filtered: pl.DataFrame) -> pl.DataFrame:
        """Keep a tracking result's groups to one level, so their members are one shape.

        A sequence names a video and a frame names ``(sequence, frame)``, so a list holding both
        cannot be walked without asking each element what it is. Image results are left alone: an
        item and a target are both a :class:`~dataeval.types.SourceIndex` there, so mixing them
        stays navigable -- and that behaviour shipped.

        The narrowed views reach the rest: :attr:`sequences`, :attr:`tracks` and
        :attr:`detections` each hold one level already, so ``.sequences.exact`` is how a
        sequence-level group is asked for.
        """
        if self.frame_map is None or filtered.is_empty():
            return filtered
        present = set(filtered["level"].unique().to_list())
        if len(present) < 2:
            return filtered
        return filtered.filter(pl.col("level") == _TRACKING_LEVELS["item"])

    def _filtered_by_level(self, *levels: str) -> Self:
        """Return a new DuplicatesOutput holding only rows at one of the given levels.

        The view's ``levels`` narrow with it, so a re-detection off the view stays at the level it
        was taken at: ``result.sequences.with_radius(8)`` is still about sequences.
        """
        wanted = set(levels)
        return self._with(
            self.data().filter(pl.col("level").is_in(list(levels))),
            levels=frozenset(self.levels & wanted) or frozenset(wanted),
        )

    def _with(self, frame: pl.DataFrame, levels: frozenset[str] | None = None) -> Self:
        """Return these results over a narrowed frame, carrying every setting forward.

        The settings travel because a narrowed view is still re-detectable and still has to know
        what it was measured under.
        """
        return type(self)(  # type: ignore[return-value]
            frame,
            calculation_results=self.calculation_results,
            cluster_result=self.cluster_result,
            cluster_sensitivity=self.cluster_sensitivity,
            merge_near_duplicates=self.merge_near_duplicates,
            flags=self.flags,
            hash_radius=self.hash_radius,
            redundancy_radius=self.redundancy_radius,
            min_segment_frames=self.min_segment_frames,
            max_segment_gap=self.max_segment_gap,
            segment_offset_tolerance=self.segment_offset_tolerance,
            verify_alignment=self.verify_alignment,
            min_track_frames=self.min_track_frames,
            levels=self.levels if levels is None else levels,
            frame_map=self.frame_map,
            track_map=self.track_map,
        )

    @property
    def items(self) -> Self:
        """Filtered DuplicatesOutput containing only groups at the task's item level.

        ``item`` for an image dataset, ``unit`` -- a frame -- for a tracking one, so this reads
        "one piece of media" for either. The returned object supports the same properties
        (``exact``, ``near``) and aggregation methods as the original output.
        """
        return self._filtered_by_level("item", "unit")

    @property
    def targets(self) -> Self:
        """Filtered DuplicatesOutput containing only groups at the task's label level.

        ``target`` for an image dataset, ``instance`` -- a detection within a frame -- for a
        tracking one.
        """
        return self._filtered_by_level("target", "instance")

    @property
    def sequences(self) -> Self:
        """Filtered DuplicatesOutput containing only whole-sequence relations.

        Video only, and empty for an image dataset. These are the rows whose members are whole
        videos rather than frames inside one: ``dup_type="exact"`` for two videos holding the same
        frames in the same order, and ``dup_type="segment"`` for a stretch two videos share.
        """
        return self._filtered_by_level("sequence")

    @property
    def frames(self) -> Self:
        """Filtered DuplicatesOutput containing only frame-level duplicate groups.

        The tracking-facing spelling of :attr:`items`, empty for an image dataset.
        """
        return self._filtered_by_level("unit")

    @property
    def crossing(self) -> Self:
        """Filtered DuplicatesOutput holding only relations that cross a dataset boundary.

        The split-boundary view. A relation whose members all come from one dataset is ordinary
        redundancy; one whose members span two is the same content on both sides of a split, which
        is what makes a held-out score stop measuring what it claims to.

        Structural, not a judgement: this says *which relations cross*, and leaves how much overlap
        matters to the reader. Pair it with
        :meth:`~dataeval.quality.DuplicatesOutput.aggregate_by_pair` to read how much of each side
        the other accounts for::

            result = Duplicates(hash_radius=6).evaluate(train, test)
            result.crossing.aggregate_by_pair("sequence")

        Empty for a single-dataset result, which has no boundary to cross.

        Unlike the level views, this narrows *rows* rather than what was detected, so it does not
        survive a re-detection: take it after :meth:`with_radius`, not before.
        """
        frame = self.data()
        if "dataset_indices" not in frame.columns:
            return self._with(frame.clear())
        return self._with(frame.filter(pl.col("dataset_indices").list.n_unique() > 1))

    @property
    def tracks(self) -> Self:
        """Filtered DuplicatesOutput containing only track-level duplicate groups.

        One object annotated under two track ids, or a track carried along with a reused clip.
        Populated only when the evaluation asked for detections with ``per_target=True``, which is
        where the crop hashes a track is matched by come from.
        """
        return self._filtered_by_level("track")

    @property
    def detections(self) -> Self:
        """Filtered DuplicatesOutput containing only detection-level duplicate groups.

        The tracking-facing spelling of :attr:`targets`, empty for an image dataset.
        """
        return self._filtered_by_level("instance")

    @property
    def exact(self) -> TExactDuplicatesGroup:
        """Exact duplicate groups as lists of indices.

        - For single-dataset item results: ``list[list[int]]``
        - For single-dataset target results: ``list[list[SourceIndex]]``
        - For cross-dataset item results: ``dict[int, list[list[int]]]``
        - For cross-dataset target results: ``dict[int, list[list[SourceIndex]]]``
        """
        return self._get_groups("exact")  # type: ignore[return-value]

    @property
    def near(self) -> TNearDuplicatesGroup:
        """Near-duplicate groups as ``(indices, methods)`` tuples.

        Each group is a tuple of ``(indices, methods)`` where ``methods`` is
        the ``list[str]`` of detection methods that flagged the group.

        - For single-dataset item results: ``list[tuple[list[int], list[str]]]``
        - For single-dataset target results: ``list[tuple[list[SourceIndex], list[str]]]``
        - For cross-dataset item results: ``dict[int, list[tuple[list[int], list[str]]]]``
        - For cross-dataset target results: ``dict[int, list[tuple[list[SourceIndex], list[str]]]]``
        """
        return self._get_groups("near")  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Aggregation methods
    # ------------------------------------------------------------------

    def aggregate_by_image(self) -> pl.DataFrame:
        """Return a DataFrame listing each unique image involved in duplicates.

        Explodes item_indices so each image appears once, with counts and
        metadata about which groups and methods flagged it.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns:

            - item_index: int - The image index, or the sequence index for a video dataset
            - unit_index: int - Video datasets only: the frame's position in its sequence
            - group_count: int - Number of duplicate groups this image appears in
            - dup_types: list[str] - Unique duplicate types for this image
            - methods: list[str] - All unique methods that detected this image

        Notes
        -----
        A video dataset's rows are per **frame**, which takes two columns to name: a sequence
        index alone would fold every frame of one video onto one row and count a video's groups
        rather than a frame's.
        """
        if "dataset_indices" in self.data().columns:
            raise ValueError("aggregate_by_image only works with output from a single dataset.")

        # A frame is addressed by its sequence *and* its position within it, so both columns key
        # the grouping when they are there.
        has_units = "unit_indices" in self.data().columns
        keys = ["item_index", "unit_index"] if has_units else ["item_index"]
        schema: Any = {
            "item_index": pl.Int64,
            **({"unit_index": pl.Int64} if has_units else {}),
            "group_count": pl.UInt32,
            "dup_types": pl.List(pl.Utf8),
            "methods": pl.List(pl.Utf8),
        }

        # A whole-sequence row names videos rather than frames, so it carries no unit index at all.
        # Dropped rather than filled: exploding it beside a populated `item_indices` is a length
        # mismatch polars refuses outright, and a per-frame summary is not what such a row is about.
        rows = self.data().filter(pl.col("unit_indices").is_not_null()) if has_units else self.data()
        if rows.shape[0] == 0:
            return pl.DataFrame(schema=schema)

        columns = ["item_indices", "unit_indices"] if has_units else ["item_indices"]
        exploded = rows.explode(columns).rename(dict(zip(columns, keys, strict=True)))

        return (
            exploded
            .group_by(keys)
            .agg(
                pl.len().cast(pl.UInt32).alias("group_count"),
                pl.col("dup_type").unique().sort().alias("dup_types"),
                pl.col("methods").explode().unique().sort().alias("methods"),
            )
            .select(list(schema))
            .sort(["group_count", *keys], descending=[True, *(False for _ in keys)])
        )

    def aggregate_by_group(self) -> pl.DataFrame:
        """Return a DataFrame summarizing each duplicate group.

        Adds a member_count column showing the size of each group.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns:

            - group_id: int - Group identifier
            - level: str - ``"item"`` or ``"target"``
            - dup_type: str - ``"exact"`` or ``"near"``
            - member_count: int - Number of members in the group
            - methods: list[str] - Detection methods
            - orientation: str | None - Only present when both basic and D4
              hashes were computed
        """
        has_orientation = "orientation" in self.data().columns

        schema: Any = {
            "group_id": pl.Int64,
            "level": pl.Utf8,
            "dup_type": pl.Utf8,
            "member_count": pl.UInt32,
            "methods": pl.List(pl.Utf8),
        }
        if has_orientation:
            schema["orientation"] = pl.Utf8

        if self.data().shape[0] == 0:
            return pl.DataFrame(schema=schema)

        select_cols: list[Any] = [
            "group_id",
            "level",
            "dup_type",
            pl.col("item_indices").list.len().cast(pl.UInt32).alias("member_count"),
            "methods",
        ]
        if has_orientation:
            select_cols.append("orientation")

        return self.data().select(select_cols).sort("group_id")

    def aggregate_by_pair(self, levels: str | Sequence[str] | None = None) -> pl.DataFrame:
        """Summarize each *pair* of things that duplicate one another, one row per pair.

        The relation-shaped view of the result. The main frame is organized by *group*, which suits
        a transitive relation like an exact match and suits a directed one like a contained clip
        badly: reading "which of these two is inside the other" means knowing that ``containment``
        aligns positionally with ``item_indices``. Here the two sides are separate columns.

        Parameters
        ----------
        levels : str or Sequence[str] or None, default None
            Which levels to pair. None uses every level in the result. Naming a level is also how
            a large frame-level result is kept affordable -- see Raises.

        Returns
        -------
        pl.DataFrame
            One row per pair, with columns:

            - level: str - The level the two sides sit at
            - item_a, item_b: int - The two items, ordered so a pair appears once
            - dataset_a, dataset_b: int - Cross-dataset results only
            - track_a, track_b: int - Track-level rows only
            - relations: list[str] - The ``dup_type`` values linking the two
            - n_groups: int - Groups linking them; at frame level, frames they share
            - containment_a, containment_b: float - How much of each side the other accounts for,
              where a directed relation reports it
            - mean_distance: float - The closest evidence linking the two

        Raises
        ------
        ValueError
            If ``levels`` names a level the result does not hold, or if pairing would expand past
            what this can return. A group of *n* members implies *n(n-1)/2* pairs, so one group of
            several thousand near-identical frames is millions of rows on its own. Naming a
            narrower ``levels`` is the way through.

        See Also
        --------
        :meth:`~dataeval.quality.DuplicatesOutput.aggregate_by_group` : Per group rather than per pair
        :meth:`~dataeval.quality.DuplicatesOutput.aggregate_by_sequence` : Per sequence, for video

        Notes
        -----
        **The asymmetry is the signal.** ``containment_a`` near 1.0 against a low
        ``containment_b`` means *a is contained in b* — a clip cut from a longer source, and across
        splits, leakage. Two high values mean each is most of the other, which is ordinary
        redundancy. Filtering ``dataset_a != dataset_b`` is how a cross-dataset call asks the
        leakage question.

        ``redundant`` rows take no part: a run of repeated frames relates a sequence to itself, and
        so names no pair. Pairs of one item with itself are dropped for the same reason.

        Examples
        --------
        >>> Duplicates().evaluate(train, test).aggregate_by_pair("sequence")  # doctest: +SKIP
        """
        frame = self.data()
        if frame.is_empty():
            return pl.DataFrame(schema=_PAIR_SCHEMA)
        wanted = _checked_levels(levels, frame)
        rows = frame.filter(pl.col("dup_type") != "redundant").filter(pl.col("level").is_in(wanted))
        return _pair_rows(rows, cross="dataset_indices" in frame.columns)

    def aggregate_by_sequence(self) -> pl.DataFrame:
        """Summarize each video sequence: how much of it repeats itself, and how much is copied.

        Returns
        -------
        pl.DataFrame
            One row per sequence that contributed frames, with columns:

            - sequence: int - The sequence's index in the source dataset
            - n_frames: int - Frames measured from it, after any frame selection
            - redundant_frames: int - Frames inside a redundant run, less one representative each
            - redundant_fraction: float - ``redundant_frames / n_frames``
            - longest_run: int - Frames in this sequence's longest redundant run
            - duplicate_frames: int - Distinct frames of this sequence that also appear in
              *another* sequence
            - shared_with: int - How many other sequences this one shares content with
            - group_count: int - Duplicate groups touching this sequence

        Raises
        ------
        ValueError
            If this output did not come from a video dataset, or came from several at once --
            sequence 0 of one dataset is not sequence 0 of the next, and a per-sequence row
            cannot say which it is.

        See Also
        --------
        :meth:`~dataeval.quality.DuplicatesOutput.aggregate_by_image` : Per frame rather than per sequence

        Notes
        -----
        ``redundant_fraction`` is the headline figure: a sequence reading 0.9 holds ten frames'
        worth of information for every hundred frames it costs. It counts every member of a
        redundant run except one representative, which is what could be dropped without losing
        content -- not what *should* be, since dwell time can itself be the signal.

        ``duplicate_frames`` counts distinct frames, so a frame in several groups is counted once.
        It counts only frames shared with a *different* sequence: a frame that merely resembles its
        own neighbour is this sequence repeating itself, which ``redundant_frames`` already
        reports. Counting both would make a corpus of unrelated videos read as wholly duplicated,
        since consecutive frames of any video resemble one another at a useful ``hash_radius``.

        The two columns answer different questions, and both are worth asking: ``redundant_frames``
        is *what this video costs you for nothing*, and ``duplicate_frames`` with ``shared_with``
        is *what it has in common with the rest of the corpus*.

        ``longest_run`` beside ``redundant_fraction`` separates two very different sequences that
        score the same. A high fraction made of one long run is a camera holding still, and dropping
        it costs one scene; the same fraction spread over many short runs is footage that simply
        moves slowly, where the redundancy is the pace of the scene rather than a stare.

        Examples
        --------
        >>> Duplicates().evaluate(video_dataset).aggregate_by_sequence()  # doctest: +SKIP
        """
        if self.frame_map is None:
            raise ValueError(
                "aggregate_by_sequence() requires results from a multi-object-tracking dataset; "
                "an image dataset has no sequences. Use aggregate_by_image() instead."
            )
        if "dataset_indices" in self.data().columns:
            raise ValueError(
                "aggregate_by_sequence() only works with output from a single dataset: sequence 0 "
                "of one dataset is not sequence 0 of the next, so one row per sequence cannot say "
                "which dataset it summarizes."
            )

        measured = pl.DataFrame({"sequence": self.frame_map[:, 0]}).group_by("sequence").len("n_frames")
        schema: Any = {
            "sequence": measured.schema["sequence"],
            "n_frames": pl.UInt32,
            "redundant_frames": pl.UInt32,
            "redundant_fraction": pl.Float64,
            "longest_run": pl.UInt32,
            "duplicate_frames": pl.UInt32,
            "shared_with": pl.UInt32,
            "group_count": pl.UInt32,
        }
        if self.data().shape[0] == 0:
            counted = measured.with_columns(
                pl.lit(0, pl.UInt32).alias("redundant_frames"),
                pl.lit(0.0).alias("redundant_fraction"),
                pl.lit(0, pl.UInt32).alias("longest_run"),
                pl.lit(0, pl.UInt32).alias("duplicate_frames"),
                pl.lit(0, pl.UInt32).alias("shared_with"),
                pl.lit(0, pl.UInt32).alias("group_count"),
            )
            return counted.select(list(schema)).sort("sequence")

        # A whole-sequence row names videos rather than frames: it has no unit index to explode
        # against, and its members are already sequences. It still touches the sequences it names,
        # so it counts toward `group_count` -- but a frame count cannot be drawn from it.
        touched = self.data().explode("item_indices").rename({"item_indices": "sequence"})
        exploded = (
            self
            .data()
            .filter(pl.col("unit_indices").is_not_null())
            .explode(["item_indices", "unit_indices"])
            .rename({"item_indices": "sequence"})
        )
        # A run of k frames could lose k - 1 of them, so redundancy is counted per group rather
        # than per frame; duplicates are counted per distinct frame, since one frame can sit in
        # several groups.
        redundant = (
            exploded
            .filter(pl.col("dup_type") == "redundant")
            .group_by(["sequence", "group_id"])
            .agg((pl.len() - 1).alias("droppable"), pl.len().alias("run"))
            .group_by("sequence")
            .agg(
                pl.col("droppable").sum().cast(pl.UInt32).alias("redundant_frames"),
                pl.col("run").max().cast(pl.UInt32).alias("longest_run"),
            )
        )
        # Only groups reaching *another* sequence count as duplication. A frame that merely
        # resembles its own neighbour is this sequence repeating itself, which `redundant_frames`
        # already reports -- counting it here too makes an unrelated corpus read as wholly
        # duplicated, which is the opposite of what the column is for.
        shared = exploded.filter(pl.col("dup_type") != "redundant")
        crossing = shared.join(
            shared.group_by("group_id").agg(pl.col("sequence").n_unique().alias("_reach")),
            on="group_id",
            how="left",
        ).filter(pl.col("_reach") > 1)
        duplicated = crossing.group_by("sequence").agg(
            pl.col("unit_indices").n_unique().cast(pl.UInt32).alias("duplicate_frames")
        )
        # Partners come from every row, not just the frame-level ones: a whole-sequence match
        # names two videos and carries no frame index to explode.
        reach = touched.filter(pl.col("dup_type") != "redundant").select("group_id", "sequence").unique()
        partners = (
            reach
            .join(reach, on="group_id", how="inner", suffix="_other")
            .filter(pl.col("sequence") != pl.col("sequence_other"))
            .group_by("sequence")
            .agg(pl.col("sequence_other").n_unique().cast(pl.UInt32).alias("shared_with"))
        )
        groups = touched.group_by("sequence").agg(pl.col("group_id").n_unique().cast(pl.UInt32).alias("group_count"))

        return (
            measured
            .join(redundant, on="sequence", how="left")
            .join(duplicated, on="sequence", how="left")
            .join(partners, on="sequence", how="left")
            .join(groups, on="sequence", how="left")
            .with_columns(
                pl.col("redundant_frames").fill_null(0).cast(pl.UInt32),
                pl.col("longest_run").fill_null(0).cast(pl.UInt32),
                pl.col("duplicate_frames").fill_null(0).cast(pl.UInt32),
                pl.col("shared_with").fill_null(0).cast(pl.UInt32),
                pl.col("group_count").fill_null(0).cast(pl.UInt32),
            )
            .with_columns((pl.col("redundant_frames") / pl.col("n_frames")).alias("redundant_fraction"))
            .select(list(schema))
            .sort("sequence")
        )

    def aggregate_by_method(self) -> pl.DataFrame:
        """Return a DataFrame summarizing duplicate counts per detection method.

        Explodes the methods list so each method is counted individually.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns:

            - method: str - Detection method name
            - group_count: int - Number of groups detected by this method
            - total_members: int - Total members across those groups
        """
        schema: Any = {
            "method": pl.Utf8,
            "group_count": pl.UInt32,
            "total_members": pl.UInt32,
        }

        if self.data().shape[0] == 0:
            return pl.DataFrame(schema=schema)

        with_count = self.data().with_columns(
            pl.col("item_indices").list.len().alias("_member_count"),
        )

        return (
            with_count
            .explode("methods")
            .rename({"methods": "method"})
            .group_by("method")
            .agg(
                pl.len().cast(pl.UInt32).alias("group_count"),
                pl.col("_member_count").sum().cast(pl.UInt32).alias("total_members"),
            )
            .sort(["group_count", "method"], descending=[True, False])
        )

    # ------------------------------------------------------------------
    # Redetection
    # ------------------------------------------------------------------

    def with_sensitivity(self, cluster_sensitivity: float) -> Self:
        """Re-detect cluster-based duplicates with a different distance factor.

        Hash-based duplicates are deterministic and are not affected.
        Only cluster-based near duplicates are recomputed using the stored
        clustering result (MST + cluster assignments).

        Parameters
        ----------
        cluster_sensitivity : float
            Controls how aggressively points are considered duplicates by
            scaling the cluster's standard deviation. An edge is flagged as
            a duplicate link when its distance is below
            ``cluster_sensitivity * cluster_std``. Lower values are stricter
            (fewer near duplicates), higher values are more sensitive.
            Typical range: 0.1 – 3.0. Must be positive.

        Returns
        -------
        DuplicatesOutput
            New output with re-detected duplicates using the new distance factor.

        Raises
        ------
        ValueError
            If this output was not created from an evaluation with cluster results.
        """
        if self.cluster_result is None:
            raise ValueError("with_sensitivity() requires cluster results stored from evaluate() or from_clusters().")
        return self._redetect(cluster_sensitivity=cluster_sensitivity)

    def with_radius(self, hash_radius: int) -> Self:
        """Re-detect hash-based near duplicates at a different Hamming radius.

        The digests are already stored, so this costs no re-read of the data and no re-hashing —
        only the grouping is redone. Cluster-based duplicates are unaffected, as hash-based ones
        are by :meth:`with_sensitivity`.

        Parameters
        ----------
        hash_radius : int
            Maximum Hamming distance, in bits, for two digests to be grouped. ``0`` groups only
            identical digests. For a 64-bit perceptual hash, ``1-5`` is very similar and ``6-10``
            possibly similar. Must not be negative.

        Returns
        -------
        DuplicatesOutput
            New output with near duplicates re-detected at the new radius.

        Raises
        ------
        ValueError
            If this output was not created from hash statistics.

        See Also
        --------
        :meth:`~dataeval.quality.DuplicatesOutput.with_sensitivity` : Re-detect cluster duplicates
        :func:`~dataeval.core.hash_groups` : The grouping this re-runs

        Examples
        --------
        >>> result = Duplicates(hash_radius=6).evaluate(images)
        >>> stricter = result.with_radius(2)
        """
        if self.calculation_results is None:
            raise ValueError("with_radius() requires hash statistics stored from evaluate() or from_stats().")
        return self._redetect(hash_radius=hash_radius)

    def _redetect(self, cluster_sensitivity: float | None = None, hash_radius: int | None = None) -> Self:
        """Re-run duplicate detection with a new cluster distance factor or hash radius.

        Recomputes hash results from stored calculation_results (deterministic
        and cheap) and cluster results from the stored ClusterResult.
        """
        cluster_sensitivity = self.cluster_sensitivity if cluster_sensitivity is None else cluster_sensitivity
        hash_radius = self.hash_radius if hash_radius is None else hash_radius
        # Recompute hash results from stored calculation_results
        available_stats: set[str] = set()
        dataset_steps: Sequence[int] | None = None
        found = _Relations.empty()

        if self.calculation_results is not None:
            stats, source_index, available_stats, dataset_steps = _prepare_hash_inputs(self.calculation_results)
            # Every relation is rebuilt, not just the ones the re-detected knobs move: the rows
            # hold indices the groups cannot be read back out of, so carrying them over is not an
            # option and dropping them would make a re-detection lose findings.
            found = _find_relations(
                stats,
                source_index,
                self.frame_map,
                self.track_map,
                self._detection_policy(hash_radius),
                dataset_steps,
            )

        # Recompute cluster results with new distance factor
        if self.cluster_result is not None and cluster_sensitivity is not None:
            cluster_dupes = _find_cluster_duplicates(
                mst=self.cluster_result["mst"],
                clusters=self.cluster_result["clusters"],
                cluster_sensitivity=cluster_sensitivity,
            )
            found = found._replace(item_near=found.item_near + [(group, "cluster") for group in cluster_dupes])

        df = _relations_frame(found, available_stats, self.merge_near_duplicates, self.frame_map, dataset_steps)

        return DuplicatesOutput(  # type: ignore[return-value]
            df,
            calculation_results=self.calculation_results,
            cluster_result=self.cluster_result,
            cluster_sensitivity=cluster_sensitivity,
            merge_near_duplicates=self.merge_near_duplicates,
            flags=self.flags,
            hash_radius=hash_radius,
            redundancy_radius=self.redundancy_radius,
            min_segment_frames=self.min_segment_frames,
            max_segment_gap=self.max_segment_gap,
            segment_offset_tolerance=self.segment_offset_tolerance,
            verify_alignment=self.verify_alignment,
            min_track_frames=self.min_track_frames,
            levels=self.levels,
            frame_map=self.frame_map,
            track_map=self.track_map,
        )


# Convenience type aliases for parameterized output
SingleDuplicatesOutput = DuplicatesOutput[SingleExactDuplicatesGroup, SingleNearDuplicatesGroup]
SingleTargetDuplicatesOutput = DuplicatesOutput[SingleExactTargetDuplicatesGroup, SingleNearTargetDuplicatesGroup]
MultiDuplicatesOutput = DuplicatesOutput[MultiExactDuplicatesGroup, MultiNearDuplicatesGroup]
MultiTargetDuplicatesOutput = DuplicatesOutput[MultiExactTargetDuplicatesGroup, MultiNearTargetDuplicatesGroup]


class Duplicates(Evaluator):
    """Finds duplicate images using hashing and/or embedding-based clustering.

    Supports multiple complementary detection methods:

    - **Hash-based exact (xxhash)**: Detects exact duplicates (identical pixel values) using xxhash.
    - **Hash-based near (phash)**: DCT-based perceptual hashing for compression/resize detection.
    - **Hash-based near (dhash)**: Gradient hash for brightness-invariant detection.
    - **Multidirectional hashing (phash_d4, dhash_d4)**: Rotation/flip-invariant variants that
      detect duplicates regardless of orientation.
    - **Cluster-based**: Uses neural network embeddings to find semantic duplicates.

    The multiple perceptual hash methods (phash, dhash) are complementary
    and can catch different types of image modifications. Using all hashes provides
    more robust near-duplicate detection without requiring a trained model.

    Three convenience flags are provided for common use cases:

    - ``ImageStats.HASH_DUPLICATES_BASIC``: Standard duplicate detection (xxhash + phash + dhash)
    - ``ImageStats.HASH_DUPLICATES_D4``: Rotation/flip-invariant detection (xxhash + phash_d4 + dhash_d4)
    - ``ImageStats.HASH``: All hash statistics (enables rotation/flip awareness)

    Parameters
    ----------
    flags : ImageStats, default ImageStats.HASH_DUPLICATES_BASIC
        Statistics to compute for hash-based duplicate detection. Set to
        ``ImageStats.NONE`` to disable hash-based detection.
    cluster_sensitivity : float, optional
        Controls how aggressively points within a cluster are considered
        duplicates, by scaling the cluster's standard deviation of MST edge
        distances. An edge is flagged as a duplicate link when its distance
        is less than ``cluster_sensitivity * cluster_std``. Lower values
        (e.g. 0.5) are stricter (fewer duplicates); higher values (e.g. 2.0)
        are more sensitive. Typical range: 0.1 – 3.0. Must be positive.
        Must be provided together with ``extractor`` to enable clustering.
        When None or when extractor is None, cluster-based detection is
        skipped entirely.
    merge_near_duplicates : bool, default True
        If True, overlapping near duplicate groups from different detection
        methods are merged into unified groups. Each group tracks which methods
        detected it, providing confidence information. If False, groups from
        each method are kept separate.
    hash_radius : int, default 0
        Maximum Hamming distance, in bits, for two perceptual hashes to be treated as
        near duplicates. ``0`` -- the default -- groups only *identical* digests, which is
        how this detector has always behaved: a re-saved PNG frequently reproduces its
        perceptual hash exactly, so a great deal of real redundancy is found at radius 0.

        Raising it finds re-encodes that moved a few bits. For the 64-bit hashes DataEval
        computes, ``1-5`` is very similar and ``6-10`` possibly similar; above roughly 12
        unrelated images begin to group. Grouping is transitive at any radius, so a large
        radius can chain a run of gradually-changing images into one group.

        .. versionadded:: 1.2
    redundancy_radius : int, default 4
        Maximum Hamming distance, in bits, for one video frame to be treated as carrying nothing
        new over the frame before it. Stretches of such frames are reported as
        ``dup_type="redundant"`` groups, and summarized per sequence by
        :meth:`~dataeval.quality.DuplicatesOutput.aggregate_by_sequence`.

        Deliberately tighter than ``hash_radius``: *carries no new information* is a stronger
        claim than *is a copy of*. Has no effect on image datasets, which have no temporal order
        for a run to span.

        .. versionadded:: 1.2
    min_segment_frames : int, default 30
        Shortest stretch two videos may share and still be reported as a ``dup_type="segment"``
        row, in query frames and measured after ``segment_offset_tolerance`` has joined what it
        can. Below roughly a second's worth, shared intros, title cards and stock footage dominate
        the answer. Must be at least 1. Has no effect on image datasets.

        .. versionadded:: 1.2
    max_segment_gap : int, default 5
        Frames a shared stretch may skip and still count as continuous, bridging a dropped frame
        or one the hash missed. Too large a value bridges a cut. Must not be negative. Has no
        effect on image datasets.

        .. versionadded:: 1.2
    segment_offset_tolerance : int, default 0
        How far two stretches may differ in offset and still be joined into one. ``0`` requires a
        single constant offset, which is what a cut, trim or insertion produces; raise it where
        the two videos were sampled at slightly different rates and the offset drifts. Must not be
        negative. Has no effect on image datasets.

        .. versionadded:: 1.2
    verify_alignment : int or None, default None
        Mean bits per frame two videos may differ by, along a *warped* alignment, and still be
        reported as a ``dup_type="aligned"`` row. None -- the default -- skips warped matching.

        Segments look for a constant offset, so a copy played back at another rate, or converted
        to another frame rate, arrives as fragments too short to report. Warping the two against
        each other absorbs that slope. It runs only on sequence pairs the segment search could not
        explain, and only up to the same ``min_segment_frames`` bar, because an unconstrained warp
        will otherwise fold a whole video onto a handful of frames and call it a match. Around
        ``8`` is a reasonable starting point for perceptual hashes. Quadratic in the two lengths,
        against a near-linear segment search -- expect it to cost. Has no effect on image datasets.

    min_track_frames : int, default 5
        Shortest stretch two tracks may share and still be reported as a ``dup_type="segment"``
        row at ``level="track"``, in detections. Two tracks whose crops match *exactly*, end to
        end, are reported however short they are -- as whole-sequence matches are. Its own knob
        rather than ``min_segment_frames`` because the two measure
        genuinely different things: thirty frames is a second of video and a reasonable bar for a
        shared clip, while a track that survives thirty frames is already a long one. Track
        relations are computed only when ``per_target=True`` asks for the detection crop hashes
        they read. Has no effect on image datasets.

        .. versionadded:: 1.2
    frame_sample : int or float or FrameSelector or None, default None
        How much of each video to look at. None measures every frame. An ``int`` is a stride in
        frames (``5`` keeps every fifth); a ``float`` is a target rate in frames per second
        (``2.0`` keeps about two a second, read from each frame's own timestamps, which is what
        makes it comparable across videos captured at different rates). A
        :class:`~dataeval.data.FrameSelector` is used as given, so
        :class:`~dataeval.data.Redundancy` and any custom rule reach this the same way.

        Thinning changes what every count is over: an unweighted read of a thinned view describes
        the frames kept rather than the source stream. Has no effect on image datasets.

        .. versionadded:: 1.2
    extractor : FeatureExtractor, optional
        Feature extractor for cluster-based duplicate detection. Must be provided
        together with cluster_sensitivity to enable clustering. When provided alone
        without cluster_sensitivity, clustering is skipped.
    batch_size : int or None, default None
        Batch size for feature extraction during cluster-based detection. If None, uses DataEval
        default. Must be set by either parameter or global default if extractor is provided.
    cluster_algorithm : {"kmeans", "hdbscan"}, default "hdbscan"
        Clustering algorithm for cluster-based detection.
    n_clusters : int, optional
        Expected number of clusters. For HDBSCAN, this is a hint that adjusts
        min_cluster_size. For KMeans, this is the exact number of clusters.
    config : Duplicates.Config or None, default None
        Optional configuration object with default parameters. Parameters
        specified directly in __init__ will override config defaults.

    Attributes
    ----------
    stats : StatsResult
        Hash statistics computed during the last evaluate() call.
    flags : ImageStats
        Statistics to compute for duplicate detection.
    extractor : FeatureExtractor | None
        Feature extractor for cluster-based detection.
    cluster_sensitivity : float | None
        Sensitivity for cluster-based near duplicate detection. Values should be positive,
        with typical range 0.1 – 3.0. When None, cluster-based detection is disabled.
    cluster_algorithm : Literal["kmeans", "hdbscan"]
        Clustering algorithm to use.
    n_clusters : int | None
        Expected number of clusters.
    merge_near_duplicates : bool
        Whether to merge overlapping near duplicate groups.
    hash_radius : int
        Hamming radius, in bits, for hash-based near duplicate detection.
    redundancy_radius : int
        Hamming radius, in bits, for temporal redundancy between consecutive video frames.
    min_segment_frames : int
        Shortest stretch two videos may share and still be reported, in query frames.
    max_segment_gap : int
        Frames a shared stretch may skip and still count as continuous.
    segment_offset_tolerance : int
        How far two stretches may differ in offset and still be joined into one.
    verify_alignment : int or None
        Mean bits per frame a warped alignment may cost and still be reported, or None if warped
        matching is off.
    min_track_frames : int
        Shortest stretch two tracks may share and still be reported, in detections.
    frame_sample : int or float or FrameSelector or None
        Which frames of a video take part: a frame stride, a target frame rate, or a selector.

    References
    ----------
    [1] Implementation and benchmarking of perceptual image hash functions.
        Zauner, C. (2010). Bachelor's thesis, Upper Austria University of Applied Sciences.
        https://www.phash.org/docs/pubs/thesis_zauner.pdf
    [2] Semantic redundancy in image classification datasets.
        Birodkar, V., Mobahi, H., & Bengio, S. (2019). arXiv preprint arXiv:1901.11409.
        https://arxiv.org/abs/1901.11409

    Examples
    --------
    Basic hash-based detection (default):

    >>> detector = Duplicates()
    >>> result = detector.evaluate(images)

    Fast exact-only detection for large datasets:

    >>> fast_detector = Duplicates(flags=ImageStats.HASH_XXHASH)
    >>> result = fast_detector.evaluate(images)

    Tolerant near-duplicate detection, catching re-encodes whose hashes moved a few bits:

    >>> tolerant = Duplicates(hash_radius=6)
    >>> result = tolerant.evaluate(images)

    Combined hash and cluster-based detection:

    >>> from dataeval.extractors import FlattenExtractor

    >>> detector = Duplicates(extractor=FlattenExtractor(), cluster_sensitivity=1.0)
    >>> result = detector.evaluate(train_ds)

    Using configuration:

    >>> config = Duplicates.Config(
    ...     extractor=FlattenExtractor(),
    ...     cluster_algorithm="kmeans",
    ...     merge_near_duplicates=False,
    ... )
    >>> detector = Duplicates(config=config)
    """

    class Config(EvaluatorConfig, ClusterConfigMixin):
        """
        Configuration for Duplicates detector.

        Attributes
        ----------
        flags : ImageStats, default ImageStats.HASH_DUPLICATES_BASIC
            Statistics to compute for hash-based duplicate detection.
        cluster_sensitivity : float or None, default None
            Distance factor for cluster-based near duplicate detection. Scales
            the cluster's standard deviation to set the duplicate cutoff.
            Must be provided together with extractor to enable clustering.
        merge_near_duplicates : bool, default True
            Whether to merge overlapping near duplicate groups.
        hash_radius : int, default 0
            Maximum Hamming distance, in bits, for two perceptual hashes to be treated as
            near duplicates. 0 groups only identical digests.
        redundancy_radius : int, default 4
            Maximum Hamming distance, in bits, for a video frame to be treated as carrying
            nothing new over the frame before it. Tracking datasets only.
        min_segment_frames : int, default 30
            Shortest stretch two videos may share and still be reported, in query frames, measured
            after ``segment_offset_tolerance`` has joined what it can. Tracking datasets only.
        max_segment_gap : int, default 5
            Frames a shared stretch may skip and still count as continuous. Tracking datasets only.
        segment_offset_tolerance : int, default 0
            How far two stretches may differ in offset and still be joined into one. ``0`` requires
            a single constant offset. Tracking datasets only.
        verify_alignment : int or None, default None
            Mean bits per frame two videos may differ by, along a warped alignment, and still be
            reported as a match. None -- the default -- skips warped matching entirely. Tracking
            datasets only.
        min_track_frames : int, default 5
            Shortest stretch two tracks may share and still be reported, in detections. Computed
            only when ``per_target=True``. Tracking datasets only.
        frame_sample : int or float or FrameSelector or None, default None
            Which frames of a video take part. An ``int`` is a stride in frames, a ``float`` a
            target frame rate in frames per second, and a
            :class:`~dataeval.data.FrameSelector` is used as given. None measures every frame.
            Tracking datasets only.
        extractor : FeatureExtractor or None, default None
            Feature extractor for cluster-based duplicate detection.
        batch_size : int or None, default None
            Batch size for feature extraction during cluster-based detection. If None, uses DataEval
            default. Must be set by either parameter or global default if extractor is provided.
        cluster_algorithm : {"kmeans", "hdbscan"}, default "hdbscan"
            Clustering algorithm for cluster-based detection.
        n_clusters : int or None, default None
            Expected number of clusters.
        """

        flags: ImageStats = DEFAULT_DUPLICATES_FLAGS
        cluster_sensitivity: float | None = DEFAULT_DUPLICATES_CLUSTER_DISTANCE_FACTOR
        merge_near_duplicates: bool = DEFAULT_DUPLICATES_MERGE_NEAR_DUPLICATES
        hash_radius: int = DEFAULT_DUPLICATES_HASH_RADIUS
        redundancy_radius: int = DEFAULT_DUPLICATES_REDUNDANCY_RADIUS
        min_segment_frames: int = DEFAULT_DUPLICATES_MIN_SEGMENT_FRAMES
        max_segment_gap: int = DEFAULT_DUPLICATES_MAX_SEGMENT_GAP
        segment_offset_tolerance: int = DEFAULT_DUPLICATES_SEGMENT_OFFSET_TOLERANCE
        verify_alignment: int | None = DEFAULT_DUPLICATES_VERIFY_ALIGNMENT
        min_track_frames: int = DEFAULT_DUPLICATES_MIN_TRACK_FRAMES
        frame_sample: FrameSample = DEFAULT_DUPLICATES_FRAME_SAMPLE

    stats: StatsResult
    flags: ImageStats
    cluster_sensitivity: float | None
    merge_near_duplicates: bool
    hash_radius: int
    redundancy_radius: int
    min_segment_frames: int
    max_segment_gap: int
    segment_offset_tolerance: int
    verify_alignment: int | None
    min_track_frames: int
    frame_sample: FrameSample
    extractor: FeatureExtractor | None
    batch_size: int | None
    cluster_algorithm: Literal["kmeans", "hdbscan"]
    n_clusters: int | None
    config: Config

    def __init__(
        self,
        flags: ImageStats | None = None,
        cluster_sensitivity: float | None = None,
        merge_near_duplicates: bool | None = None,
        hash_radius: int | None = None,
        redundancy_radius: int | None = None,
        min_segment_frames: int | None = None,
        max_segment_gap: int | None = None,
        segment_offset_tolerance: int | None = None,
        verify_alignment: int | None = None,
        min_track_frames: int | None = None,
        frame_sample: FrameSample = None,
        extractor: FeatureExtractor | None = None,
        batch_size: int | None = None,
        cluster_algorithm: Literal["kmeans", "hdbscan"] | None = None,
        n_clusters: int | None = None,
        config: Config | None = None,
    ) -> None:
        super().__init__(locals())
        # Whether the caller *chose* this radius or simply did not choose one. Only the second is
        # warned about: an explicit 0 goes on meaning 0 after the default moves, so there is
        # nothing coming for that caller to prepare for.
        self._radius_chosen: bool = hash_radius is not None or (
            config is not None and "hash_radius" in config.model_fields_set
        )

    def _warn_default_radius(self, tracking: bool) -> None:
        """Announce the coming change to the image ``hash_radius`` default.

        ``hash_radius=0`` groups only bit-identical digests, which makes ``Duplicates`` stricter
        than its own documentation: it advertises perceptual near-duplicate detection and, at the
        default, finds none. Raising the default fixes that and changes results for every existing
        caller, so it waits for a major release and is announced ahead of it.

        Tracking datasets are left to :func:`_warn_strict_radius`, which says something sharper --
        for video the strict default does not miss relations, it measures them wrongly.
        """
        if tracking or self._radius_chosen or self.hash_radius != 0:
            return
        warnings.warn(
            "The default value of hash_radius will change from 0 in a future major release, so "
            "that Duplicates finds the perceptual near-duplicates it documents rather than only "
            "bit-identical ones (5 of 64 bits is the expected value). Pass hash_radius explicitly "
            "to pin the current behaviour and silence this warning.",
            FutureWarning,
            stacklevel=3,
        )

    def _detection_policy(self, plan: _LevelPlan) -> _DetectionPolicy:
        """Bundle every knob one pass of detection consults."""
        return _DetectionPolicy(
            levels=plan.levels,
            hash_radius=self.hash_radius,
            redundancy_radius=self.redundancy_radius,
            segment=self._segment_policy(),
            track=self._track_policy(),
        )

    def _segment_policy(self) -> _SegmentPolicy:
        """Bundle every knob deciding what counts as a stretch two sequences share."""
        return _SegmentPolicy(
            radius=self.hash_radius,
            min_length=self.min_segment_frames,
            max_gap=self.max_segment_gap,
            offset_tolerance=self.segment_offset_tolerance,
            verify_alignment=self.verify_alignment,
            track_length=self.min_track_frames,
        )

    def _track_policy(self) -> _SegmentPolicy:
        """Return the same policy one level down, where a shared stretch is measured in detections."""
        policy = self._segment_policy()
        return policy._replace(min_length=policy.track_length, verify_alignment=None)

    @overload
    def from_stats(
        self,
        stats: StatsResult,
        *,
        per_image: bool = True,
        per_target: Literal[False] = ...,
    ) -> SingleDuplicatesOutput: ...

    @overload
    def from_stats(
        self,
        stats: StatsResult,
        *,
        per_image: bool = True,
        per_target: Literal[True],
    ) -> SingleTargetDuplicatesOutput: ...

    @overload
    def from_stats(
        self,
        stats: Sequence[StatsResult],
        *,
        per_image: bool = True,
        per_target: Literal[False] = ...,
    ) -> MultiDuplicatesOutput: ...

    @overload
    def from_stats(
        self,
        stats: Sequence[StatsResult],
        *,
        per_image: bool = True,
        per_target: Literal[True],
    ) -> MultiTargetDuplicatesOutput: ...

    @set_metadata(state=["flags", "merge_near_duplicates", "hash_radius"])
    def from_stats(
        self,
        stats: StatsResult | Sequence[StatsResult],
        *,
        per_image: bool = True,
        per_target: bool = False,
    ) -> SingleDuplicatesOutput | SingleTargetDuplicatesOutput | MultiDuplicatesOutput | MultiTargetDuplicatesOutput:
        """
        Find duplicates from pre-computed hash statistics.

        Use this method when hash statistics have already been computed
        via :func:`~dataeval.core.compute_stats` to avoid redundant computation.

        Parameters
        ----------
        stats : StatsResult | Sequence[StatsResult]
            Pre-computed statistics containing hash values. Must include
            at least one of: xxhash, phash, dhash, rhash. Can be a single
            result or a sequence of results.
        per_image : bool, default True
            Whether to include item-level (image) duplicate groups.
        per_target : bool, default False
            Whether to include target-level duplicate groups.
            When True, accessor properties return :class:`SourceIndex` indices;
            when False, they return plain ``int`` item indices.

        Returns
        -------
        DuplicatesOutput
            Duplicate detection results as a DataFrame of duplicate groups.
            For cross-dataset detection, includes a dataset_indices column.

        See Also
        --------
        :meth:`~dataeval.quality.Duplicates.evaluate` : Compute hashes and find duplicates in one call
        :meth:`~dataeval.quality.Duplicates.from_clusters` : Find duplicates using cluster-based detection
        """
        # Normalize to a single or list of StatsResults
        calc_results: StatsResult | list[StatsResult]
        calc_results = stats if isinstance(stats, dict) else list(stats)

        hash_stats, source_index, available_stats, dataset_steps = _prepare_hash_inputs(calc_results)
        (item_exact, item_near), (target_exact, target_near) = _detect_hash_duplicates(
            hash_stats, source_index, self.hash_radius
        )

        df = _build_duplicates_dataframe(
            (item_exact or None) if per_image else None,
            item_near if per_image else [],
            _selected_targets(target_exact, per_target) or None,
            _selected_targets(target_near, per_target, near=True),
            available_stats,
            self.merge_near_duplicates,
            dataset_steps=dataset_steps,
        )
        return DuplicatesOutput(
            df,
            calculation_results=calc_results,
            merge_near_duplicates=self.merge_near_duplicates,
            flags=self.flags,
            hash_radius=self.hash_radius,
            redundancy_radius=self.redundancy_radius,
            min_segment_frames=self.min_segment_frames,
            max_segment_gap=self.max_segment_gap,
            segment_offset_tolerance=self.segment_offset_tolerance,
            verify_alignment=self.verify_alignment,
            min_track_frames=self.min_track_frames,
            frame_map=None,
        )

    @set_metadata(state=["cluster_sensitivity", "cluster_algorithm", "n_clusters"])
    def from_clusters(
        self,
        cluster_result: ClusterResult,
    ) -> SingleDuplicatesOutput:
        """
        Find duplicates using cluster-based detection from minimum spanning tree.

        Analyzes the minimum spanning tree and cluster assignments to identify
        near duplicates based on distance relationships within clusters.

        Parameters
        ----------
        cluster_result : ClusterResult
            Clustering results from the cluster() function.

        Returns
        -------
        DuplicatesOutput
            Duplicate detection results with item-level duplicate groups.
            Cluster-based detection operates on items only (no target separation).

        See Also
        --------
        :func:`~dataeval.core.cluster` : Function to compute clusters from embeddings
        :meth:`~dataeval.quality.Duplicates.from_stats` : Find duplicates from pre-computed hash statistics
        :meth:`~dataeval.quality.Duplicates.evaluate` : Find duplicates by computing hashes from images

        Notes
        -----
        This method identifies duplicates in embedding space. All cluster-based
        duplicates are returned as **near duplicates** because embeddings are
        approximate representations - identical embeddings don't guarantee
        pixel-identical images.
        """
        threshold = self.cluster_sensitivity if self.cluster_sensitivity is not None else 1.0
        cluster_dupes = _find_cluster_duplicates(
            mst=cluster_result["mst"],
            clusters=cluster_result["clusters"],
            cluster_sensitivity=threshold,
        )

        cluster_method_groups: MethodGroups = [(group, "cluster") for group in cluster_dupes]

        df = _build_duplicates_dataframe(
            item_exact=None,
            item_near_method_groups=cluster_method_groups,
            target_exact=None,
            target_near_method_groups=[],
            available_stats=set(),
            merge=self.merge_near_duplicates,
        )
        return DuplicatesOutput(
            df,
            cluster_result=cluster_result,
            cluster_sensitivity=threshold,
            merge_near_duplicates=self.merge_near_duplicates,
            flags=self.flags,
            hash_radius=self.hash_radius,
            redundancy_radius=self.redundancy_radius,
            min_segment_frames=self.min_segment_frames,
            max_segment_gap=self.max_segment_gap,
            segment_offset_tolerance=self.segment_offset_tolerance,
            verify_alignment=self.verify_alignment,
            min_track_frames=self.min_track_frames,
            frame_map=None,
        )

    # Video is in here too: a tracking datum's first element is a VideoStream rather than an
    # array, so a MOT dataset satisfies neither of the image spellings and would be refused by a
    # type checker despite `evaluate` dispatching on it at runtime.
    _DatasetInput = Dataset[ArrayLike] | Dataset[tuple[ArrayLike, Any, Any]] | MultiobjectTrackingDataset

    @overload
    def evaluate(  # pyright: ignore[reportOverlappingOverload]
        self,
        data: _DatasetInput,
        *,
        levels: str | Sequence[str] | None = ...,
        per_image: bool | None = ...,
        per_target: Literal[False] | None = ...,
    ) -> SingleDuplicatesOutput: ...

    @overload
    def evaluate(  # type: ignore[reportOverlappingOverload]
        self,
        data: _DatasetInput,
        *,
        levels: str | Sequence[str] | None = ...,
        per_image: bool | None = ...,
        per_target: Literal[True],
    ) -> SingleTargetDuplicatesOutput: ...

    @overload
    def evaluate(
        self,
        data: _DatasetInput,
        *other: _DatasetInput,
        levels: str | Sequence[str] | None = ...,
        per_image: bool | None = ...,
        per_target: Literal[False] | None = ...,
    ) -> MultiDuplicatesOutput: ...

    @overload
    def evaluate(
        self,
        data: _DatasetInput,
        *other: _DatasetInput,
        levels: str | Sequence[str] | None = ...,
        per_image: bool | None = ...,
        per_target: Literal[True],
    ) -> MultiTargetDuplicatesOutput: ...

    @set_metadata(
        state=[
            "flags",
            "cluster_sensitivity",
            "cluster_algorithm",
            "n_clusters",
            "hash_radius",
            "redundancy_radius",
            "min_segment_frames",
            "max_segment_gap",
            "segment_offset_tolerance",
            "verify_alignment",
            "min_track_frames",
            "frame_sample",
        ]
    )
    def evaluate(
        self,
        data: _DatasetInput,
        *other: _DatasetInput,
        levels: str | Sequence[str] | None = None,
        per_image: bool | None = None,
        per_target: bool | None = None,
    ) -> SingleDuplicatesOutput | SingleTargetDuplicatesOutput | MultiDuplicatesOutput | MultiTargetDuplicatesOutput:
        """Find duplicates by computing hashes and/or analyzing embeddings.

        Performs duplicate detection using hash statistics and/or cluster-based
        analysis depending on configuration. Supports single or multiple datasets.

        Parameters
        ----------
        data : Dataset
            The images to scan for duplicates. Accepts any object satisfying the
            :class:`~dataeval.protocols.Dataset` interface (indexed access via
            ``__getitem__`` and ``__len__``). Each item may be:

            - a bare image array, i.e. ``Dataset[ArrayLike]``; or
            - a full MAITE ``(image, target, metadata)`` tuple, i.e.
              ``Dataset[tuple[ArrayLike, Any, Any]]``.

            A plain array or list of image arrays also structurally satisfies
            :class:`~dataeval.protocols.Dataset` and is accepted directly.
        *other : Dataset
            Zero or more additional datasets for cross-dataset duplicate
            detection, each accepting the same forms as ``data``. When provided,
            duplicates are searched across all datasets and the output includes a
            ``dataset_indices`` column identifying each item's originating dataset.
        levels : str or Sequence[str] or None, default None
            Which levels to report relations at, and so which to measure for. An image dataset
            accepts ``"item"`` and ``"target"``; a tracking dataset accepts ``"sequence"``,
            ``"unit"`` (a frame), ``"track"`` and ``"instance"`` (a detection).

            This and ``per_image``/``per_target`` are two spellings of one thing -- one names the
            answers wanted, the other the measurements taken -- so passing both raises rather
            than reconciling them silently. ``levels`` is the more direct spelling for video,
            where ``per_target=True`` is otherwise the only way to ask for track relations.

            A level not asked for is not searched for: ``levels="sequence"`` over a video corpus
            reports which videos duplicate which and pays for no frame-level grouping at all.

            None keeps the behaviour of whichever of ``per_image``/``per_target`` was given, and
            of their defaults when neither was.

            .. versionadded:: 1.2
        per_image : bool, default True
            Whether to compute hashes for full items (images/videos).
        per_target : bool, default False
            Whether to compute hashes for individual targets/detections.
            When True, accessor properties return :class:`SourceIndex` indices;
            when False, they return plain ``int`` item indices. For a tracking dataset this is
            also what asks for track-level relations; ``levels`` says so directly.

        Returns
        -------
        SingleDuplicatesOutput or MultiDuplicatesOutput
            Duplicate detection results as a DataFrame of duplicate groups.
            For multi-dataset input, includes a ``dataset_indices`` column.

        Raises
        ------
        ValueError
            If flags is NONE and no extractor is provided.

        Examples
        --------
        Hash-based duplicates with merged near duplicates (default):

        >>> detector = Duplicates()
        >>> detector.evaluate(images)
        shape: (3, 5)
        ┌──────────┬───────┬──────────┬───────────────┬────────────┐
        │ group_id ┆ level ┆ dup_type ┆ item_indices  ┆ methods    │
        │ ---      ┆ ---   ┆ ---      ┆ ---           ┆ ---        │
        │ i64      ┆ str   ┆ str      ┆ list[i64]     ┆ list[str]  │
        ╞══════════╪═══════╪══════════╪═══════════════╪════════════╡
        │ 0        ┆ item  ┆ exact    ┆ [3, 20]       ┆ ["xxhash"] │
        │ 1        ┆ item  ┆ exact    ┆ [7, 11, … 25] ┆ ["xxhash"] │
        │ 2        ┆ item  ┆ exact    ┆ [16, 37]      ┆ ["xxhash"] │
        └──────────┴───────┴──────────┴───────────────┴────────────┘

        Cross-dataset detection:

        >>> detector = Duplicates()
        >>> detector.evaluate(train_ds, test_ds)
        shape: (3, 6)
        ┌──────────┬───────┬──────────┬───────────────┬─────────────────┬────────────┐
        │ group_id ┆ level ┆ dup_type ┆ item_indices  ┆ dataset_indices ┆ methods    │
        │ ---      ┆ ---   ┆ ---      ┆ ---           ┆ ---             ┆ ---        │
        │ i64      ┆ str   ┆ str      ┆ list[i64]     ┆ list[i64]       ┆ list[str]  │
        ╞══════════╪═══════╪══════════╪═══════════════╪═════════════════╪════════════╡
        │ 0        ┆ item  ┆ exact    ┆ [3, 20]       ┆ [0, 0]          ┆ ["xxhash"] │
        │ 1        ┆ item  ┆ exact    ┆ [7, 11, … 25] ┆ [0, 0, … 0]     ┆ ["xxhash"] │
        │ 2        ┆ item  ┆ exact    ┆ [16, 37]      ┆ [0, 0]          ┆ ["xxhash"] │
        └──────────┴───────┴──────────┴───────────────┴─────────────────┴────────────┘
        """
        tracking = _yields_frames(data) or any(_yields_frames(dataset) for dataset in other)
        self._warn_default_radius(tracking)
        plan = _resolve_levels(levels, per_image, per_target, tracking)

        if other:
            return self._evaluate_multi([data, *other], plan)

        return self._evaluate_single(data, plan)

    def _evaluate_single(self, data: _DatasetInput, plan: _LevelPlan) -> SingleDuplicatesOutput:
        """Single-dataset evaluate implementation."""
        # Validate parameters - need either hash-based or cluster-based detection
        # Cluster-based detection requires both extractor AND cluster_sensitivity
        has_hash_detection = bool(self.flags & ImageStats.HASH)
        has_cluster_detection = self.extractor is not None and self.cluster_sensitivity is not None
        if not has_hash_detection and not has_cluster_detection:
            raise ValueError(
                "Either flags must contain hash stats, or both extractor and "
                "cluster_sensitivity must be provided for cluster-based detection.",
            )

        # Initialize results
        stored_cluster_result: ClusterResult | None = None
        found = _Relations.empty()

        # Bound to a new name rather than over `data`: for a tracking dataset this is the frame
        # view, which is image-shaped, and rebinding the parameter would keep its declared type.
        measured, frame_map, track_map = _as_frames(data, self.frame_sample, type(self).__name__, self.hash_radius)

        # Hash-based duplicate detection
        if self.flags & ImageStats.HASH:
            self.stats = checked_compute_stats(
                [measured],
                stats=self.flags & ImageStats.HASH,
                caller=type(self).__name__,
                per_image=plan.per_image,
                per_target=plan.per_target,
                normalize_pixel_values=False,
            )[0]
            found = _find_relations(
                self.stats["stats"],
                self.stats["source_index"],
                frame_map,
                track_map,
                self._detection_policy(plan),
            )

        # Cluster-based duplicate detection (requires both extractor and cluster_sensitivity)
        if self.extractor is not None and self.cluster_sensitivity is not None:
            embeddings = Embeddings(measured, self.extractor, batch_size=self.batch_size)

            stored_cluster_result = cluster(
                embeddings,
                algorithm=self.cluster_algorithm,
                n_clusters=self.n_clusters,
            )

            factor = self.cluster_sensitivity if self.cluster_sensitivity is not None else 1.0
            cluster_dupes = _find_cluster_duplicates(
                mst=stored_cluster_result["mst"],
                clusters=stored_cluster_result["clusters"],
                cluster_sensitivity=factor,
            )
            found = found._replace(item_near=found.item_near + [(group, "cluster") for group in cluster_dupes])

        available_stats = set(self.stats["stats"].keys()) if self.flags & ImageStats.HASH else set()
        df = _relations_frame(found, available_stats, self.merge_near_duplicates, frame_map)
        return DuplicatesOutput(  # type: ignore[return-value]
            df,
            calculation_results=self.stats if has_hash_detection else None,
            cluster_result=stored_cluster_result,
            cluster_sensitivity=self.cluster_sensitivity,
            merge_near_duplicates=self.merge_near_duplicates,
            flags=self.flags,
            hash_radius=self.hash_radius,
            redundancy_radius=self.redundancy_radius,
            min_segment_frames=self.min_segment_frames,
            max_segment_gap=self.max_segment_gap,
            segment_offset_tolerance=self.segment_offset_tolerance,
            verify_alignment=self.verify_alignment,
            min_track_frames=self.min_track_frames,
            levels=plan.levels,
            frame_map=frame_map,
            track_map=track_map,
        )

    def _evaluate_multi(self, datasets: Sequence[_DatasetInput], plan: _LevelPlan) -> MultiDuplicatesOutput:
        """Multi-dataset evaluate: compute stats per dataset, then combine."""
        has_hash_detection = bool(self.flags & ImageStats.HASH)
        has_cluster_detection = self.extractor is not None and self.cluster_sensitivity is not None
        if not has_hash_detection and not has_cluster_detection:
            raise ValueError(
                "Either flags must contain hash stats, or both extractor and "
                "cluster_sensitivity must be provided for cluster-based detection.",
            )

        measured, frame_map, track_map = _as_frames_multi(
            datasets, self.frame_sample, type(self).__name__, self.hash_radius
        )

        # Hash-based: compute stats per dataset, delegate to from_stats
        calc_results: list[StatsResult] = []
        if has_hash_detection:
            calc_results = checked_compute_stats(
                measured,
                stats=self.flags & ImageStats.HASH,
                caller=type(self).__name__,
                per_image=plan.per_image,
                per_target=plan.per_target,
                normalize_pixel_values=False,
            )
            self.stats = calc_results[-1]

        hash_stats, source_index, available_stats, dataset_steps = (
            _prepare_hash_inputs(calc_results) if calc_results else ({}, [], set(), None)
        )
        # Item indices are combined across the datasets; the frame map has to be addressable the
        # same way before anything reads a frame's sequence out of it.
        frame_map = _aligned_frame_map(source_index, frame_map)
        track_map = _aligned_track_map(source_index, track_map)

        stored_cluster_result: ClusterResult | None = None
        found = _Relations.empty()
        if calc_results:
            found = _find_relations(
                hash_stats, source_index, frame_map, track_map, self._detection_policy(plan), dataset_steps
            )

        # Cluster-based: combine all images, extract, cluster together
        if has_cluster_detection:
            all_images = [img for ds in measured for img in iter_images(ds)]
            embeddings = self.extractor(all_images)  # type: ignore[union-attr]
            embeddings_array = flatten_samples(to_numpy(embeddings))

            stored_cluster_result = cluster(
                embeddings_array,
                algorithm=self.cluster_algorithm,
                n_clusters=self.n_clusters,
            )

            factor = self.cluster_sensitivity if self.cluster_sensitivity is not None else 1.0
            cluster_dupes = _find_cluster_duplicates(
                mst=stored_cluster_result["mst"],
                clusters=stored_cluster_result["clusters"],
                cluster_sensitivity=factor,
            )
            found = found._replace(item_near=found.item_near + [(group, "cluster") for group in cluster_dupes])

        df = _relations_frame(found, available_stats, self.merge_near_duplicates, frame_map, dataset_steps)
        return DuplicatesOutput(  # type: ignore[return-value]
            df,
            calculation_results=calc_results if calc_results else None,
            cluster_result=stored_cluster_result,
            cluster_sensitivity=self.cluster_sensitivity,
            merge_near_duplicates=self.merge_near_duplicates,
            flags=self.flags,
            hash_radius=self.hash_radius,
            redundancy_radius=self.redundancy_radius,
            min_segment_frames=self.min_segment_frames,
            max_segment_gap=self.max_segment_gap,
            segment_offset_tolerance=self.segment_offset_tolerance,
            verify_alignment=self.verify_alignment,
            min_track_frames=self.min_track_frames,
            levels=plan.levels,
            frame_map=frame_map,
            track_map=track_map,
        )
