"""Duplicate detection for images using hashing and clustering."""

__all__ = []

from collections.abc import Mapping, Sequence
from typing import Any, Generic, Literal, TypeVar, overload

import numpy as np
import polars as pl
from numpy.typing import NDArray
from typing_extensions import Self

from dataeval import Embeddings
from dataeval.core import ClusterResult, StatsResult, cluster, combine_stats_results, compute_stats
from dataeval.flags import ImageStats
from dataeval.protocols import ArrayLike, Dataset, FeatureExtractor
from dataeval.quality._shared import drop_null_index_columns, get_dataset_step_from_idx
from dataeval.types import (
    ClusterConfigMixin,
    DataFrameOutput,
    Evaluator,
    EvaluatorConfig,
    SourceIndex,
    StatsMap,
    set_metadata,
)
from dataeval.utils.arrays import flatten_samples, to_numpy

DEFAULT_DUPLICATES_FLAGS = ImageStats.HASH_DUPLICATES_BASIC
DEFAULT_DUPLICATES_CLUSTER_THRESHOLD: float | None = None
DEFAULT_DUPLICATES_MERGE_NEAR_DUPLICATES = True


_BASIC_HASH_METHODS = frozenset({"phash", "dhash"})
_D4_HASH_METHODS = frozenset({"phash_d4", "dhash_d4"})

# Type alias for raw detection output: (exact_groups, near_method_groups)
# method_groups are (indices, method_name) tuples before merging
MethodGroups = list[tuple[Sequence[Any], str]]

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
    "target_indices": pl.List(pl.Int64),
    "methods": pl.List(pl.Utf8),
    "orientation": pl.Utf8,
}


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


def _merge_near_groups(
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
                tuple(sorted(group)),
                frozenset({method}),
                None if is_unknown else _get_orientation(frozenset({method})),
            )
            for group, method in method_groups
        ]
        return sorted(groups, key=lambda g: g[0])

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
            tuple(sorted(indices)),
            frozenset(methods),
            None if is_unknown else _get_orientation(frozenset(methods)),
        )
        for indices, methods in merged
        if len(indices) > 1
    ]
    return sorted(result, key=lambda g: g[0])


def _find_cluster_duplicates(
    mst: NDArray[np.float32],
    clusters: NDArray[np.intp],
    cluster_threshold: float = 1.0,
) -> tuple[list[list[int]], list[list[int]]]:
    """Find duplicate and near duplicate data based on cluster average distance.

    Parameters
    ----------
    mst : NDArray[np.float32]
        Minimum spanning tree from cluster() output.
    clusters : NDArray[np.intp]
        Cluster labels from cluster() output.
    cluster_threshold : float, default 1.0
        Multiplier on cluster standard deviation for near duplicate detection.

    Returns
    -------
    tuple[list[list[int]], list[list[int]]]
        Exact duplicates and near duplicates as lists of related indices.
    """
    from dataeval.core._fast_hdbscan._mst import compare_links_to_cluster_std

    exact_indices, near_indices = compare_links_to_cluster_std(mst, clusters, cluster_threshold)
    exact_dupes = _sorted_union_find(exact_indices)
    near_dupes = _sorted_union_find(near_indices)

    return [[int(ii) for ii in il] for il in exact_dupes], [[int(ii) for ii in il] for il in near_dupes]


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


def _extract_members(row: Mapping[str, Any], has_targets: bool) -> list[int] | list[SourceIndex]:
    """Extract member indices from a single DataFrame row."""
    if has_targets:
        return [
            SourceIndex(item=item, target=target)
            for item, target in zip(row["item_indices"], row["target_indices"], strict=True)
        ]
    return row["item_indices"]


def _group_by_dataset(row: Mapping[str, Any], has_targets: bool) -> dict[int, list[Any]]:
    """Group a row's members by dataset index."""
    by_ds: dict[int, list[Any]] = {}
    if has_targets:
        for item, target, ds in zip(row["item_indices"], row["target_indices"], row["dataset_index"], strict=True):
            by_ds.setdefault(ds, []).append(SourceIndex(item=item, target=target))
    else:
        for item, ds in zip(row["item_indices"], row["dataset_index"], strict=True):
            by_ds.setdefault(ds, []).append(item)
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
) -> tuple[list[int], list[int | None], list[int] | None]:
    """Extract item_indices, target_indices, and optional dataset_ids from raw indices."""
    item_indices: list[int] = []
    target_indices: list[int | None] = []
    dataset_ids: list[int] = [] if dataset_steps is not None else None  # type: ignore[assignment]

    for idx in indices:
        if isinstance(idx, SourceIndex):
            item_idx = idx.item
            target_idx = idx.target
        else:
            item_idx = idx
            target_idx = None

        if dataset_steps is not None:
            ds_idx, item_idx = get_dataset_step_from_idx(item_idx, dataset_steps)
            dataset_ids.append(ds_idx)

        item_indices.append(item_idx)
        target_indices.append(target_idx)

    return item_indices, target_indices, dataset_ids


def _make_row(
    indices: Sequence[Any],
    group_id: int,
    level: str,
    dup_type: str,
    methods: Sequence[str],
    orientation: str | None,
    dataset_steps: Sequence[int] | None,
) -> dict[str, Any]:
    """Build a single DataFrame row dict from a duplicate group."""
    item_ids, target_ids, ds_ids = _indices_to_row_fields(indices, dataset_steps)
    row: dict[str, Any] = {
        "group_id": group_id,
        "level": level,
        "dup_type": dup_type,
        "item_indices": item_ids,
        "target_indices": target_ids,
        "methods": methods,
        "orientation": orientation,
    }
    if ds_ids is not None:
        row["dataset_index"] = ds_ids
    return row


def _build_duplicates_dataframe(
    item_exact: Sequence[Sequence[int]] | None,
    item_near_method_groups: Sequence[tuple[Sequence[Any], str]],
    target_exact: Sequence[Sequence[SourceIndex]] | None,
    target_near_method_groups: Sequence[tuple[Sequence[Any], str]],
    available_stats: set[str],
    merge: bool,
    dataset_steps: Sequence[int] | None = None,
) -> pl.DataFrame:
    """Build a unified DataFrame of duplicate groups from raw detection data.

    Handles near-group merging internally via ``_merge_near_groups``.
    Each row represents one duplicate group with columns defined by ``_EMPTY_DUPS_SCHEMA``.
    """
    rows: list[dict[str, Any]] = []
    group_id = 0

    for level, exact_groups, near_method_groups in (
        ("item", item_exact, item_near_method_groups),
        ("target", target_exact, target_near_method_groups),
    ):
        if exact_groups:
            for group in sorted(sorted(g) for g in exact_groups):
                rows.append(_make_row(group, group_id, level, "exact", ["xxhash"], None, dataset_steps))
                group_id += 1

        if near_method_groups:
            for indices, methods, orientation in _merge_near_groups(near_method_groups, available_stats, merge):
                rows.append(_make_row(indices, group_id, level, "near", sorted(methods), orientation, dataset_steps))
                group_id += 1

    # Orientation is only meaningful when both basic and D4 hashes were computed
    has_basic_stats = bool(available_stats & _BASIC_HASH_METHODS)
    has_d4_stats = bool(available_stats & _D4_HASH_METHODS)
    include_orientation = has_basic_stats and has_d4_stats

    if not rows:
        schema = {k: v for k, v in _EMPTY_DUPS_SCHEMA.items() if k != "orientation" or include_orientation}
        return pl.DataFrame(schema=schema)

    df = pl.DataFrame(rows)

    # Omit orientation when it cannot be determined
    if not include_orientation and "orientation" in df.columns:
        df = df.drop("orientation")

    return drop_null_index_columns(df, ["target_indices"])


def _find_hash_groups(
    stats: StatsMap,
    hash_key: str,
    source_index: Sequence[SourceIndex],
    indices: Sequence[int],
    exact_groups: Sequence[Sequence[Any]],
    use_source_index: bool = False,
) -> list[list[Any]]:
    """Find near duplicates for a specific hash type.

    When use_source_index is True, stores full SourceIndex objects (for targets).
    Otherwise stores item integers (for items).
    """
    near_dict: dict[str, list[Any]] = {}
    for i in indices:
        value = stats[hash_key][i]
        if value:  # Skip empty hashes
            near_dict.setdefault(value, []).append(source_index[i] if use_source_index else source_index[i].item)

    return [sorted(v) for v in near_dict.values() if len(v) > 1 and not any(set(v).issubset(x) for x in exact_groups)]


def _detect_hash_duplicates(
    stats: StatsMap,
    source_index: Sequence[SourceIndex],
) -> tuple[
    tuple[SingleExactDuplicatesGroup, MethodGroups],
    tuple[SingleExactTargetDuplicatesGroup, MethodGroups],
]:
    """Extract duplicate groups from hash statistics, separating items and targets.

    Returns
    -------
    tuple of ((item_exact, item_method_groups), (target_exact, target_method_groups))
        Raw detection results for item-level and target-level duplicates.
    """
    item_indices: list[int] = []
    target_indices: list[int] = []
    for i, src_idx in enumerate(source_index):
        (target_indices if src_idx.target is not None else item_indices).append(i)

    hash_methods = ["phash", "dhash", "phash_d4", "dhash_d4"]

    # Item-level detection
    item_exact: list[list[int]] = []
    if "xxhash" in stats:
        d: dict[str, list[int]] = {}
        for i in item_indices:
            d.setdefault(stats["xxhash"][i], []).append(source_index[i].item)
        item_exact = [sorted(v) for v in d.values() if len(v) > 1]

    item_near: MethodGroups = []
    for method in hash_methods:
        if method in stats:
            item_near.extend(
                (g, method) for g in _find_hash_groups(stats, method, source_index, item_indices, item_exact)
            )

    # Target-level detection
    target_exact: list[list[SourceIndex]] = []
    if "xxhash" in stats:
        td: dict[str, list[SourceIndex]] = {}
        for i in target_indices:
            td.setdefault(stats["xxhash"][i], []).append(source_index[i])
        target_exact = [sorted(v) for v in td.values() if len(v) > 1]

    target_near: MethodGroups = []
    for method in hash_methods:
        if method in stats:
            target_near.extend(
                (g, method)
                for g in _find_hash_groups(
                    stats, method, source_index, target_indices, target_exact, use_source_index=True
                )
            )

    return (sorted(item_exact) or [], item_near), (sorted(target_exact) or [], target_near)


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


class DuplicatesOutput(DataFrameOutput, Generic[TExactDuplicatesGroup, TNearDuplicatesGroup]):
    """
    Output class for :class:`.Duplicates` detector.

    Wraps a Polars DataFrame of duplicate groups with aggregation helpers
    and threshold-based redetection for cluster duplicates.

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
    - dataset_index: list[int] - Dataset indices for cross-dataset results (only
      present for multi-dataset output, positionally aligned with item_indices)

    Attributes
    ----------
    calculation_results : StatsResult or Sequence[StatsResult] or None
        The original hash statistics. Used internally for redetection via
        :meth:`with_threshold`.
    cluster_result : ClusterResult or None
        The clustering result (MST + cluster assignments). Used internally
        for redetection via :meth:`with_threshold`.
    cluster_threshold : float or None
        Threshold used for cluster-based near duplicate detection.
    merge_near_duplicates : bool
        Whether overlapping near duplicate groups were merged.
    flags : ImageStats
        The hash statistics flags used for detection.
    """

    def __init__(
        self,
        data: pl.DataFrame,
        *,
        calculation_results: StatsResult | Sequence[StatsResult] | None = None,
        cluster_result: ClusterResult | None = None,
        cluster_threshold: float | None = None,
        merge_near_duplicates: bool = True,
        flags: ImageStats = ImageStats.NONE,
    ) -> None:
        super().__init__(data)
        self.calculation_results = calculation_results
        self.cluster_result = cluster_result
        self.cluster_threshold = cluster_threshold
        self.merge_near_duplicates = merge_near_duplicates
        self.flags = flags

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
        is_cross = "dataset_index" in self.data().columns
        has_targets = "target_indices" in self.data().columns
        is_near = dup_type == "near"

        filtered = self.data().filter(pl.col("dup_type") == dup_type)

        if is_cross:
            return _get_groups_cross(filtered, has_targets, is_near)
        return _get_groups_single(filtered, has_targets, is_near)

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

            - item_index: int - The image index
            - group_count: int - Number of duplicate groups this image appears in
            - dup_types: list[str] - Unique duplicate types for this image
            - methods: list[str] - All unique methods that detected this image
        """
        if "dataset_index" in self.data().columns:
            raise ValueError("aggregate_by_image only works with output from a single dataset.")

        schema: Any = {
            "item_index": pl.Int64,
            "group_count": pl.UInt32,
            "dup_types": pl.List(pl.Utf8),
            "methods": pl.List(pl.Utf8),
        }

        if self.data().shape[0] == 0:
            return pl.DataFrame(schema=schema)

        exploded = self.data().explode("item_indices").rename({"item_indices": "item_index"})

        return (
            exploded.group_by("item_index")
            .agg(
                pl.len().cast(pl.UInt32).alias("group_count"),
                pl.col("dup_type").unique().sort().alias("dup_types"),
                pl.col("methods").explode().unique().sort().alias("methods"),
            )
            .sort(["group_count", "item_index"], descending=[True, False])
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
            with_count.explode("methods")
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

    def with_threshold(self, cluster_threshold: float) -> Self:
        """Re-detect cluster-based duplicates with a different threshold.

        Hash-based duplicates are deterministic and are not affected.
        Only cluster-based near duplicates are recomputed using the stored
        clustering result (MST + cluster assignments).

        Parameters
        ----------
        cluster_threshold : float
            New threshold for cluster-based near duplicate detection.
            Lower values are stricter (fewer near duplicates).

        Returns
        -------
        DuplicatesOutput
            New output with re-detected duplicates using the new threshold.

        Raises
        ------
        ValueError
            If this output was not created from an evaluation with cluster results.
        """
        if self.cluster_result is None:
            raise ValueError("with_threshold() requires cluster results stored from evaluate() or from_clusters().")
        return self._redetect(cluster_threshold=cluster_threshold)

    def _redetect(self, cluster_threshold: float) -> Self:
        """Re-run duplicate detection with a new cluster threshold.

        Recomputes hash results from stored calculation_results (deterministic
        and cheap) and cluster results from the stored ClusterResult.
        """
        # Recompute hash results from stored calculation_results
        item_exact: SingleExactDuplicatesGroup | None = None
        item_near: MethodGroups = []
        target_exact: SingleExactTargetDuplicatesGroup | None = None
        target_near: MethodGroups = []
        available_stats: set[str] = set()
        dataset_steps: Sequence[int] | None = None

        if self.calculation_results is not None:
            stats, source_index, available_stats, dataset_steps = _prepare_hash_inputs(self.calculation_results)
            (i_e, i_n), (t_e, t_n) = _detect_hash_duplicates(stats, source_index)
            item_exact, item_near = i_e or None, i_n
            target_exact, target_near = t_e or None, t_n

        # Recompute cluster results with new threshold
        if self.cluster_result is not None:
            cluster_exact, cluster_near = _find_cluster_duplicates(
                mst=self.cluster_result["mst"],
                clusters=self.cluster_result["clusters"],
                cluster_threshold=cluster_threshold,
            )
            item_near = item_near + [(group, "cluster") for group in cluster_exact + cluster_near]

        df = _build_duplicates_dataframe(
            item_exact,
            item_near,
            target_exact,
            target_near,
            available_stats,
            self.merge_near_duplicates,
            dataset_steps=dataset_steps,
        )

        return DuplicatesOutput(  # type: ignore[return-value]
            df,
            calculation_results=self.calculation_results,
            cluster_result=self.cluster_result,
            cluster_threshold=cluster_threshold,
            merge_near_duplicates=self.merge_near_duplicates,
            flags=self.flags,
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
    extractor : FeatureExtractor, optional
        Feature extractor for cluster-based duplicate detection. Must be provided
        together with cluster_threshold to enable clustering. When provided alone
        without cluster_threshold, clustering is skipped.
    cluster_threshold : float, optional
        Threshold for cluster-based near duplicate detection. Must be provided
        together with extractor to enable clustering. When None or when
        extractor is None, cluster-based detection is skipped entirely.
        Lower values are stricter.
    cluster_algorithm : {"kmeans", "hdbscan"}, default "hdbscan"
        Clustering algorithm for cluster-based detection.
    n_clusters : int, optional
        Expected number of clusters. For HDBSCAN, this is a hint that adjusts
        min_cluster_size. For KMeans, this is the exact number of clusters.
    merge_near_duplicates : bool, default True
        If True, overlapping near duplicate groups from different detection
        methods are merged into unified groups. Each group tracks which methods
        detected it, providing confidence information. If False, groups from
        each method are kept separate.
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
    cluster_threshold : float | None
        Threshold for cluster-based near duplicate detection.
    cluster_algorithm : Literal["kmeans", "hdbscan"]
        Clustering algorithm to use.
    n_clusters : int | None
        Expected number of clusters.
    merge_near_duplicates : bool
        Whether to merge overlapping near duplicate groups.

    Examples
    --------
    Basic hash-based detection (default):

    >>> detector = Duplicates()
    >>> result = detector.evaluate(images)

    Fast exact-only detection for large datasets:

    >>> fast_detector = Duplicates(flags=ImageStats.HASH_XXHASH)
    >>> result = fast_detector.evaluate(images)

    Combined hash and cluster-based detection:

    >>> from dataeval.extractors import FlattenExtractor

    >>> detector = Duplicates(extractor=FlattenExtractor(), cluster_threshold=1.0)
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
        flags : ImageStats, default ImageStats.HASH
            Statistics to compute for hash-based duplicate detection.
        cluster_threshold : float or None, default None
            Threshold for cluster-based near duplicate detection. Must be
            provided together with extractor to enable clustering.
        merge_near_duplicates : bool, default True
            Whether to merge overlapping near duplicate groups.
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
        cluster_threshold: float | None = DEFAULT_DUPLICATES_CLUSTER_THRESHOLD
        merge_near_duplicates: bool = DEFAULT_DUPLICATES_MERGE_NEAR_DUPLICATES

    stats: StatsResult
    flags: ImageStats
    cluster_threshold: float | None
    merge_near_duplicates: bool
    extractor: FeatureExtractor | None
    batch_size: int | None
    cluster_algorithm: Literal["kmeans", "hdbscan"]
    n_clusters: int | None
    config: Config

    def __init__(
        self,
        flags: ImageStats | None = None,
        cluster_threshold: float | None = None,
        merge_near_duplicates: bool | None = None,
        extractor: FeatureExtractor | None = None,
        batch_size: int | None = None,
        cluster_algorithm: Literal["kmeans", "hdbscan"] | None = None,
        n_clusters: int | None = None,
        config: Config | None = None,
    ) -> None:
        super().__init__(locals())

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

    @set_metadata(state=["flags", "merge_near_duplicates"])
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
        via :func:`~dataeval.core.calculate` to avoid redundant computation.

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
            For cross-dataset detection, includes a dataset_index column.

        See Also
        --------
        evaluate : Compute hashes and find duplicates in one call
        from_clusters : Find duplicates using cluster-based detection
        """
        # Normalize to a single or list of StatsResults
        calc_results: StatsResult | list[StatsResult]
        calc_results = stats if isinstance(stats, dict) else list(stats)

        hash_stats, source_index, available_stats, dataset_steps = _prepare_hash_inputs(calc_results)
        (item_exact, item_near), (target_exact, target_near) = _detect_hash_duplicates(hash_stats, source_index)

        df = _build_duplicates_dataframe(
            (item_exact or None) if per_image else None,
            item_near if per_image else [],
            (target_exact or None) if per_target else None,
            target_near if per_target else [],
            available_stats,
            self.merge_near_duplicates,
            dataset_steps=dataset_steps,
        )
        return DuplicatesOutput(
            df,
            calculation_results=calc_results,
            merge_near_duplicates=self.merge_near_duplicates,
            flags=self.flags,
        )

    @set_metadata(state=["cluster_threshold", "cluster_algorithm", "n_clusters"])
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

        Notes
        -----
        This method identifies duplicates in embedding space. All cluster-based
        duplicates are returned as **near duplicates** because embeddings are
        approximate representations - identical embeddings don't guarantee
        pixel-identical images.

        See Also
        --------
        dataeval.core.cluster : Function to compute clusters from embeddings
        from_stats : Find duplicates from pre-computed hash statistics
        evaluate : Find duplicates by computing hashes from images
        """
        threshold = self.cluster_threshold if self.cluster_threshold is not None else 1.0
        exact_duplicates, near_duplicates = _find_cluster_duplicates(
            mst=cluster_result["mst"],
            clusters=cluster_result["clusters"],
            cluster_threshold=threshold,
        )

        # Treat ALL cluster-based duplicates as near duplicates since embeddings
        # are approximate representations.
        cluster_method_groups: MethodGroups = [(group, "cluster") for group in exact_duplicates + near_duplicates]

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
            cluster_threshold=threshold,
            merge_near_duplicates=self.merge_near_duplicates,
            flags=self.flags,
        )

    _DatasetInput = Dataset[ArrayLike] | Dataset[tuple[ArrayLike, Any, Any]]

    @overload
    def evaluate(  # pyright: ignore[reportOverlappingOverload]
        self,
        data: _DatasetInput,
        *,
        per_image: bool = True,
        per_target: Literal[False] = ...,
    ) -> SingleDuplicatesOutput: ...

    @overload
    def evaluate(  # type: ignore[reportOverlappingOverload]
        self,
        data: _DatasetInput,
        *,
        per_image: bool = True,
        per_target: Literal[True],
    ) -> SingleTargetDuplicatesOutput: ...

    @overload
    def evaluate(
        self,
        data: _DatasetInput,
        *other: _DatasetInput,
        per_image: bool = True,
        per_target: Literal[False] = ...,
    ) -> MultiDuplicatesOutput: ...

    @overload
    def evaluate(
        self,
        data: _DatasetInput,
        *other: _DatasetInput,
        per_image: bool = True,
        per_target: Literal[True],
    ) -> MultiTargetDuplicatesOutput: ...

    @set_metadata(state=["flags", "cluster_threshold", "cluster_algorithm", "n_clusters"])
    def evaluate(
        self,
        data: _DatasetInput,
        *other: _DatasetInput,
        per_image: bool = True,
        per_target: bool = False,
    ) -> SingleDuplicatesOutput | SingleTargetDuplicatesOutput | MultiDuplicatesOutput | MultiTargetDuplicatesOutput:
        """Find duplicates by computing hashes and/or analyzing embeddings.

        Performs duplicate detection using hash statistics and/or cluster-based
        analysis depending on configuration. Supports single or multiple datasets.

        Parameters
        ----------
        data : Dataset
            A dataset of images.
        *other : Dataset
            Additional datasets for cross-dataset duplicate detection.
        per_image : bool, default True
            Whether to compute hashes for full items (images/videos).
        per_target : bool, default False
            Whether to compute hashes for individual targets/detections.
            When True, accessor properties return :class:`SourceIndex` indices;
            when False, they return plain ``int`` item indices.

        Returns
        -------
        SingleDuplicatesOutput or MultiDuplicatesOutput
            Duplicate detection results as a DataFrame of duplicate groups.
            For multi-dataset input, includes a ``dataset_index`` column.

        Raises
        ------
        ValueError
            If flags is NONE and no extractor is provided.

        Examples
        --------
        Hash-based duplicates with merged near duplicates (default):

        >>> detector = Duplicates()
        >>> result = detector.evaluate(images)
        >>> result
        shape: (4, 5)
        ┌──────────┬───────┬──────────┬───────────────┬────────────────────┐
        │ group_id ┆ level ┆ dup_type ┆ item_indices  ┆ methods            │
        │ ---      ┆ ---   ┆ ---      ┆ ---           ┆ ---                │
        │ i64      ┆ str   ┆ str      ┆ list[i64]     ┆ list[str]          │
        ╞══════════╪═══════╪══════════╪═══════════════╪════════════════════╡
        │ 0        ┆ item  ┆ exact    ┆ [3, 20]       ┆ ["xxhash"]         │
        │ 1        ┆ item  ┆ exact    ┆ [7, 11, … 25] ┆ ["xxhash"]         │
        │ 2        ┆ item  ┆ exact    ┆ [16, 37]      ┆ ["xxhash"]         │
        │ 3        ┆ item  ┆ near     ┆ [0, 1, … 49]  ┆ ["dhash", "phash"] │
        └──────────┴───────┴──────────┴───────────────┴────────────────────┘

        Cross-dataset detection:

        >>> detector = Duplicates()
        >>> result = detector.evaluate(train_ds, test_ds)
        """
        if other:
            return self._evaluate_multi([data, *other], per_image=per_image, per_target=per_target)

        return self._evaluate_single(data, per_image=per_image, per_target=per_target)

    def _evaluate_single(
        self,
        data: _DatasetInput,
        *,
        per_image: bool = True,
        per_target: bool = True,
    ) -> SingleDuplicatesOutput:
        """Single-dataset evaluate implementation."""
        # Validate parameters - need either hash-based or cluster-based detection
        # Cluster-based detection requires both extractor AND cluster_threshold
        has_hash_detection = bool(self.flags & ImageStats.HASH)
        has_cluster_detection = self.extractor is not None and self.cluster_threshold is not None
        if not has_hash_detection and not has_cluster_detection:
            raise ValueError(
                "Either flags must contain hash stats, or both extractor and "
                "cluster_threshold must be provided for cluster-based detection.",
            )

        # Initialize results
        item_exact: SingleExactDuplicatesGroup = []
        item_near: MethodGroups = []
        target_exact: SingleExactTargetDuplicatesGroup = []
        target_near: MethodGroups = []
        stored_cluster_result: ClusterResult | None = None

        # Hash-based duplicate detection
        if self.flags & ImageStats.HASH:
            self.stats = compute_stats(
                data, stats=self.flags & ImageStats.HASH, per_image=per_image, per_target=per_target
            )
            (item_exact, item_near), (target_exact, target_near) = _detect_hash_duplicates(
                self.stats["stats"], self.stats["source_index"]
            )

        # Cluster-based duplicate detection (requires both extractor and cluster_threshold)
        if self.extractor is not None and self.cluster_threshold is not None:
            embeddings = Embeddings(data, self.extractor, batch_size=self.batch_size)

            stored_cluster_result = cluster(
                embeddings,
                algorithm=self.cluster_algorithm,
                n_clusters=self.n_clusters,
            )

            cluster_exact, cluster_near = _find_cluster_duplicates(
                mst=stored_cluster_result["mst"],
                clusters=stored_cluster_result["clusters"],
                cluster_threshold=self.cluster_threshold if self.cluster_threshold is not None else 1.0,
            )
            # Treat ALL cluster results as near duplicates (embeddings are approximate)
            item_near = item_near + [(group, "cluster") for group in cluster_exact + cluster_near]

        available_stats = set(self.stats["stats"].keys()) if self.flags & ImageStats.HASH else set()
        df = _build_duplicates_dataframe(
            item_exact or None,
            item_near,
            target_exact or None,
            target_near,
            available_stats,
            self.merge_near_duplicates,
        )
        return DuplicatesOutput(  # type: ignore[return-value]
            df,
            calculation_results=self.stats if has_hash_detection else None,
            cluster_result=stored_cluster_result,
            cluster_threshold=self.cluster_threshold,
            merge_near_duplicates=self.merge_near_duplicates,
            flags=self.flags,
        )

    def _evaluate_multi(
        self,
        datasets: Sequence[_DatasetInput],
        *,
        per_image: bool = True,
        per_target: bool = True,
    ) -> MultiDuplicatesOutput:
        """Multi-dataset evaluate: compute stats per dataset, then combine."""
        has_hash_detection = bool(self.flags & ImageStats.HASH)
        has_cluster_detection = self.extractor is not None and self.cluster_threshold is not None
        if not has_hash_detection and not has_cluster_detection:
            raise ValueError(
                "Either flags must contain hash stats, or both extractor and "
                "cluster_threshold must be provided for cluster-based detection.",
            )

        # Hash-based: compute stats per dataset, delegate to from_stats
        calc_results: list[StatsResult] = []
        if has_hash_detection:
            calc_results = [
                compute_stats(ds, stats=self.flags & ImageStats.HASH, per_image=per_image, per_target=per_target)
                for ds in datasets
            ]
            self.stats = calc_results[-1]

        hash_stats, source_index, available_stats, dataset_steps = (
            _prepare_hash_inputs(calc_results) if calc_results else ({}, [], set(), None)
        )

        item_exact: SingleExactDuplicatesGroup = []
        item_near: MethodGroups = []
        target_exact: SingleExactTargetDuplicatesGroup = []
        target_near: MethodGroups = []
        stored_cluster_result: ClusterResult | None = None

        if calc_results:
            (item_exact, item_near), (target_exact, target_near) = _detect_hash_duplicates(hash_stats, source_index)

        # Cluster-based: combine all images, extract, cluster together
        if has_cluster_detection:
            all_images = [item[0] if isinstance(item, tuple) else item for ds in datasets for item in ds]
            embeddings = self.extractor(all_images)  # type: ignore[union-attr]
            embeddings_array = flatten_samples(to_numpy(embeddings))

            stored_cluster_result = cluster(
                embeddings_array,
                algorithm=self.cluster_algorithm,
                n_clusters=self.n_clusters,
            )

            cluster_exact, cluster_near = _find_cluster_duplicates(
                mst=stored_cluster_result["mst"],
                clusters=stored_cluster_result["clusters"],
                cluster_threshold=self.cluster_threshold if self.cluster_threshold is not None else 1.0,
            )
            item_near = item_near + [(group, "cluster") for group in cluster_exact + cluster_near]

        df = _build_duplicates_dataframe(
            item_exact or None,
            item_near,
            target_exact or None,
            target_near,
            available_stats,
            self.merge_near_duplicates,
            dataset_steps=dataset_steps,
        )
        return DuplicatesOutput(  # type: ignore[return-value]
            df,
            calculation_results=calc_results if calc_results else None,
            cluster_result=stored_cluster_result,
            cluster_threshold=self.cluster_threshold,
            merge_near_duplicates=self.merge_near_duplicates,
            flags=self.flags,
        )
