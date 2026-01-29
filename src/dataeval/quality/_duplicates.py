"""Duplicate detection for images using hashing and clustering."""

__all__ = []

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Generic, Literal, NamedTuple, TypeVar, overload

import numpy as np
from numpy.typing import NDArray

from dataeval.core._calculate import CalculationResult, calculate
from dataeval.core._clusterer import ClusterResult, cluster
from dataeval.flags import ImageStats
from dataeval.protocols import ArrayLike, Dataset, FeatureExtractor
from dataeval.quality._results import StatsMap, combine_results, get_dataset_step_from_idx
from dataeval.types import ClusterConfigMixin, DictOutput, Evaluator, EvaluatorConfig, SourceIndex, set_metadata
from dataeval.utils.arrays import flatten_samples, to_numpy

DEFAULT_DUPLICATES_FLAGS = ImageStats.HASH_DUPLICATES_BASIC
DEFAULT_DUPLICATES_CLUSTER_THRESHOLD: float | None = None
DEFAULT_DUPLICATES_MERGE_NEAR_DUPLICATES = True


class DatasetItemTuple(NamedTuple):
    """Tuple representing an item within a specific dataset.

    Attributes
    ----------
    dataset_id : int
        The ID of the dataset.
    id : int | SourceIndex
        The ID of the item or target within the dataset.
    """

    dataset_id: int
    id: int | SourceIndex

    def __repr__(self) -> str:
        return f"({self.dataset_id}, {self.id})"


TIndexType = TypeVar("TIndexType", int, SourceIndex, DatasetItemTuple)


_BASIC_HASH_METHODS = frozenset({"phash", "dhash"})
_D4_HASH_METHODS = frozenset({"phash_d4", "dhash_d4"})


@dataclass(frozen=True)
class NearDuplicateGroup(Generic[TIndexType]):
    """A group of near-duplicate items with detection method metadata.

    Attributes
    ----------
    indices : Sequence[TIndexType]
        The indices of items in this near-duplicate group.
    methods : frozenset[str]
        The detection methods that identified this group. Possible values include:
        "phash", "dhash" (basic perceptual hashes), "phash_d4", "dhash_d4"
        (rotation/flip-invariant D4 hashes), and "cluster" (embedding-based).
    orientation : Literal["rotated", "same"] | None, optional
        Indicates whether the duplicates are of the same orientation or
        rotated/flipped versions. Set automatically when both basic (phash/dhash)
        and D4 (phash_d4/dhash_d4) hashes are computed:
        - "same": Detected by basic hashes (same orientation)
        - "rotated": Detected only by D4 hashes (rotated/flipped)
        - None: Cannot determine (only one hash type was computed)
    """

    indices: Sequence[TIndexType]
    methods: frozenset[str]
    orientation: Literal["rotated", "same"] | None = None

    def __repr__(self) -> str:
        orientation = f", orientation={self.orientation}" if self.orientation else ""
        return f"NearDuplicateGroup({list(self.indices)}, methods={sorted(self.methods)}{orientation})"


@dataclass(frozen=True)
class DuplicateDetectionResult(Generic[TIndexType]):
    """
    Results for duplicate detection at a specific level (item or target).

    Attributes
    ----------
    exact : Sequence[Sequence[TIndexType]] | None
        Groups of exact duplicates (identical pixel values via xxhash).
        Each inner sequence contains indices that are exact duplicates.
        Includes cluster-based exact duplicates (zero distance in MST)
        when a feature_extractor is used. None if not computed.
    near : Sequence[NearDuplicateGroup[TIndexType]] | None
        Groups of near duplicates with detection method metadata.
        Each group contains indices and the set of methods that detected it.
        None if not computed.

    Notes
    -----
    Near duplicate detection methods are complementary:

    - **phash**: DCT frequency domain - best for compression artifacts
    - **dhash**: Gradient domain - robust to brightness changes
    - **phash_d4**: DCT frequency domain with D4 symmetry - rotation/flip invariant
    - **dhash_d4**: Gradient domain with D4 symmetry - rotation/flip invariant
    - **cluster**: Embedding space - semantic similarity

    Use ``HASH_DUPLICATES_BASIC`` for standard detection or
    ``HASH_DUPLICATES_D4`` for rotation/flip-invariant detection.

    When ``merge_near_duplicates=True`` (default), overlapping groups from
    different methods are merged, and the methods set shows which detection
    methods identified the group. Groups detected by multiple methods
    indicate higher confidence.
    """

    exact: Sequence[Sequence[TIndexType]] | None = None
    near: Sequence[NearDuplicateGroup[TIndexType]] | None = None


@dataclass(frozen=True)
class DuplicatesOutput(DictOutput):
    """
    Output class for :class:`.Duplicates` detector.

    Provides separate duplicate detection results for item-level (full images/videos)
    and target-level (bounding boxes/detections) duplicates.

    Attributes
    ----------
    items : DuplicateDetectionResult[int] | DuplicateDetectionResult[DatasetItemTuple]
        Duplicate groups for full items (images, videos, etc.). Indices are simple
        integers referring to the item index in the dataset for single-dataset detection.
        For cross-dataset detection, indices are DatasetItemTuple objects containing
        dataset id and item id.
    targets : DuplicateDetectionResult[SourceIndex] | DuplicateDetectionResult[DatasetItemTuple]
        Duplicate groups for individual targets/detections within items. Indices are
        SourceIndex objects containing (item, target, channel) information for single-dataset.
        For cross-dataset detection, indices are DatasetItemTuple objects where the id field
        contains a SourceIndex.

    Notes
    -----
    - Item indices are simple integers (e.g., [0, 5, 7]) for single-dataset
    - Target indices are SourceIndex objects with item, target, and channel info
    - For cross-dataset detection, indices are DatasetItemTuple objects
    """

    items: DuplicateDetectionResult[int] | DuplicateDetectionResult[DatasetItemTuple]
    targets: DuplicateDetectionResult[SourceIndex] | DuplicateDetectionResult[DatasetItemTuple]


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
    - ``ImageStats.HASH_DUPLICATES_D4``: Rotation/flip-invariant detection
      (xxhash + phash_d4 + dhash_d4)

    Parameters
    ----------
    flags : ImageStats, default ImageStats.HASH_DUPLICATES_BASIC
        Statistics to compute for hash-based duplicate detection:

        - ``ImageStats.HASH_DUPLICATES_BASIC`` (default): Standard detection with exact
          and perceptual hashes (xxhash + phash + dhash)
        - ``ImageStats.HASH_DUPLICATES_D4``: Rotation/flip-invariant detection
          using D4 symmetry hashes (xxhash + phash_d4 + dhash_d4)
        - ``ImageStats.HASH``: Compute all hash types (includes both basic and D4 variants)
        - ``ImageStats.HASH_XXHASH``: Compute only exact duplicates (fastest)
        - ``ImageStats.HASH_PHASH``: Compute only phash-based near duplicates
        - ``ImageStats.HASH_DHASH``: Compute only dhash-based near duplicates
        - ``ImageStats.NONE``: Skip hash computation, use only cluster-based detection
    feature_extractor : FeatureExtractor, optional
        Feature extractor for cluster-based duplicate detection. When provided,
        embeddings are extracted and clustered to find semantic duplicates.
    cluster_threshold : float, optional
        Threshold for cluster-based *near* duplicate detection. This does NOT affect
        exact duplicates (which are zero distance in the MST). When None, only exact
        cluster duplicates are detected. Lower values are stricter.
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
    stats : CalculationResult
        Hash statistics computed during the last evaluate() call.
    flags : ImageStats
        Statistics to compute for duplicate detection.
    feature_extractor : FeatureExtractor | None
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

    >>> from dataeval import Embeddings
    >>> extractor = Embeddings(encoder=encoder)
    >>> detector = Duplicates(feature_extractor=extractor, cluster_threshold=1.0)
    >>> result = detector.evaluate(train_ds)

    Using configuration:

    >>> config = Duplicates.Config(cluster_algorithm="kmeans", merge_near_duplicates=False)
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
            Threshold for cluster-based near duplicate detection.
        cluster_algorithm : {"kmeans", "hdbscan"}, default "hdbscan"
            Clustering algorithm for cluster-based detection.
        n_clusters : int or None, default None
            Expected number of clusters.
        merge_near_duplicates : bool, default True
            Whether to merge overlapping near duplicate groups.
        """

        flags: ImageStats = DEFAULT_DUPLICATES_FLAGS
        cluster_threshold: float | None = DEFAULT_DUPLICATES_CLUSTER_THRESHOLD
        merge_near_duplicates: bool = DEFAULT_DUPLICATES_MERGE_NEAR_DUPLICATES

    stats: CalculationResult
    flags: ImageStats
    cluster_threshold: float | None
    cluster_algorithm: Literal["kmeans", "hdbscan"]
    n_clusters: int | None
    merge_near_duplicates: bool
    config: Config
    feature_extractor: FeatureExtractor | None

    def __init__(
        self,
        flags: ImageStats | None = None,
        cluster_threshold: float | None = None,
        cluster_algorithm: Literal["kmeans", "hdbscan"] | None = None,
        n_clusters: int | None = None,
        merge_near_duplicates: bool | None = None,
        config: Config | None = None,
        feature_extractor: FeatureExtractor | None = None,
    ) -> None:
        super().__init__(locals())
        self.feature_extractor = feature_extractor

    def _get_duplicates(
        self, stats: StatsMap, source_index: Sequence[SourceIndex]
    ) -> tuple[DuplicateDetectionResult[int], DuplicateDetectionResult[SourceIndex]]:
        """
        Extract duplicate groups from hash statistics, separating items and targets.

        Parameters
        ----------
        stats : StatsMap
            Hash statistics containing xxhash and/or perceptual hash values.
        source_index : Sequence[SourceIndex]
            Source index information for each hash value.

        Returns
        -------
        tuple[DuplicateDetectionResult[int], DuplicateDetectionResult[SourceIndex]]
            Item-level and target-level duplicate results.
        """
        # Separate indices for full items vs individual targets
        item_indices: list[int] = []
        target_indices: list[int] = []

        for i, src_idx in enumerate(source_index):
            if src_idx.target is None:
                item_indices.append(i)
            else:
                target_indices.append(i)

        # Find duplicates at each level
        item_result = self._find_item_duplicates(stats, source_index, item_indices)
        target_result = self._find_target_duplicates(stats, source_index, target_indices)

        return item_result, target_result

    def _find_item_duplicates(
        self, stats: StatsMap, source_index: Sequence[SourceIndex], item_indices: list[int]
    ) -> DuplicateDetectionResult[int]:
        """
        Find item-level duplicates from hash statistics.

        Parameters
        ----------
        stats : StatsMap
            Hash statistics containing hash values.
        source_index : Sequence[SourceIndex]
            Source index information for each hash value.
        item_indices : list[int]
            Indices in stats/source_index that correspond to full items.

        Returns
        -------
        DuplicateDetectionResult[int]
            Item-level duplicate groups with simple integer indices.
        """
        item_exact: list[list[int]] = []

        # Find exact duplicates using xxhash if available
        if "xxhash" in stats:
            item_exact_dict: dict[str, list[int]] = {}
            for i in item_indices:
                value = stats["xxhash"][i]
                item_exact_dict.setdefault(value, []).append(source_index[i].item)
            item_exact = [sorted(v) for v in item_exact_dict.values() if len(v) > 1]

        # Collect near duplicates from each method with method labels
        method_groups: list[tuple[list[int], str]] = []
        methods = ["phash", "dhash", "phash_d4", "dhash_d4"]

        for method in methods:
            if method in stats:
                for group in self._find_hash_duplicates(stats, method, source_index, item_indices, item_exact):
                    method_groups.append((group, method))

        # Merge or keep separate based on configuration
        available_stats = set(stats.keys()) & set(methods)
        near_groups = self._build_near_duplicate_groups(method_groups, available_stats)

        return DuplicateDetectionResult(
            exact=sorted(item_exact) or None,
            near=near_groups or None,
        )

    def _find_hash_duplicates(
        self,
        stats: StatsMap,
        hash_key: str,
        source_index: Sequence[SourceIndex],
        indices: list[int],
        exact_groups: list[list[int]],
    ) -> list[list[int]]:
        """
        Find near duplicates for a specific hash type.

        Parameters
        ----------
        stats : StatsMap
            Hash statistics.
        hash_key : str
            Key for the hash type in stats (e.g., "phash", "dhash", "phash_d4", "dhash_d4").
        source_index : Sequence[SourceIndex]
            Source index information.
        indices : list[int]
            Indices to process.
        exact_groups : list[list[int]]
            Exact duplicate groups to exclude from near duplicates.

        Returns
        -------
        list[list[int]]
            Near duplicate groups for this hash type.
        """
        near_dict: dict[str, list[int]] = {}
        for i in indices:
            value = stats[hash_key][i]
            if value:  # Skip empty hashes
                near_dict.setdefault(value, []).append(source_index[i].item)

        # Filter: more than one item, not a subset of exact duplicates
        return [
            sorted(v) for v in near_dict.values() if len(v) > 1 and not any(set(v).issubset(x) for x in exact_groups)
        ]

    def _find_target_duplicates(
        self, stats: StatsMap, source_index: Sequence[SourceIndex], target_indices: list[int]
    ) -> DuplicateDetectionResult[SourceIndex]:
        """
        Find target-level duplicates from hash statistics.

        Parameters
        ----------
        stats : StatsMap
            Hash statistics containing hash values.
        source_index : Sequence[SourceIndex]
            Source index information for each hash value.
        target_indices : list[int]
            Indices in stats/source_index that correspond to targets.

        Returns
        -------
        DuplicateDetectionResult[SourceIndex]
            Target-level duplicate groups with SourceIndex indices.
        """
        target_exact: list[list[SourceIndex]] = []

        # Find exact duplicates using xxhash if available
        if "xxhash" in stats:
            target_exact_dict: dict[str, list[SourceIndex]] = {}
            for i in target_indices:
                value = stats["xxhash"][i]
                target_exact_dict.setdefault(value, []).append(source_index[i])
            target_exact = [sorted(v) for v in target_exact_dict.values() if len(v) > 1]

        # Collect near duplicates from each method with method labels
        method_groups: list[tuple[list[SourceIndex], str]] = []
        methods = ["phash", "dhash", "phash_d4", "dhash_d4"]

        for method in methods:
            if method in stats:
                for group in self._find_target_hash_duplicates(
                    stats, method, source_index, target_indices, target_exact
                ):
                    method_groups.append((group, method))

        # Merge or keep separate based on configuration
        available_stats = set(stats.keys()) & set(methods)
        near_groups = self._build_near_duplicate_groups(method_groups, available_stats)

        return DuplicateDetectionResult(
            exact=sorted(target_exact) or None,
            near=near_groups or None,
        )

    def _find_target_hash_duplicates(
        self,
        stats: StatsMap,
        hash_key: str,
        source_index: Sequence[SourceIndex],
        indices: list[int],
        exact_groups: list[list[SourceIndex]],
    ) -> list[list[SourceIndex]]:
        """
        Find target-level near duplicates for a specific hash type.

        Parameters
        ----------
        stats : StatsMap
            Hash statistics.
        hash_key : str
            Key for the hash type in stats.
        source_index : Sequence[SourceIndex]
            Source index information.
        indices : list[int]
            Indices to process.
        exact_groups : list[list[SourceIndex]]
            Exact duplicate groups to exclude.

        Returns
        -------
        list[list[SourceIndex]]
            Near duplicate groups for this hash type.
        """
        near_dict: dict[str, list[SourceIndex]] = {}
        for i in indices:
            value = stats[hash_key][i]
            if value:  # Skip empty hashes
                near_dict.setdefault(value, []).append(source_index[i])

        # Filter: more than one item, not a subset of exact duplicates
        return [
            sorted(v) for v in near_dict.values() if len(v) > 1 and not any(set(v).issubset(x) for x in exact_groups)
        ]

    def _get_orientation(self, methods: frozenset[str]) -> Literal["rotated", "same"]:
        """Determine orientation based on which methods detected the group."""
        has_basic = bool(methods & _BASIC_HASH_METHODS)
        has_d4 = bool(methods & _D4_HASH_METHODS)
        if has_d4 and not has_basic:
            return "rotated"
        return "same"

    def _build_near_duplicate_groups(
        self,
        method_groups: Sequence[tuple[Sequence[Any], str]],
        available_stats: set[str],
    ) -> list[NearDuplicateGroup[Any]]:
        """
        Build NearDuplicateGroup objects, optionally merging overlapping groups.

        Parameters
        ----------
        method_groups : Sequence[tuple[Sequence[Any], str]]
            List of (indices, method_name) tuples from each detection method.
        available_stats : set[str]
            Set of hash types that were computed (e.g., {"phash", "dhash", "phash_d4"}).

        Returns
        -------
        list[NearDuplicateGroup[Any]]
            Near duplicate groups with method metadata.
        """
        if not method_groups:
            return []

        # Determine if we can compute orientation (need both basic and D4 hashes)
        has_basic_stats = bool(available_stats & _BASIC_HASH_METHODS)
        has_d4_stats = bool(available_stats & _D4_HASH_METHODS)
        is_unknown = not (has_basic_stats and has_d4_stats)

        if not self.merge_near_duplicates:
            # Keep groups separate - each group has a single method
            groups = [
                NearDuplicateGroup(
                    indices=tuple(sorted(group)),
                    methods=frozenset({method}),
                    orientation=None if is_unknown else self._get_orientation(frozenset({method})),
                )
                for group, method in method_groups
            ]
            return sorted(groups, key=lambda g: tuple(g.indices))

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

        # Convert to NearDuplicateGroup objects
        result = [
            NearDuplicateGroup(
                indices=tuple(sorted(indices)),
                methods=frozenset(methods),
                orientation=None if is_unknown else self._get_orientation(frozenset(methods)),
            )
            for indices, methods in merged
            if len(indices) > 1
        ]
        return sorted(result, key=lambda g: tuple(g.indices))

    @overload
    def from_stats(
        self,
        stats: CalculationResult,
    ) -> DuplicatesOutput: ...

    @overload
    def from_stats(
        self,
        stats: CalculationResult,
        *other_stats: CalculationResult,
    ) -> DuplicatesOutput: ...

    @overload
    def from_stats(
        self,
        stats: Sequence[CalculationResult],
    ) -> DuplicatesOutput: ...

    @set_metadata
    def from_stats(
        self,
        stats: CalculationResult | Sequence[CalculationResult],
        *other_stats: CalculationResult,
    ) -> DuplicatesOutput:
        """
        Find duplicates from pre-computed hash statistics.

        Use this method when hash statistics have already been computed
        via :func:`~dataeval.core.calculate` to avoid redundant computation.

        Parameters
        ----------
        stats : CalculationResult | Sequence[CalculationResult]
            Pre-computed statistics containing hash values. Must include
            at least one of: xxhash, phash, dhash, rhash. Can be a single
            result, a sequence of results, or multiple results passed as
            positional arguments.
        *other_stats : CalculationResult
            Additional statistics from other datasets for cross-dataset
            duplicate detection.

        Returns
        -------
        DuplicatesOutput
            Duplicate detection results with separate item and target groups.
            For cross-dataset detection, indices are DatasetItemTuple objects.

        See Also
        --------
        evaluate : Compute hashes and find duplicates in one call
        from_clusters : Find duplicates using cluster-based detection
        """
        # Handle single stats case
        if isinstance(stats, dict) and not other_stats:
            item_result, target_result = self._get_duplicates(stats["stats"], stats["source_index"])
            return DuplicatesOutput(items=item_result, targets=target_result)

        # Handle multiple stats case
        stats_list: list[CalculationResult]
        stats_list = [stats, *other_stats] if isinstance(stats, dict) else list(stats)

        # Combine stats from multiple datasets
        combined_stats, dataset_steps = combine_results(stats_list)

        # Combine source_index from all stats, offsetting item indices to be global
        combined_source_index: list[SourceIndex] = []
        offset = 0
        for stat in stats_list:
            for src_idx in stat["source_index"]:
                # Offset the item index to be global across all datasets
                global_src_idx = SourceIndex(
                    item=src_idx.item + offset,
                    target=src_idx.target,
                    channel=src_idx.channel,
                )
                combined_source_index.append(global_src_idx)
            offset += len(stat["source_index"])

        # Find duplicates in combined stats
        item_result, target_result = self._get_duplicates(combined_stats, combined_source_index)

        # Split results back to per-dataset indices
        final_item = self._split_duplicate_result(item_result.exact, item_result.near, dataset_steps)
        final_target = self._split_duplicate_result(target_result.exact, target_result.near, dataset_steps)

        return DuplicatesOutput(items=final_item, targets=final_target)

    def _split_duplicate_result(
        self,
        exact_groups: Sequence[Sequence[int]] | Sequence[Sequence[SourceIndex]] | None,
        near_groups: Sequence[NearDuplicateGroup[int]] | Sequence[NearDuplicateGroup[SourceIndex]] | None,
        dataset_steps: Sequence[int],
    ) -> DuplicateDetectionResult[DatasetItemTuple]:
        """
        Split combined duplicate groups back into per-dataset references.

        Parameters
        ----------
        exact_groups : Sequence[Sequence[int]] | Sequence[Sequence[SourceIndex]] | None
            Exact duplicate groups with combined dataset indices.
        near_groups : Sequence[NearDuplicateGroup[int]] | Sequence[NearDuplicateGroup[SourceIndex]] | None
            Near duplicate groups with combined dataset indices.
        dataset_steps : Sequence[int]
            Cumulative counts of items per dataset for index mapping.

        Returns
        -------
        DuplicateDetectionResult[DatasetItemTuple]
            Result with DatasetItemTuple objects for cross-dataset references.
        """

        def split_indices(
            indices: Sequence[int] | Sequence[SourceIndex],
        ) -> tuple[DatasetItemTuple, ...]:
            """Split indices into DatasetItemTuple objects."""
            result: list[DatasetItemTuple] = []
            for idx in indices:
                if isinstance(idx, SourceIndex):
                    dataset_idx, local_item = get_dataset_step_from_idx(idx.item, dataset_steps)
                    local_src_idx = SourceIndex(item=local_item, target=idx.target, channel=idx.channel)
                    result.append(DatasetItemTuple(dataset_id=dataset_idx, id=local_src_idx))
                elif isinstance(idx, int):
                    dataset_idx, local_idx = get_dataset_step_from_idx(idx, dataset_steps)
                    result.append(DatasetItemTuple(dataset_id=dataset_idx, id=local_idx))
            return tuple(result)

        exact_split = [list(split_indices(group)) for group in exact_groups] if exact_groups else None
        near_split = (
            [
                NearDuplicateGroup(indices=split_indices(g.indices), methods=g.methods, orientation=g.orientation)
                for g in near_groups
            ]
            if near_groups
            else None
        )

        return DuplicateDetectionResult(exact=exact_split, near=near_split)

    @set_metadata
    def from_clusters(
        self,
        cluster_result: ClusterResult,
    ) -> DuplicatesOutput:
        """
        Find duplicates using cluster-based detection from minimum spanning tree.

        Analyzes the minimum spanning tree and cluster assignments to identify
        exact and near duplicates based on distance relationships within clusters.

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
        This method identifies duplicates in embedding space:

        - **Exact duplicates**: Points at zero distance in the MST
        - **Near duplicates**: Points within cluster-specific distance thresholds

        See Also
        --------
        dataeval.core.cluster : Function to compute clusters from embeddings
        from_stats : Find duplicates from pre-computed hash statistics
        evaluate : Find duplicates by computing hashes from images
        """
        exact_duplicates, near_duplicates = self._find_duplicates(
            mst=cluster_result["mst"],
            clusters=cluster_result["clusters"],
        )

        # Convert near duplicates to NearDuplicateGroup with "cluster" method
        near_groups: list[NearDuplicateGroup[int]] | None = None
        if near_duplicates:
            near_groups = [
                NearDuplicateGroup(indices=tuple(group), methods=frozenset({"cluster"})) for group in near_duplicates
            ]

        item_result = DuplicateDetectionResult(
            exact=exact_duplicates or None,
            near=near_groups,
        )

        return DuplicatesOutput(items=item_result, targets=DuplicateDetectionResult())

    def _find_duplicates(
        self,
        mst: NDArray[np.float32],
        clusters: NDArray[np.intp],
    ) -> tuple[list[list[int]], list[list[int]]]:
        """
        Find duplicate and near duplicate data based on cluster average distance.

        Parameters
        ----------
        mst : NDArray[np.float32]
            Minimum spanning tree from cluster() output.
        clusters : NDArray[np.intp]
            Cluster labels from cluster() output.

        Returns
        -------
        tuple[list[list[int]], list[list[int]]]
            Exact duplicates and near duplicates as lists of related indices.
        """
        from dataeval.core._fast_hdbscan._mst import compare_links_to_cluster_std

        exact_indices, near_indices = compare_links_to_cluster_std(mst, clusters)  # type: ignore
        exact_dupes = self._sorted_union_find(exact_indices)
        near_dupes = self._sorted_union_find(near_indices)

        return [[int(ii) for ii in il] for il in exact_dupes], [[int(ii) for ii in il] for il in near_dupes]

    def _sorted_union_find(self, index_groups: Any) -> list[list[Any]]:
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

    @set_metadata(state=["flags", "cluster_threshold", "cluster_algorithm", "n_clusters"])
    def evaluate(
        self,
        data: Dataset[ArrayLike] | Dataset[tuple[ArrayLike, Any, Any]],
        *,
        per_image: bool = True,
        per_target: bool = True,
    ) -> DuplicatesOutput:
        """Find duplicates by computing hashes and/or analyzing embeddings.

        Performs duplicate detection using hash statistics and/or cluster-based
        analysis depending on configuration.

        Parameters
        ----------
        data : Dataset[ArrayLike] or Dataset[tuple[ArrayLike, Any, Any]]
            Dataset of images in array format.
        per_image : bool, default True
            Whether to compute hashes for full items (images/videos).
        per_target : bool, default True
            Whether to compute hashes for individual targets/detections.

        Returns
        -------
        DuplicatesOutput
            Duplicate detection results with separate item and target groups.

            - items.exact: Exact duplicates (hash-based and/or cluster-based)
            - items.near: Near duplicate groups with detection method metadata.
              Each group has ``indices`` and ``methods`` (e.g., {"phash", "rhash"}).
            - targets: Target-level duplicates (hash-based only)

        Raises
        ------
        ValueError
            If flags is NONE and no feature_extractor is provided.

        Examples
        --------
        Hash-based duplicates with merged near duplicates (default):

        >>> detector = Duplicates()
        >>> result = detector.evaluate(images)
        >>> print(result.items.exact)
        [[3, 20], [7, 11, 18, 25], [16, 37]]
        >>> for group in result.items.near:
        ...     print(f"Index count: {len(group.indices)}, Methods: {sorted(group.methods)}")
        Index count: 50, Methods: ['dhash', 'phash']

        Fast exact-only detection:

        >>> detector = Duplicates(flags=ImageStats.HASH_XXHASH)
        >>> result = detector.evaluate(images)

        Combined hash and cluster-based detection:

        >>> from dataeval import Embeddings
        >>> extractor = Embeddings(encoder=encoder)
        >>> detector = Duplicates(feature_extractor=extractor, cluster_threshold=1.0)
        >>> result = detector.evaluate(train_ds)
        """
        # Validate parameters
        if not (self.flags & ImageStats.HASH) and self.feature_extractor is None:
            raise ValueError("Either flags must contain hash stats or feature_extractor must be provided.")

        # Initialize results
        hash_item_result: DuplicateDetectionResult[int] | None = None
        hash_target_result: DuplicateDetectionResult[SourceIndex] | None = None
        cluster_exact: list[list[int]] = []
        cluster_near: list[list[int]] = []

        # Hash-based duplicate detection
        if self.flags & ImageStats.HASH:
            self.stats = calculate(data, None, self.flags & ImageStats.HASH, per_image=per_image, per_target=per_target)
            hash_item_result, hash_target_result = self._get_duplicates(self.stats["stats"], self.stats["source_index"])

        # Cluster-based duplicate detection
        if self.feature_extractor is not None:
            embeddings = self.feature_extractor(data)
            embeddings_array = flatten_samples(to_numpy(embeddings))

            cluster_result = cluster(
                embeddings_array,
                algorithm=self.cluster_algorithm,
                n_clusters=self.n_clusters,
            )

            cluster_exact, cluster_near = self._find_duplicates(
                mst=cluster_result["mst"],
                clusters=cluster_result["clusters"],
            )

            if self.cluster_threshold is None:
                cluster_near = []

        # Merge results
        available_stats = set(self.stats["stats"].keys()) if self.flags & ImageStats.HASH else set()
        final_item_result = self._merge_item_results(hash_item_result, cluster_exact, cluster_near, available_stats)
        final_target_result = hash_target_result or DuplicateDetectionResult()

        return DuplicatesOutput(items=final_item_result, targets=final_target_result)

    def _merge_item_results(
        self,
        hash_result: DuplicateDetectionResult[int] | None,
        cluster_exact: list[list[int]],
        cluster_near: list[list[int]],
        available_stats: set[str],
    ) -> DuplicateDetectionResult[int]:
        """Merge hash-based and cluster-based item duplicate results."""
        if hash_result is None and not cluster_exact and not cluster_near:
            return DuplicateDetectionResult()

        # Convert cluster_near to method_groups format for merging
        cluster_method_groups: list[tuple[Sequence[Any], str]] = [(group, "cluster") for group in cluster_near]

        if hash_result is None:
            # Only cluster results - no hash stats available for orientation
            near_groups = self._build_near_duplicate_groups(cluster_method_groups, available_stats)
            return DuplicateDetectionResult(
                exact=cluster_exact if cluster_exact else None,
                near=near_groups or None,
            )

        if not cluster_exact and not cluster_near:
            return hash_result

        # Merge both - combine exact duplicates
        merged_exact = self._merge_duplicate_groups(
            list(hash_result.exact or []),
            cluster_exact,
        )

        # Combine near duplicates from hash and cluster
        hash_method_groups: list[tuple[Sequence[Any], str]] = []
        if hash_result.near:
            for g in hash_result.near:
                # Each hash group may have multiple methods already
                for method in g.methods:
                    hash_method_groups.append((list(g.indices), method))

        all_method_groups = hash_method_groups + cluster_method_groups
        merged_near = self._build_near_duplicate_groups(all_method_groups, available_stats)

        return DuplicateDetectionResult(
            exact=merged_exact if merged_exact else None,
            near=merged_near or None,
        )

    def _merge_duplicate_groups(
        self,
        groups_a: Sequence[Sequence[int]],
        groups_b: Sequence[Sequence[int]],
    ) -> list[list[int]]:
        """Merge two sets of duplicate groups, combining overlapping groups."""
        all_groups = [set(g) for g in groups_a] + [set(g) for g in groups_b]

        if not all_groups:
            return []

        merged: list[set[int]] = []
        for group in all_groups:
            overlapping = [i for i, m in enumerate(merged) if m & group]

            if not overlapping:
                merged.append(group)
            else:
                new_group = group.copy()
                for i in sorted(overlapping, reverse=True):
                    new_group |= merged.pop(i)
                merged.append(new_group)

        return sorted([sorted(g) for g in merged if len(g) > 1])
