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
from dataeval.types import DictOutput, SourceIndex, set_metadata
from dataeval.utils.arrays import flatten_samples, to_numpy


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


@dataclass(frozen=True)
class DuplicateDetectionResult(Generic[TIndexType]):
    """
    Results for duplicate detection at a specific level (item or target).

    Attributes
    ----------
    exact : Sequence[Sequence[int]] | Sequence[Sequence[SourceIndex]] | None
        Groups of exact duplicates. Each inner sequence contains indices or SourceIndex
        objects that are exact duplicates of each other. None if not computed.
        Includes both hash-based exact duplicates and cluster-based exact duplicates
        (zero distance in minimum spanning tree) when a feature_extractor is used.
    near : Sequence[Sequence[int]] | Sequence[Sequence[SourceIndex]] | None
        Groups of perceptual hashing-based near duplicates. Each inner sequence contains
        indices or SourceIndex objects that are perceptually similar but not exact
        duplicates. None if not computed.
    near_cluster : Sequence[Sequence[int]] | Sequence[Sequence[SourceIndex]] | None
        Groups of cluster-based near duplicates from minimum spanning tree analysis
        in embedding space. Each inner sequence contains indices that are semantically
        similar based on neural network embeddings. None if no feature_extractor was
        provided or only exact duplicates were requested.
    """

    exact: Sequence[Sequence[TIndexType]] | None = None
    near: Sequence[Sequence[TIndexType]] | None = None
    near_cluster: Sequence[Sequence[TIndexType]] | None = None


@dataclass(frozen=True)
class DuplicatesOutput(DictOutput):
    """
    Output class for :class:`.Duplicates` detector.

    Provides separate duplicate detection results for item-level (full images/videos)
    and target-level (bounding boxes/detections) duplicates.

    Attributes
    ----------
    items : DuplicateDetectionResult[int] | DuplicateDetectionResult[DatasetItemTuple] | None
        Duplicate groups for full items (images, videos, etc.). Indices are simple
        integers referring to the item index in the dataset for single-dataset detection.
        For cross-dataset detection, indices are DatasetItemTuple objects containing
        dataset id and item id.
        None if item-level duplicates were not computed.
    targets : DuplicateDetectionResult[SourceIndex] | DuplicateDetectionResult[DatasetItemTuple] | None
        Duplicate groups for individual targets/detections within items. Indices are
        SourceIndex objects containing (item, target, channel) information for single-dataset.
        For cross-dataset detection, indices are DatasetItemTuple objects where the id field
        contains a SourceIndex.
        None if target-level duplicates were not computed.

    Notes
    -----
    - Item indices are simple integers (e.g., [0, 5, 7]) for single-dataset
    - Target indices are SourceIndex objects with item, target, and channel info for single-dataset
    - For cross-dataset detection, indices are DatasetItemTuple objects with dataset_id and id fields
    """

    items: DuplicateDetectionResult[int] | DuplicateDetectionResult[DatasetItemTuple]
    targets: DuplicateDetectionResult[SourceIndex] | DuplicateDetectionResult[DatasetItemTuple]


class Duplicates:
    """Finds duplicate images using hashing and/or embedding-based clustering.

    Supports two complementary detection methods:

    1. **Hash-based**: Detects exact duplicates (identical pixel values) using xxhash
       and near duplicates (visually similar) using perceptual hashing with DCT.
    2. **Cluster-based**: Uses neural network embeddings to cluster images and identifies
       duplicates based on distance relationships in embedding space via minimum spanning tree.

    Both methods can be used together or independently based on the ``flags`` parameter.

    Parameters
    ----------
    flags : ImageStats, default ImageStats.HASH
        Statistics to compute for hash-based duplicate detection:

        - ``ImageStats.HASH`` (default): Compute both exact and near duplicates
        - ``ImageStats.HASH_XXHASH``: Compute only exact duplicates (faster)
        - ``ImageStats.HASH_PCHASH``: Compute only near duplicates
        - ``ImageStats.NONE``: Skip hash computation, use only cluster-based detection
          (requires ``feature_extractor``)
    feature_extractor : FeatureExtractor, optional
        Feature extractor for cluster-based duplicate detection. When provided, embeddings
        are extracted and clustered to find semantic duplicates in embedding space.
        Common extractors include :class:`~dataeval.Embeddings`.
    cluster_threshold : float, optional
        Threshold for cluster-based *near* duplicate detection. This does NOT affect
        exact duplicates (which are zero distance in the MST). When None, only exact
        cluster duplicates are detected. Lower values are stricter (fewer near duplicates).
    cluster_algorithm : {"kmeans", "hdbscan"}, default "hdbscan"
        Clustering algorithm for cluster-based detection.
    n_clusters : int, optional
        Expected number of clusters. For HDBSCAN, this is a hint that adjusts
        min_cluster_size. For KMeans, this is the exact number of clusters.

    Attributes
    ----------
    stats : CalculationResult
        Hash statistics computed during the last evaluate() call.
        Contains xxhash and pchash values for all processed images.
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

    Examples
    --------
    End-to-end hash-based detection:

    >>> detector = Duplicates()
    >>> result = detector.evaluate(dataset)

    Reuse pre-computed statistics for efficiency:

    >>> result = detector.from_stats(hashes1)

    Fast exact-only detection for large datasets:

    >>> fast_detector = Duplicates(flags=ImageStats.HASH_XXHASH)
    >>> result = fast_detector.evaluate(duplicate_images)

    Cluster-based detection with embeddings:

    >>> from dataeval import Embeddings
    >>> extractor = Embeddings(model=my_model)
    >>> detector = Duplicates(feature_extractor=extractor, cluster_threshold=1.0)
    >>> result = detector.evaluate(embeddings_dataset)

    Cluster-only detection (skip hash computation):

    >>> detector = Duplicates(flags=ImageStats.NONE, feature_extractor=extractor)
    >>> result = detector.evaluate(embeddings_dataset)
    """

    def __init__(
        self,
        flags: ImageStats = ImageStats.HASH,
        feature_extractor: FeatureExtractor | None = None,
        cluster_threshold: float | None = None,
        cluster_algorithm: Literal["kmeans", "hdbscan"] = "hdbscan",
        n_clusters: int | None = None,
    ) -> None:
        self.stats: CalculationResult
        self.flags: ImageStats = flags & ImageStats.HASH
        self.feature_extractor = feature_extractor
        self.cluster_threshold = cluster_threshold
        self.cluster_algorithm: Literal["kmeans", "hdbscan"] = cluster_algorithm
        self.n_clusters = n_clusters

    def _get_duplicates(
        self, stats: StatsMap, source_index: Sequence[SourceIndex]
    ) -> tuple[DuplicateDetectionResult[int], DuplicateDetectionResult[SourceIndex]]:
        """
        Extract duplicate groups from hash statistics, separating items and targets.

        Parameters
        ----------
        stats : StatsMap
            Hash statistics containing xxhash and pchash values
        source_index : Sequence[SourceIndex]
            Source index information for each hash value

        Returns
        -------
        tuple[DuplicateDetectionResult[int], DuplicateDetectionResult[SourceIndex] | None]
            A tuple of (item_duplicates, target_duplicates) where:

            - item_duplicates: Duplicates for full items (target=None)
            - target_duplicates: Duplicates for targets (target is not None), or None if no targets
        """
        # Separate indices into items (target=None) and targets (target is not None)
        item_indices: list[int] = []
        target_indices: list[int] = []

        for i, src_idx in enumerate(source_index):
            if src_idx.target is None:
                item_indices.append(i)
            else:
                target_indices.append(i)

        # Find item-level duplicates if items exist
        item_result = (
            DuplicateDetectionResult()
            if not item_indices
            else self._find_item_duplicates(stats, source_index, item_indices)
        )

        # Find target-level duplicates if targets exist
        target_result = (
            DuplicateDetectionResult()
            if not target_indices
            else self._find_target_duplicates(stats, source_index, target_indices)
        )

        return item_result, target_result

    def _find_item_duplicates(
        self, stats: StatsMap, source_index: Sequence[SourceIndex], item_indices: list[int]
    ) -> DuplicateDetectionResult[int]:
        """
        Find item-level duplicates from hash statistics.

        Parameters
        ----------
        stats : StatsMap
            Hash statistics containing xxhash and/or pchash values
        source_index : Sequence[SourceIndex]
            Source index information for each hash value
        item_indices : list[int]
            Indices in stats/source_index that correspond to full items (target=None)

        Returns
        -------
        DuplicateDetectionResult[int]
            Item-level duplicate groups with simple integer indices
        """
        item_exact: list[list[int]] = []
        item_near: list[list[int]] = []

        # Find exact duplicates using xxhash if available
        if "xxhash" in stats:
            item_exact_dict: dict[str, list[int]] = {}
            for i in item_indices:
                value = stats["xxhash"][i]
                item_exact_dict.setdefault(value, []).append(source_index[i].item)
            item_exact = [sorted(v) for v in item_exact_dict.values() if len(v) > 1]

        # Find near duplicates using pchash if available
        if "pchash" in stats:
            item_near_dict: dict[str, list[int]] = {}
            for i in item_indices:
                value = stats["pchash"][i]
                if value:
                    item_near_dict.setdefault(value, []).append(source_index[i].item)
            item_near = [
                sorted(v)
                for v in item_near_dict.values()
                if len(v) > 1 and not any(set(v).issubset(x) for x in item_exact)
            ]

        return DuplicateDetectionResult(exact=sorted(item_exact) or None, near=sorted(item_near) or None)

    def _find_target_duplicates(
        self, stats: StatsMap, source_index: Sequence[SourceIndex], target_indices: list[int]
    ) -> DuplicateDetectionResult[SourceIndex]:
        """
        Find target-level duplicates from hash statistics.

        Parameters
        ----------
        stats : StatsMap
            Hash statistics containing xxhash and/or pchash values
        source_index : Sequence[SourceIndex]
            Source index information for each hash value
        target_indices : list[int]
            Indices in stats/source_index that correspond to targets (target is not None)

        Returns
        -------
        DuplicateDetectionResult[SourceIndex]
            Target-level duplicate groups with SourceIndex objects
        """
        target_exact: list[list[SourceIndex]] = []
        target_near: list[list[SourceIndex]] = []

        # Find exact duplicates using xxhash if available
        if "xxhash" in stats:
            target_exact_dict: dict[str, list[SourceIndex]] = {}
            for i in target_indices:
                value = stats["xxhash"][i]
                target_exact_dict.setdefault(value, []).append(source_index[i])
            target_exact = [sorted(v) for v in target_exact_dict.values() if len(v) > 1]

        # Find near duplicates using pchash if available
        if "pchash" in stats:
            target_near_dict: dict[str, list[SourceIndex]] = {}
            for i in target_indices:
                value = stats["pchash"][i]
                if value:
                    target_near_dict.setdefault(value, []).append(source_index[i])
            target_near = [
                sorted(v)
                for v in target_near_dict.values()
                if len(v) > 1 and not any(set(v).issubset(x) for x in target_exact)
            ]

        return DuplicateDetectionResult(exact=sorted(target_exact) or None, near=sorted(target_near) or None)

    @overload
    def from_stats(self, stats: CalculationResult) -> DuplicatesOutput: ...

    @overload
    def from_stats(self, stats: Sequence[CalculationResult]) -> DuplicatesOutput: ...

    @set_metadata
    def from_stats(self, stats: CalculationResult | Sequence[CalculationResult]) -> DuplicatesOutput:
        """Find duplicates from pre-computed hash statistics.

        Analyzes previously computed hash values to identify duplicate groups
        without recomputing hashes. Separates item-level duplicates (full images/videos)
        from target-level duplicates (bounding boxes/detections). Supports both single
        dataset and cross-dataset duplicate detection.

        Parameters
        ----------
        stats : CalculationResult or Sequence[CalculationResult]
            Hash statistics from calculate() with ImageStats.HASH. Must include
            source_index information to distinguish items from targets.

            - Single CalculationResult: within-dataset duplicate detection
            - Sequence of CalculationResults: cross-dataset duplicate detection

        Returns
        -------
        DuplicatesOutput
            Duplicate detection results with separate item and target duplicate groups.

            - items: Contains item-level duplicates (indices are simple integers for single dataset,
              DatasetItemTuple objects for cross-dataset)
            - targets: Contains target-level duplicates (indices are SourceIndex objects for single dataset,
              DatasetItemTuple objects for cross-dataset), or None if no targets were processed

            For single dataset: indices are simple integers or SourceIndex objects
            For multiple datasets: indices are DatasetItemTuple objects with dataset_id and id fields

        Examples
        --------
        Single dataset - item-level duplicates only:

        >>> detector = Duplicates()
        >>> stats = calculate(duplicate_images, None, ImageStats.HASH, per_image=True, per_target=False)
        >>> result = detector.from_stats(stats)
        >>> print(result.items.exact)
        [[3, 20], [16, 37]]
        >>> print(result.targets.exact)
        None

        Single dataset - both item and target duplicates:

        >>> stats = calculate(od_dataset, None, ImageStats.HASH, per_image=True, per_target=True)
        >>> result = detector.from_stats(stats)
        >>> print(result.items.exact)
        [[1, 2]]
        >>> print(result.targets.exact)
        [[SourceIndex(0, 0), SourceIndex(0, 1)], [SourceIndex(1, 0), SourceIndex(1, 1), SourceIndex(1, 2)]]

        Cross-dataset duplicate detection:

        >>> stats1 = calculate(duplicate_images, None, ImageStats.HASH)
        >>> stats2 = calculate(od_dataset, None, ImageStats.HASH)
        >>> result = detector.from_stats([stats1, stats2])
        >>> print(result.items.exact)
        [[(0, 3), (0, 20)], [(0, 5), (1, 1), (1, 2)], [(0, 16), (0, 37)]]
        """
        # Single dataset case
        if not isinstance(stats, Sequence):
            item_duplicates, target_duplicates = self._get_duplicates(stats["stats"], stats["source_index"])
            return DuplicatesOutput(items=item_duplicates, targets=target_duplicates)

        # Multi-dataset case
        combined_stats_map, _ = combine_results(stats)

        # Combine source_index from all datasets, adjusting item indices
        # Also track which dataset each item originally came from
        combined_source_index: list[SourceIndex] = []
        dataset_id_map: dict[int, int] = {}  # Maps global item index -> dataset index
        item_dataset_steps: list[int] = []  # Cumulative item counts per dataset

        cumulative_items = 0
        for dataset_idx, result in enumerate(stats):
            for src_idx in result["source_index"]:
                # Adjust item index to be global across all datasets
                global_item = src_idx.item + cumulative_items
                adjusted_idx = SourceIndex(item=global_item, target=src_idx.target, channel=src_idx.channel)
                combined_source_index.append(adjusted_idx)

                # Track which dataset this item came from
                dataset_id_map[global_item] = dataset_idx

            # Update cumulative count - count unique items in this dataset
            unique_items = len({si.item for si in result["source_index"]})
            cumulative_items += unique_items
            item_dataset_steps.append(cumulative_items)

        item_duplicates, target_duplicates = self._get_duplicates(combined_stats_map, combined_source_index)

        # Split results back into per-dataset groups using item-based steps
        item_result = self._split_cross_dataset_duplicates(
            item_duplicates.exact,
            item_duplicates.near,
            item_dataset_steps,
        )

        # Check if target_duplicates has any actual data (not just None fields)
        if target_duplicates.exact is not None or target_duplicates.near is not None:
            target_result = self._split_cross_dataset_duplicates(
                target_duplicates.exact,
                target_duplicates.near,
                item_dataset_steps,
            )
        else:
            target_result = DuplicateDetectionResult()

        return DuplicatesOutput(items=item_result, targets=target_result)

    def _split_cross_dataset_duplicates(
        self,
        exact_groups: Sequence[Sequence[TIndexType]] | None,
        near_groups: Sequence[Sequence[TIndexType]] | None,
        dataset_steps: Sequence[int],
    ) -> DuplicateDetectionResult[DatasetItemTuple]:
        """
        Split cross-dataset duplicate groups back into list of DatasetItemTuple.

        Parameters
        ----------
        exact_groups : Sequence[Sequence[int]] | Sequence[Sequence[SourceIndex]] | None
            Exact duplicate groups with combined dataset indices
        near_groups : Sequence[Sequence[int]] | Sequence[Sequence[SourceIndex]] | None
            Near duplicate groups with combined dataset indices
        dataset_steps : Sequence[int]
            Cumulative counts of items per dataset for index mapping

        Returns
        -------
        DuplicateDetectionResult[DatasetItemTuple]
            Result where each duplicate group is a list of DatasetItemTuple objects
        """

        def split_group(
            group: Sequence[TIndexType],
        ) -> list[DatasetItemTuple]:
            """Split a duplicate group into list of DatasetItemTuple."""
            result: list[DatasetItemTuple] = []
            for idx in group:
                # Get dataset index and local index within that dataset
                if isinstance(idx, SourceIndex):
                    # For SourceIndex, map the item field and keep as SourceIndex
                    dataset_idx, local_item = get_dataset_step_from_idx(idx.item, dataset_steps)
                    local_src_idx = SourceIndex(item=local_item, target=idx.target, channel=idx.channel)
                    result.append(DatasetItemTuple(dataset_id=dataset_idx, id=local_src_idx))
                elif isinstance(idx, int):
                    # For int indices
                    dataset_idx, local_idx = get_dataset_step_from_idx(idx, dataset_steps)
                    result.append(DatasetItemTuple(dataset_id=dataset_idx, id=local_idx))
            return result

        exact_split = [split_group(group) for group in exact_groups] if exact_groups is not None else None
        near_split = [split_group(group) for group in near_groups] if near_groups is not None else None

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
        This method is particularly effective for finding semantic or visual
        duplicates in image embeddings.

        Parameters
        ----------
        cluster_result : ClusterResult
            Clustering results from the cluster() function, containing the
            minimum spanning tree (mst) and cluster assignments needed for
            duplicate detection.

        Returns
        -------
        DuplicatesOutput
            Duplicate detection results with item-level duplicate groups.
            Cluster-based detection operates on items only (no target separation).

        Notes
        -----
        This method uses cluster distance standards to identify duplicates:

        - **Exact duplicates**: Points at zero distance in the MST
        - **Near duplicates**: Points within cluster-specific distance thresholds

        Unlike hash-based duplicate detection (from_stats/evaluate), cluster-based
        detection identifies duplicates in embedding space, which can capture
        semantic or visual similarity rather than pixel-level equality.

        Cluster-based detection returns item-level duplicates only. The targets
        field will always be None since clustering operates on the embedding level.

        See Also
        --------
        dataeval.core.cluster : Function to compute clusters from embeddings
        from_stats : Find duplicates from pre-computed hash statistics
        evaluate : Find duplicates by computing hashes from images
        """
        # Find duplicates using MST and cluster assignments
        exact_duplicates, near_duplicates = self._find_duplicates(
            mst=cluster_result["mst"],
            clusters=cluster_result["clusters"],
        )

        item_result = DuplicateDetectionResult(
            exact=exact_duplicates or None,
            near_cluster=near_duplicates or None,
        )

        return DuplicatesOutput(items=item_result, targets=DuplicateDetectionResult())

    def _find_duplicates(
        self,
        mst: NDArray[np.float32],
        clusters: NDArray[np.intp],
    ) -> tuple[list[list[int]], list[list[int]]]:
        """
        Finds duplicate and near duplicate data based on cluster average distance.

        Parameters
        ----------
        mst : NDArray[np.float32]
            Minimum spanning tree from cluster() output
        clusters : NDArray[np.intp]
            Cluster labels from cluster() output

        Returns
        -------
        Tuple[List[List[int]], List[List[int]]]
            The exact duplicates and near duplicates as lists of related indices
        """
        # Delay load numba compiled functions
        from dataeval.core._fast_hdbscan._mst import compare_links_to_cluster_std

        exact_indices, near_indices = compare_links_to_cluster_std(mst, clusters)  # type: ignore
        exact_dupes = self._sorted_union_find(exact_indices)
        near_dupes = self._sorted_union_find(near_indices)

        return [[int(ii) for ii in il] for il in exact_dupes], [[int(ii) for ii in il] for il in near_dupes]

    def _sorted_union_find(self, index_groups: Any) -> list[list[Any]]:
        """Merges and sorts groups of indices that share any common index"""
        import numpy as np

        # Import disjoint set functions from our cached implementation
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
        analysis depending on configuration. When both methods are enabled,
        results are merged with exact duplicates combined and near duplicates
        separated by detection method.

        Parameters
        ----------
        data : Dataset[ArrayLike] or Dataset[tuple[ArrayLike, Any, Any]]
            Dataset of images in array format. Can be image-only dataset
            or dataset with additional tuple elements (labels, metadata).
            Images should be in standard array format (C, H, W).
        per_image : bool, default True
            Whether to compute hashes for full items (images/videos).
            When True, item-level duplicates will be detected.
        per_target : bool, default True
            Whether to compute hashes for individual targets/detections.
            When True and targets are present, target-level duplicates will be detected.
            Has no effect for datasets without targets or for cluster-based detection.

        Returns
        -------
        DuplicatesOutput
            Duplicate detection results with separate item and target duplicate groups.

            - items.exact: Exact duplicates (hash-based and/or cluster-based with zero distance)
            - items.near: Hash-based near duplicates (perceptual hash similarity)
            - items.near_cluster: Cluster-based near duplicates (embedding space similarity)
            - targets: Target-level duplicates (hash-based only, cluster detection is item-level)

        Raises
        ------
        ValueError
            If ``flags`` is ``ImageStats.NONE`` and no ``feature_extractor`` is provided.

        Examples
        --------
        Hash-based duplicates only (default):

        >>> detector = Duplicates()
        >>> result = detector.evaluate(duplicate_images)
        >>> print(result.items.exact)
        [[3, 20], [16, 37]]
        >>> print(result.items.near_cluster)
        None

        Cluster-based duplicates with embeddings:

        >>> from dataeval import Embeddings
        >>> extractor = Embeddings(model=my_model)
        >>> detector = Duplicates(feature_extractor=extractor, cluster_threshold=1.0)
        >>> result = detector.evaluate(embeddings_dataset)

        Cluster-only detection (skip hash computation):

        >>> detector = Duplicates(flags=ImageStats.NONE, feature_extractor=extractor)
        >>> result = detector.evaluate(embeddings_dataset)
        >>> print(result.items.near)  # None - hash-based near not computed
        None

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
            # Extract embeddings
            embeddings = self.feature_extractor(data)
            embeddings_array = flatten_samples(to_numpy(embeddings))

            # Cluster the embeddings
            cluster_result = cluster(
                embeddings_array,
                algorithm=self.cluster_algorithm,
                n_clusters=self.n_clusters,
            )

            # Find cluster-based duplicates
            cluster_exact, cluster_near = self._find_duplicates(
                mst=cluster_result["mst"],
                clusters=cluster_result["clusters"],
            )

            # If cluster_threshold is None, don't include near duplicates from clusters
            if self.cluster_threshold is None:
                cluster_near = []

        # Merge results
        final_item_result = self._merge_item_results(hash_item_result, cluster_exact, cluster_near)
        final_target_result = hash_target_result or DuplicateDetectionResult()

        return DuplicatesOutput(items=final_item_result, targets=final_target_result)

    def _merge_item_results(
        self,
        hash_result: DuplicateDetectionResult[int] | None,
        cluster_exact: list[list[int]],
        cluster_near: list[list[int]],
    ) -> DuplicateDetectionResult[int]:
        """Merge hash-based and cluster-based item duplicate results."""
        if hash_result is None and not cluster_exact and not cluster_near:
            return DuplicateDetectionResult()

        if hash_result is None:
            # Only cluster results
            return DuplicateDetectionResult(
                exact=cluster_exact if cluster_exact else None,
                near=None,
                near_cluster=cluster_near if cluster_near else None,
            )

        if not cluster_exact and not cluster_near:
            # Only hash results
            return hash_result

        # Merge both - combine exact duplicates, keep near separate
        merged_exact = self._merge_duplicate_groups(
            list(hash_result.exact or []),
            cluster_exact,
        )

        return DuplicateDetectionResult(
            exact=merged_exact if merged_exact else None,
            near=hash_result.near,
            near_cluster=cluster_near if cluster_near else None,
        )

    def _merge_duplicate_groups(
        self,
        groups_a: Sequence[Sequence[int]],
        groups_b: Sequence[Sequence[int]],
    ) -> list[list[int]]:
        """Merge two sets of duplicate groups, combining overlapping groups."""
        # Convert to sets for easier manipulation
        all_groups = [set(g) for g in groups_a] + [set(g) for g in groups_b]

        if not all_groups:
            return []

        # Merge overlapping groups using union-find approach
        merged: list[set[int]] = []
        for group in all_groups:
            # Find all existing groups that overlap with this one
            overlapping = [i for i, m in enumerate(merged) if m & group]

            if not overlapping:
                merged.append(group)
            else:
                # Merge all overlapping groups with the new group
                new_group = group.copy()
                for i in sorted(overlapping, reverse=True):
                    new_group |= merged.pop(i)
                merged.append(new_group)

        # Convert back to sorted lists, filter to groups with more than 1 element
        return sorted([sorted(g) for g in merged if len(g) > 1])
