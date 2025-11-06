from __future__ import annotations

__all__ = []

from collections.abc import Sequence
from typing import Any, overload

import numpy as np
from numpy.typing import NDArray

from dataeval.core._clusterer import ClusterResult
from dataeval.data._images import Images
from dataeval.metrics.stats import hashstats
from dataeval.metrics.stats._base import combine_stats, get_dataset_step_from_idx
from dataeval.outputs import DuplicatesOutput, HashStatsOutput
from dataeval.outputs._base import set_metadata
from dataeval.outputs._linters import DatasetDuplicateGroupMap, DuplicateGroup
from dataeval.protocols import ArrayLike, Dataset


class Duplicates:
    """Finds duplicate images using non-cryptographic and perceptual hashing.

    Detects both exact duplicates (identical pixel values) using xxhash
    non-cryptographic hashing and near duplicates (visually similar) using
    perceptual hashing with discrete cosine transform. Supports analysis
    of single datasets or cross-dataset duplicate detection.

    Perceptual hashing identifies visually similar images that may differ
    in compression, resolution, or minor modifications while maintaining
    the same visual content structure.

    Parameters
    ----------
    only_exact : bool, default False
        Whether to detect only exact pixel-level duplicates using xxhash.
        When True, skips near duplicate computation for faster processing
        and lower memory usage. When False, detects both exact and near
        duplicates. Default False provides comprehensive duplicate detection.

    Attributes
    ----------
    stats : HashStatsOutput
        Hash statistics computed during the last evaluate() call.
        Contains xxhash and pchash values for all processed images.
    only_exact : bool
        Configuration for duplicate detection scope.

    Examples
    --------

    End-to-end detection: compute hashes + find duplicates

    >>> detector = Duplicates()
    >>> result = detector.evaluate(dataset)

    Reuse pre-computed hashes for efficiency

    >>> result = detector.from_stats(hashes1)

    Fast exact-only detection for large datasets

    >>> fast_detector = Duplicates(only_exact=True)
    >>> result = fast_detector.evaluate(duplicate_images)

    """

    def __init__(self, only_exact: bool = False) -> None:
        self.stats: HashStatsOutput
        self.only_exact = only_exact

    def _get_duplicates(self, stats: dict) -> dict[str, list[list[int]]]:
        """Extract duplicate groups from hash statistics."""
        exact_dict: dict[int, list] = {}
        for i, value in enumerate(stats["xxhash"]):
            exact_dict.setdefault(value, []).append(i)
        exact = [sorted(v) for v in exact_dict.values() if len(v) > 1]

        if not self.only_exact:
            near_dict: dict[int, list] = {}
            for i, value in enumerate(stats["pchash"]):
                if value:
                    near_dict.setdefault(value, []).append(i)
            near = [sorted(v) for v in near_dict.values() if len(v) > 1 and not any(set(v).issubset(x) for x in exact)]
        else:
            near = []

        return {
            "exact": sorted(exact),
            "near": sorted(near),
        }

    @overload
    def from_stats(self, hashes: HashStatsOutput) -> DuplicatesOutput[DuplicateGroup]: ...

    @overload
    def from_stats(self, hashes: Sequence[HashStatsOutput]) -> DuplicatesOutput[DatasetDuplicateGroupMap]: ...

    @set_metadata(state=["only_exact"])
    def from_stats(
        self, hashes: HashStatsOutput | Sequence[HashStatsOutput]
    ) -> DuplicatesOutput[DuplicateGroup] | DuplicatesOutput[DatasetDuplicateGroupMap]:
        """Find duplicates from pre-computed hash statistics.

        Analyzes previously computed hash values to identify duplicate groups
        without recomputing hashes. Supports both single dataset analysis and
        cross-dataset duplicate detection across multiple hash outputs.

        Parameters
        ----------
        hashes : HashStatsOutput or Sequence[HashStatsOutput]
            Hash statistics from hashstats function. Single HashStatsOutput
            for within-dataset duplicates, or sequence for cross-dataset analysis.

        Returns
        -------
        DuplicatesOutput[DuplicateGroup]
            When single HashStatsOutput provided. Contains exact and near
            duplicate groups as lists of image indices within the dataset.
        DuplicatesOutput[DatasetDuplicateGroupMap]
            When sequence provided. Groups map dataset indices to lists of
            image indices, enabling cross-dataset duplicate identification.

        Raises
        ------
        TypeError
            If hashes is not HashStatsOutput or Sequence[HashStatsOutput].

        Examples
        --------
        Single dataset duplicate detection:

        >>> detector = Duplicates()
        >>> result = detector.from_stats(hashes1)
        >>> print(f"Exact duplicates: {result.exact}")
        Exact duplicates: [[3, 20]]

        >>> print(f"Near duplicates: {result.near}")
        Near duplicates: [[3, 20, 22], [12, 18]]

        Cross-dataset duplicate detection:

        >>> result = detector.from_stats([hashes1, hashes2])
        >>> print(f"Exact duplicates: {result.exact}")
        Exact duplicates: [{0: [3, 20]}, {0: [16], 1: [12]}]

        """

        if isinstance(hashes, HashStatsOutput):
            return DuplicatesOutput(**self._get_duplicates(hashes.data()))

        if not isinstance(hashes, Sequence):
            raise TypeError("Invalid stats output type; only use output from hashstats.")

        combined, dataset_steps = combine_stats(hashes)
        duplicates = self._get_duplicates(combined.data())

        # split up results from combined dataset into individual dataset buckets
        for dup_type, dup_list in duplicates.items():
            dup_list_dict = []
            for idxs in dup_list:
                dup_dict = {}
                for idx in idxs:
                    k, v = get_dataset_step_from_idx(idx, dataset_steps)
                    dup_dict.setdefault(k, []).append(v)
                dup_list_dict.append(dup_dict)
            duplicates[dup_type] = dup_list_dict

        return DuplicatesOutput(**duplicates)

    @set_metadata(state=["only_exact"])
    def from_clusters(
        self,
        cluster_result: ClusterResult,
    ) -> DuplicatesOutput[DuplicateGroup]:
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
        DuplicatesOutput[DuplicateGroup]
            Duplicate detection results with exact and near duplicate groups
            as lists of image indices. Format matches the output of from_stats()
            and evaluate() methods.

        See Also
        --------
        dataeval.core.cluster : Function to compute clusters from embeddings
        from_stats : Find duplicates from pre-computed hash statistics
        evaluate : Find duplicates by computing hashes from images

        Notes
        -----
        This method uses cluster distance standards to identify duplicates:
        - **Exact duplicates**: Points at zero distance in the MST
        - **Near duplicates**: Points within cluster-specific distance thresholds

        Unlike hash-based duplicate detection (from_stats/evaluate), cluster-based
        detection identifies duplicates in embedding space, which can capture
        semantic or visual similarity rather than pixel-level equality.

        The `only_exact` parameter set during initialization controls whether
        near duplicates are computed. Set `only_exact=True` for faster processing
        when only exact duplicates are needed.
        """
        # Find duplicates using MST and cluster assignments
        exact_duplicates, near_duplicates = self._find_duplicates(
            mst=cluster_result["mst"],
            clusters=cluster_result["clusters"],
        )

        return DuplicatesOutput(
            exact=exact_duplicates,
            near=near_duplicates,
        )

    def _find_duplicates(
        self,
        mst: NDArray[np.float32],
        clusters: NDArray[np.intp],
    ) -> tuple[Sequence[Sequence[int]], Sequence[Sequence[int]]]:
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
        near_dupes = self._sorted_union_find(near_indices) if not self.only_exact else []

        return [[int(ii) for ii in il] for il in exact_dupes], [[int(ii) for ii in il] for il in near_dupes]

    def _sorted_union_find(self, index_groups: Any) -> list[list[Any]]:
        """Merges and sorts groups of indices that share any common index"""
        import numpy as np

        # Import disjoint set functions from our cached implementation
        from dataeval.core._fast_hdbscan._disjoint_set import ds_find, ds_rank_create, ds_union_by_rank

        groups: list[list[np.int32]] = [[np.int32(x) for x in range(0)] for y in range(0)]
        uniques, inverse = np.unique(index_groups, return_inverse=True)
        inverse = inverse.flatten()
        disjoint_set = ds_rank_create(uniques.size)
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

    @set_metadata(state=["only_exact"])
    def evaluate(
        self, data: Dataset[ArrayLike] | Dataset[tuple[ArrayLike, Any, Any]]
    ) -> DuplicatesOutput[DuplicateGroup]:
        """Find duplicates by computing hashes and analyzing for duplicate groups.

        Performs end-to-end duplicate detection by computing hash statistics
        for the provided dataset and then identifying duplicate groups.
        Stores computed hash statistics in the stats attribute for reuse.

        Parameters
        ----------
        data : Dataset[ArrayLike] or Dataset[tuple[ArrayLike, Any, Any]]
            Dataset of images in array format. Can be image-only dataset
            or dataset with additional tuple elements (labels, metadata).
            Images should be in standard array format (C, H, W).

        Returns
        -------
        DuplicatesOutput[DuplicateGroup]
            Duplicate detection results with exact and near duplicate groups
            as lists of image indices within the dataset.

        Examples
        --------
        Basic duplicate detection:

        >>> detector = Duplicates()
        >>> result = detector.evaluate(duplicate_images)

        >>> print(f"Exact duplicates: {result.exact}")
        Exact duplicates: [[3, 20], [16, 37]]

        >>> print(f"Near duplicates: {result.near}")
        Near duplicates: [[3, 20, 22], [12, 18], [13, 36], [14, 31], [17, 27], [19, 38, 47]]

        Access computed hashes for reuse

        >>> saved_stats = detector.stats

        """
        images = Images(data) if isinstance(data, Dataset) else data
        self.stats = hashstats(images)
        duplicates = self._get_duplicates(self.stats.data())
        return DuplicatesOutput(**duplicates)
