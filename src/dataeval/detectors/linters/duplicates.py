from __future__ import annotations

__all__ = []

from collections.abc import Sequence
from typing import Any, overload

from dataeval.data._images import Images
from dataeval.metrics.stats import hashstats
from dataeval.metrics.stats._base import combine_stats, get_dataset_step_from_idx
from dataeval.outputs import DuplicatesOutput, HashStatsOutput
from dataeval.outputs._base import set_metadata
from dataeval.outputs._linters import DatasetDuplicateGroupMap, DuplicateGroup
from dataeval.typing import ArrayLike, Dataset


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
