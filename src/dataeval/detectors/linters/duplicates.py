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
    """
    Finds the duplicate images in a dataset using xxhash for exact \
    :term:`duplicates<Duplicates>` and pchash for near duplicates.

    Attributes
    ----------
    stats : StatsOutput
        Output class of stats

    Parameters
    ----------
    only_exact : bool, default False
        Only inspect the dataset for exact image matches
    """

    def __init__(self, only_exact: bool = False) -> None:
        self.stats: HashStatsOutput
        self.only_exact = only_exact

    def _get_duplicates(self, stats: dict) -> dict[str, list[list[int]]]:
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
        """
        Returns duplicate image indices for both exact matches and near matches

        Parameters
        ----------
        hashes : HashStatsOutput | Sequence[HashStatsOutput]
            The output(s) from a hashstats analysis

        Returns
        -------
        DuplicatesOutput
            List of groups of indices that are exact and near matches

        See Also
        --------
        hashstats

        Example
        -------
        >>> exact_dupes = Duplicates(only_exact=True)
        >>> exact_dupes.from_stats([hashes1, hashes2])
        DuplicatesOutput(exact=[{0: [3, 20]}, {0: [16], 1: [12]}], near=[])
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
        """
        Returns duplicate image indices for both exact matches and near matches

        Parameters
        ----------
        data : Iterable[ArrayLike], shape - (N, C, H, W) | Dataset[tuple[ArrayLike, Any, Any]]
            A dataset of images in an Array format or the output(s) from a hashstats analysis

        Returns
        -------
        DuplicatesOutput
            List of groups of indices that are exact and near matches

        See Also
        --------
        hashstats

        Example
        -------
        >>> all_dupes = Duplicates()
        >>> all_dupes.evaluate(duplicate_images)
        DuplicatesOutput(exact=[[3, 20], [16, 37]], near=[[3, 20, 22], [12, 18], [13, 36], [14, 31], [17, 27], [19, 38, 47]])
        """  # noqa: E501
        images = Images(data) if isinstance(data, Dataset) else data
        self.stats = hashstats(images)
        duplicates = self._get_duplicates(self.stats.data())
        return DuplicatesOutput(**duplicates)
