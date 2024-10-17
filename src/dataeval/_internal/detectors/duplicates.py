from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Iterable, Sequence, TypeVar

from numpy.typing import ArrayLike

from dataeval._internal.detectors.merged_stats import combine_stats, get_dataset_step_from_idx
from dataeval._internal.metrics.stats.hashstats import HashStatsOutput, hashstats
from dataeval._internal.output import OutputMetadata, set_metadata

DuplicateGroup = list[int]
DatasetDuplicateGroupMap = dict[int, DuplicateGroup]
TIndexCollection = TypeVar("TIndexCollection", DuplicateGroup, DatasetDuplicateGroupMap)


@dataclass(frozen=True)
class DuplicatesOutput(Generic[TIndexCollection], OutputMetadata):
    """
    Output class for :class:`Duplicates` lint detector

    Attributes
    ----------
    exact : list[list[int] | dict[int, list[int]]]
        Indices of images that are exact matches
    near: list[list[int] | dict[int, list[int]]]
        Indices of images that are near matches

    - For a single dataset, indices are returned as a list of index groups.
    - For multiple datasets, indices are returned as dictionaries where the key is the
      index of the dataset, and the value is the list index groups from that dataset.
    """

    exact: list[TIndexCollection]
    near: list[TIndexCollection]


class Duplicates:
    """
    Finds the duplicate images in a dataset using xxhash for exact duplicates
    and pchash for near duplicates

    Attributes
    ----------
    stats : StatsOutput
        Output class of stats

    Parameters
    ----------
    only_exact : bool, default False
        Only inspect the dataset for exact image matches

    Example
    -------
    Initialize the Duplicates class:

    >>> all_dupes = Duplicates()
    >>> exact_dupes = Duplicates(only_exact=True)
    """

    def __init__(self, only_exact: bool = False):
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
                near_dict.setdefault(value, []).append(i)
            near = [sorted(v) for v in near_dict.values() if len(v) > 1 and not any(set(v).issubset(x) for x in exact)]
        else:
            near = []

        return {
            "exact": sorted(exact),
            "near": sorted(near),
        }

    @set_metadata("dataeval.detectors", ["only_exact"])
    def from_stats(self, hashes: HashStatsOutput | Sequence[HashStatsOutput]) -> DuplicatesOutput:
        """
        Returns duplicate image indices for both exact matches and near matches

        Parameters
        ----------
        data : HashStatsOutput | Sequence[HashStatsOutput]
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
        >>> exact_dupes.from_stats([hashes1, hashes2])
        DuplicatesOutput(exact=[{0: [3, 20]}, {0: [16], 1: [12]}], near=[])
        """

        if isinstance(hashes, HashStatsOutput):
            return DuplicatesOutput(**self._get_duplicates(hashes.dict()))

        if not isinstance(hashes, Sequence):
            raise TypeError("Invalid stats output type; only use output from hashstats.")

        combined, dataset_steps = combine_stats(hashes)
        duplicates = self._get_duplicates(combined.dict())

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

    @set_metadata("dataeval.detectors", ["only_exact"])
    def evaluate(self, data: Iterable[ArrayLike]) -> DuplicatesOutput:
        """
        Returns duplicate image indices for both exact matches and near matches

        Parameters
        ----------
        data : Iterable[ArrayLike], shape - (N, C, H, W) | StatsOutput | Sequence[StatsOutput]
            A dataset of images in an ArrayLike format or the output(s) from a hashstats analysis

        Returns
        -------
        DuplicatesOutput
            List of groups of indices that are exact and near matches

        See Also
        --------
        hashstats

        Example
        -------
        >>> all_dupes.evaluate(images)
        DuplicatesOutput(exact=[[3, 20], [16, 37]], near=[[3, 20, 22], [12, 18], [13, 36], [14, 31], [17, 27], [19, 38, 47]])
        """  # noqa: E501
        self.stats = hashstats(data)
        duplicates = self._get_duplicates(self.stats.dict())
        return DuplicatesOutput(**duplicates)
