from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Iterable, Sequence, TypeVar, cast

from numpy.typing import ArrayLike

from dataeval._internal.detectors.merged_stats import combine_stats, get_dataset_step_from_idx
from dataeval._internal.flags import ImageStat
from dataeval._internal.metrics.stats import StatsOutput, imagestats
from dataeval._internal.output import OutputMetadata, set_metadata

DuplicateGroup = list[int]
DatasetDuplicateGroupMap = dict[int, DuplicateGroup]
TIndexCollection = TypeVar("TIndexCollection", DuplicateGroup, DatasetDuplicateGroupMap)


@dataclass(frozen=True)
class DuplicatesOutput(Generic[TIndexCollection], OutputMetadata):
    """
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

    >>> dups = Duplicates()
    """

    def __init__(self, only_exact: bool = False):
        self.stats: StatsOutput
        self.only_exact = only_exact

    def _get_duplicates(self) -> dict[str, list[list[int]]]:
        stats_dict = self.stats.dict()
        if "xxhash" in stats_dict:
            exact_dict: dict[int, list] = {}
            for i, value in enumerate(stats_dict["xxhash"]):
                exact_dict.setdefault(value, []).append(i)
            exact = [sorted(v) for v in exact_dict.values() if len(v) > 1]
        else:
            exact = []

        if "pchash" in stats_dict and not self.only_exact:
            near_dict: dict[int, list] = {}
            for i, value in enumerate(stats_dict["pchash"]):
                near_dict.setdefault(value, []).append(i)
            near = [sorted(v) for v in near_dict.values() if len(v) > 1 and not any(set(v).issubset(x) for x in exact)]
        else:
            near = []

        return {
            "exact": sorted(exact),
            "near": sorted(near),
        }

    @set_metadata("dataeval.detectors", ["only_exact"])
    def evaluate(self, data: Iterable[ArrayLike] | StatsOutput | Sequence[StatsOutput]) -> DuplicatesOutput:
        """
        Returns duplicate image indices for both exact matches and near matches

        Parameters
        ----------
        data : Iterable[ArrayLike], shape - (N, C, H, W) | StatsOutput | Sequence[StatsOutput]
            A dataset of images in an ArrayLike format or the output(s) from an imagestats metric analysis

        Returns
        -------
        DuplicatesOutput
            List of groups of indices that are exact and near matches

        See Also
        --------
        imagestats

        Example
        -------
        >>> dups.evaluate(images)
        DuplicatesOutput(exact=[[3, 20], [16, 37]], near=[[3, 20, 22], [12, 18], [13, 36], [14, 31], [17, 27], [19, 38, 47]])
        """  # noqa: E501

        stats, dataset_steps = combine_stats(data)

        if isinstance(stats, StatsOutput):
            if not stats.xxhash:
                raise ValueError("StatsOutput must include xxhash information of the images.")
            if not self.only_exact and not stats.pchash:
                raise ValueError("StatsOutput must include pchash information of the images for near matches.")
            self.stats = stats
        else:
            flags = ImageStat.XXHASH | (ImageStat(0) if self.only_exact else ImageStat.PCHASH)
            self.stats = imagestats(cast(Iterable[ArrayLike], data), flags)

        duplicates = self._get_duplicates()

        # split up results from combined dataset into individual dataset buckets
        if dataset_steps:
            dup_list: list[list[int]]
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
