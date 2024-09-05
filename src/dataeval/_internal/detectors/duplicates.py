from dataclasses import dataclass
from typing import Dict, Iterable, List

from numpy.typing import ArrayLike

from dataeval._internal.metrics.stats import StatsOutput
from dataeval._internal.output import OutputMetadata, set_metadata
from dataeval.flags import ImageStat
from dataeval.metrics import imagestats


@dataclass(frozen=True)
class DuplicatesOutput(OutputMetadata):
    """
    Attributes
    ----------
    exact : List[List[int]]
        Indices of images that are exact matches
    near: List[List[int]]
        Indices of images that are near matches
    """

    exact: List[List[int]]
    near: List[List[int]]


class Duplicates:
    """
    Finds the duplicate images in a dataset using xxhash for exact duplicates
    and pchash for near duplicates

    Attributes
    ----------
    stats : StatsOutput
        Output class of stats

    Example
    -------
    Initialize the Duplicates class:

    >>> dups = Duplicates()
    """

    def __init__(self, find_exact: bool = True, find_near: bool = True):
        self.stats: StatsOutput
        self.find_exact = find_exact
        self.find_near = find_near

    def _get_duplicates(self) -> Dict[str, List[List[int]]]:
        stats_dict = self.stats.dict()
        if "xxhash" in stats_dict:
            exact = {}
            for i, value in enumerate(stats_dict["xxhash"]):
                exact.setdefault(value, []).append(i)
            exact = [v for v in exact.values() if len(v) > 1]
        else:
            exact = []

        if "pchash" in stats_dict:
            near = {}
            for i, value in enumerate(stats_dict["pchash"]):
                near.setdefault(value, []).append(i)
            near = [v for v in near.values() if len(v) > 1 and not any(set(v).issubset(x) for x in exact)]
        else:
            near = []

        return {
            "exact": sorted(exact),
            "near": sorted(near),
        }

    @set_metadata("dataeval.detectors", ["find_exact", "find_near"])
    def evaluate(self, images: Iterable[ArrayLike]) -> DuplicatesOutput:
        """
        Returns duplicate image indices for both exact matches and near matches

        Parameters
        ----------
        images : Iterable[ArrayLike], shape - (N, C, H, W)
            A set of images in an ArrayLike format

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
        flag_exact = ImageStat.XXHASH if self.find_exact else ImageStat(0)
        flag_near = ImageStat.PCHASH if self.find_near else ImageStat(0)
        self.stats = imagestats(images, flag_exact | flag_near)
        return DuplicatesOutput(**self._get_duplicates())
