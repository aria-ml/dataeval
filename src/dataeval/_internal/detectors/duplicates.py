from dataclasses import dataclass
from typing import Iterable, List

from numpy.typing import ArrayLike

from dataeval._internal.metrics.stats import StatsOutput
from dataeval._internal.output import OutputMetadata, set_metadata
from dataeval.flags import ImageStat
from dataeval.metrics import imagestats


@dataclass(frozen=True)
class DuplicatesOutput(OutputMetadata):
    exact: List[int]
    near: List[int]


class Duplicates:
    """
    Finds the duplicate images in a dataset using xxhash for exact duplicates
    and pchash for near duplicates

    Attributes
    ----------
    stats : Dict[str, Any]
        Dictionary with the stored hashes for each image

    Example
    -------
    Initialize the Duplicates class:

    >>> dups = Duplicates()
    """

    def __init__(self, find_exact: bool = True, find_near: bool = True):
        self.stats: StatsOutput
        self.find_exact = find_exact
        self.find_near = find_near

    def _get_duplicates(self) -> dict:
        stats_dict = self.stats.dict()
        exact = {}
        if "xxhash" in stats_dict:
            for i, value in enumerate(stats_dict["xxhash"]):
                exact.setdefault(value, []).append(i)
            exact = [v for v in exact.values() if len(v) > 1]

        near = {}
        if "pchash" in stats_dict:
            for i, value in enumerate(stats_dict["pchash"]):
                near.setdefault(value, []).append(i)
            near = [v for v in near.values() if len(v) > 1 and not any(set(v).issubset(x) for x in exact)]

        return {
            "exact": sorted(exact),
            "near": sorted(near),
        }

    @set_metadata("dataeval.detectors.Duplicates", ["find_exact", "find_near"])
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
        {'exact': [[3, 20], [16, 37]], 'near': [[3, 20, 22], [12, 18], [13, 36], [14, 31], [17, 27], [19, 38, 47]]}
        """
        flag_exact = ImageStat.XXHASH if self.find_exact else ImageStat(0)
        flag_near = ImageStat.PCHASH if self.find_near else ImageStat(0)
        self.stats = imagestats(images, flag_exact | flag_near)
        return DuplicatesOutput(**self._get_duplicates())
