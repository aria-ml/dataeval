from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

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

    exact: list[list[int]]
    near: list[list[int]]


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
            exact = {}
            for i, value in enumerate(stats_dict["xxhash"]):
                exact.setdefault(value, []).append(i)
            exact = [v for v in exact.values() if len(v) > 1]
        else:
            exact = []

        if "pchash" in stats_dict and not self.only_exact:
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

    @set_metadata("dataeval.detectors", ["only_exact"])
    def evaluate(self, data: Iterable[ArrayLike] | StatsOutput) -> DuplicatesOutput:
        """
        Returns duplicate image indices for both exact matches and near matches

        Parameters
        ----------
        data : Iterable[ArrayLike], shape - (N, C, H, W) | StatsOutput
            A dataset of images in an ArrayLike format or the output from an imagestats metric analysis

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
        if isinstance(data, StatsOutput):
            if not data.xxhash:
                raise ValueError("StatsOutput must include xxhash information of the images.")
            if not self.only_exact and not data.pchash:
                raise ValueError("StatsOutput must include pchash information of the images for near matches.")
            self.stats = data
        else:
            self.stats = imagestats(data, ImageStat.XXHASH | (ImageStat(0) if self.only_exact else ImageStat.PCHASH))
        return DuplicatesOutput(**self._get_duplicates())
