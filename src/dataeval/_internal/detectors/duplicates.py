from typing import Dict, Iterable, List, Literal

from dataeval._internal.flags import ImageHash
from dataeval._internal.interop import ArrayLike
from dataeval._internal.metrics.stats import ImageStats


class Duplicates:
    """
    Finds the duplicate images in a dataset using xxhash for exact duplicates
    and pchash for near duplicates

    Attributes
    ----------
    stats : ImageStats(flags=ImageHash.ALL)
        Base stats class with the flags for checking duplicates

    Example
    -------
    Initialize the Duplicates class:

    >>> dups = Duplicates()
    """

    def __init__(self):
        self.stats = ImageStats(ImageHash.ALL)

    def _get_duplicates(self) -> dict:
        exact = {}
        near = {}
        for i, value in enumerate(self.results["xxhash"]):
            exact.setdefault(value, []).append(i)
        for i, value in enumerate(self.results["pchash"]):
            near.setdefault(value, []).append(i)
        exact = [v for v in exact.values() if len(v) > 1]
        near = [v for v in near.values() if len(v) > 1 and not any(set(v).issubset(x) for x in exact)]

        return {
            "exact": sorted(exact),
            "near": sorted(near),
        }

    def evaluate(self, images: Iterable[ArrayLike]) -> Dict[Literal["exact", "near"], List[int]]:
        """
        Returns duplicate image indices for both exact matches and near matches

        Parameters
        ----------
        images : Iterable[ArrayLike], shape - (N, C, H, W)
            A set of images in an ArrayLike format

        Returns
        -------
        Dict[str, List[int]]
            exact :
                List of groups of indices that are exact matches
            near :
                List of groups of indices that are near matches

        See Also
        --------
        ImageStats

        Example
        -------
        >>> dups.evaluate(images)
        {'exact': [[3, 20], [16, 37]], 'near': [[3, 20, 22], [12, 18], [13, 36], [14, 31], [17, 27], [19, 38, 47]]}
        """
        self.stats.reset()
        self.stats.update(images)
        self.results = self.stats.compute()
        return self._get_duplicates()
