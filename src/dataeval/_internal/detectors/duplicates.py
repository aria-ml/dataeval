from typing import Dict, Iterable, List, Literal

import numpy as np

from dataeval._internal.flags import ImageHash
from dataeval._internal.metrics.stats import ImageStats


class Duplicates:
    """
    Finds the duplicate images in a dataset using xxhash for exact duplicates
    and pchash for near duplicates
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

    def evaluate(self, images: Iterable[np.ndarray]) -> Dict[Literal["exact", "near"], List[int]]:
        """
        Returns duplicate image indices for both exact matches and near matches

        Parameters
        ----------
        images : Iterable[np.ndarray]
            A set of images where each individual image is a numpy array in CxHxW format

        Returns
        -------
        Dict[Literal["exact", "near"], List[int]]
            Dictionary of exact and near match indices
        """
        self.stats.reset()
        self.stats.update(images)
        self.results = self.stats.compute()
        return self._get_duplicates()
