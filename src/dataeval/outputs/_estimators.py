from __future__ import annotations

__all__ = []

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from dataeval.outputs._base import Output


@dataclass(frozen=True)
class BEROutput(Output):
    """
    Output class for :func:`.ber` estimator metric.

    Attributes
    ----------
    ber : float
        The upper bounds of the :term:`Bayes error rate<Bayes Error Rate (BER)>`
    ber_lower : float
        The lower bounds of the Bayes Error Rate
    """

    ber: float
    ber_lower: float


@dataclass(frozen=True)
class ClustererOutput(Output):
    """
    Output class for :func:`.clusterer`.

    Attributes
    ----------
    clusters : NDArray[int]
        Assigned clusters
    mst : NDArray[int]
        The minimum spanning tree of the data
    linkage_tree : NDArray[float]
        The linkage array of the data
    condensed_tree : NDArray[float]
        The condensed tree of the data
    membership_strengths : NDArray[float]
        The strength of the data point belonging to the assigned cluster
    """

    clusters: NDArray[np.intp]
    mst: NDArray[np.float32]
    linkage_tree: NDArray[np.float32]
    condensed_tree: NDArray[np.float32]
    membership_strengths: NDArray[np.float32]

    def find_outliers(self) -> NDArray[np.int_]:
        """
        Retrieves Outliers based on when the sample was added to the cluster
        and how far it was from the cluster when it was added

        Returns
        -------
        NDArray[int]
            A numpy array of the outlier indices
        """
        return np.nonzero(self.clusters == -1)[0]

    def find_duplicates(self) -> tuple[Sequence[Sequence[int]], Sequence[Sequence[int]]]:
        """
        Finds duplicate and near duplicate data based on cluster average distance

        Returns
        -------
        Tuple[List[List[int]], List[List[int]]]
            The exact :term:`duplicates<Duplicates>` and near duplicates as lists of related indices
        """
        # Delay load numba compiled functions
        from dataeval.utils._clusterer import compare_links_to_cluster_std, sorted_union_find

        exact_indices, near_indices = compare_links_to_cluster_std(self.mst, self.clusters)  # type: ignore
        exact_dupes = sorted_union_find(exact_indices)
        near_dupes = sorted_union_find(near_indices)

        return [[int(ii) for ii in il] for il in exact_dupes], [[int(ii) for ii in il] for il in near_dupes]


@dataclass(frozen=True)
class DivergenceOutput(Output):
    """
    Output class for :func:`.divergence` estimator metric.

    Attributes
    ----------
    divergence : float
        :term:`Divergence` value calculated between 2 datasets ranging between 0.0 and 1.0
    errors : int
        The number of differing edges between the datasets
    """

    divergence: float
    errors: int


@dataclass(frozen=True)
class UAPOutput(Output):
    """
    Output class for :func:`.uap` estimator metric.

    Attributes
    ----------
    uap : float
        The empirical mean precision estimate
    """

    uap: float
