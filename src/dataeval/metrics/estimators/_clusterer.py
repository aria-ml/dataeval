from __future__ import annotations

__all__ = []

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dataeval._interop import as_numpy
from dataeval._output import Output
from dataeval.utils._clusterer import compare_links_to_cluster_std, get_clusters, sorted_union_find


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

    clusters: NDArray[np.int_]
    mst: NDArray[np.double]
    linkage_tree: NDArray[np.double]
    condensed_tree: NDArray[np.double]
    membership_strengths: NDArray[np.double]

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

    def find_duplicates(self) -> tuple[list[list[int]], list[list[int]]]:
        """
        Finds duplicate and near duplicate data based on cluster average distance

        Returns
        -------
        Tuple[List[List[int]], List[List[int]]]
            The exact :term:`duplicates<Duplicates>` and near duplicates as lists of related indices
        """
        exact_indices, near_indices = compare_links_to_cluster_std(self.mst, self.clusters)
        exact_dupes = sorted_union_find(exact_indices)
        near_dupes = sorted_union_find(near_indices)

        return [[int(ii) for ii in il] for il in exact_dupes], [[int(ii) for ii in il] for il in near_dupes]


def clusterer(data: ArrayLike) -> ClustererOutput:
    """
    Uses hierarchical clustering on the flattened data and returns clustering
    information.

    Parameters
    ----------
    data : ArrayLike, shape - (N, ...)
        A dataset in an ArrayLike format. Function expects the data to have 2
        or more dimensions which will flatten to (N, P) where N number of
        observations in a P-dimensional space.

    Returns
    -------
    :class:`.ClustererOutput`

    Note
    ----
    The clusterer works best when the length of the feature dimension, P, is
    less than 500. If flattening a CxHxW image results in a dimension larger
    than 500, then it is recommended to reduce the dimensions.

    Example
    -------
    >>> clusterer(clusterer_images).clusters
    array([ 2,  0,  0,  0,  0,  0,  4,  0,  3,  1,  1,  0,  2,  0,  0,  0,  0,
            4,  2,  0,  0,  1,  2,  0,  1,  3,  0,  3,  3,  4,  0,  0,  3,  0,
            3, -1,  0,  0,  2,  4,  3,  4,  0,  1,  0, -1,  3,  0,  0,  0])
    """
    clusters, mst, linkage_tree, condensed_tree, membership_strengths, _, _ = get_clusters(data)

    return ClustererOutput(clusters, mst, linkage_tree, as_numpy(condensed_tree), membership_strengths)
