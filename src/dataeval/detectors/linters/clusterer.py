"""
If you add stuff at the top of the page...
Where does it go???
"""

from __future__ import annotations

__all__ = []

from dataclasses import dataclass
from typing import Any

import numpy as np
from fast_hdbscan.cluster_trees import (
    CondensedTree,
    cluster_tree_from_condensed_tree,
    condense_tree,
    extract_eom_clusters,
    get_cluster_label_vector,
    get_point_membership_strength_vector,
    mst_to_linkage_tree,
)
from numpy.typing import ArrayLike, NDArray

from dataeval._interop import to_numpy
from dataeval._output import Output, set_metadata
from dataeval.utils._clusterer import cluster_variance, compare_links_to_std, group_mst_by_clusters, sorted_union_find
from dataeval.utils._fast_mst import calculate_neighbor_distances, minimum_spanning_tree
from dataeval.utils._shared import flatten


@dataclass(frozen=True)
class ClustererOutput(Output):
    """
    Output class for :class:`Clusterer` lint detector.

    Attributes
    ----------
    clusters: NDArray[int]
        Assigned clusters
    outliers : NDArray[int]
        Indices that do not fall within a cluster
    duplicates : list[list[int]]
        Groups of indices that are exact :term:`duplicates<Duplicates>`
    potential_duplicates : list[list[int]]
        Groups of indices which are not exact but closely related data points
    mst :  None or NDArray[int]
        The minimum spanning tree of the data if Clusterer.evaluate(return_trees=True)
    linkage_tree : None or NDArray[float]
        The linkage array of the data if Clusterer.evaluate(return_trees=True)
    condensed_tree : None or NDArray[float]
        The condensed tree of the data if Clusterer.evaluate(return_trees=True)
    """

    clusters: NDArray[np.int_]
    outliers: NDArray[np.int_]
    duplicates: list[list[int]]
    potential_duplicates: list[list[int]]
    mst: None | NDArray[np.double]
    linkage_tree: None | NDArray[np.double]
    condensed_tree: None | NDArray[np.double]


class Clusterer:
    """
    Uses hierarchical clustering to flag dataset properties of interest like outliers \
    and :term:`duplicates<Duplicates>`.

    Parameters
    ----------
    dataset : ArrayLike, shape - (N, P)
        A dataset in an ArrayLike format.
        Function expects the data to have 2 dimensions, N number of observations in a P-dimensional space.

    Warning
    -------
    The Clusterer class is heavily dependent on computational resources, and may fail due to insufficient memory.

    Note
    ----
    The Clusterer works best when the length of the feature dimension, P, is less than 500.
    If flattening a CxHxW image results in a dimension larger than 500,
    then it is recommended to reduce the dimensions.
    """

    def __init__(self, dataset: ArrayLike) -> None:
        # Allows an update to dataset to reset the state rather than instantiate a new class
        self._on_init(dataset)

    def _on_init(self, dataset: ArrayLike):
        self._data: NDArray[Any] = flatten(to_numpy(dataset))
        self._validate_data(self._data)
        self._num_samples = len(self._data)

        # Attributes that may shift to parameters
        min_num = int(self._num_samples * 0.05)
        self._min_cluster_size: int = min(max(5, min_num), 100)
        self._cluster_selection_method = "eom"
        self._cluster_selection_epsilon = 0.0
        self._single_cluster = False

        # Calculated attributes
        max_neighbors = min(25, self._num_samples - 1)
        self._kneighbors, self._kdistances = calculate_neighbor_distances(self._data, max_neighbors)
        self._unsorted_mst: NDArray[np.double] = minimum_spanning_tree(self._data, self._kneighbors, self._kdistances)
        self._mst: NDArray[np.double] = self._unsorted_mst[np.argsort(self._unsorted_mst.T[2])]
        self._linkage_tree: NDArray[np.double] = mst_to_linkage_tree(self._mst)
        self._condensed_tree: CondensedTree = condense_tree(self._linkage_tree, self._min_cluster_size, None)

    @property
    def data(self) -> NDArray[Any]:
        return self._data

    @data.setter
    def data(self, x: ArrayLike) -> None:
        self._on_init(x)

    @classmethod
    def _validate_data(cls, x: NDArray):
        """Checks that the data has the correct size and shape"""
        if x.ndim != 2:
            raise ValueError(
                f"Data should only have 2 dimensions; got {x.ndim}. Data should be flattened before being input"
            )

        samples, features = x.shape  # Due to flatten(), we know shape has a length of 2
        if samples < 2:
            raise ValueError(f"Data should have at least 2 samples; got {samples}")
        if features < 1:
            raise ValueError(f"Samples should have at least 1 feature; got {features}")

    def create_clusters(self) -> NDArray[np.int_]:
        """Generates clusters based on condensed tree"""
        cluster_tree = cluster_tree_from_condensed_tree(self._condensed_tree)

        selected_clusters = extract_eom_clusters(
            self._condensed_tree, cluster_tree, allow_single_cluster=self._single_cluster
        )

        # Uncomment if cluster_selection_method is made a parameter
        # if self._cluster_selection_method != "eom":
        #     selected_clusters = extract_leaves(self._condensed_tree, allow_single_cluster=self._single_cluster)

        # Uncomment if cluster_selection_epsilon is made a parameter
        # if len(selected_clusters) > 1 and self._cluster_selection_epsilon > 0.0:
        #     selected_clusters = cluster_epsilon_search(
        #         selected_clusters,
        #         cluster_tree,
        #         min_persistence=self._cluster_selection_epsilon,
        #     )

        clusters = get_cluster_label_vector(
            self._condensed_tree,
            selected_clusters,
            self._cluster_selection_epsilon,
            n_samples=self._data.shape[0],
        )

        self._membership_strengths = get_point_membership_strength_vector(
            self._condensed_tree, selected_clusters, clusters
        )

        return clusters

    def find_outliers(self, clusters: NDArray[np.int_]) -> NDArray[np.int_]:
        """
        Retrieves Outliers based on when the sample was added to the cluster
        and how far it was from the cluster when it was added

        Parameters
        ----------
        clusters : NDArray[int]
            Assigned clusters

        Returns
        -------
        NDArray[int]
            A numpy array of the outlier indices
        """
        return np.nonzero(clusters == -1)[0]

    def find_duplicates(self, clusters: NDArray[np.int_]) -> tuple[list[list[int]], list[list[int]]]:
        """
        Finds duplicate and near duplicate data based on cluster average distance

        Parameters
        ----------
        clusters : NDArray[int]
            Assigned clusters

        Returns
        -------
        Tuple[List[List[int]], List[List[int]]]
            The exact :term:`duplicates<Duplicates>` and near duplicates as lists of related indices
        """
        mst_clusters = group_mst_by_clusters(self._mst, clusters)

        cluster_std = cluster_variance(clusters, mst_clusters, self._mst)

        # exact_dupes, near_dupes = compare_links_to_std(cluster_std, mst_clusters, self._mst)
        exact_indices, near_indices = compare_links_to_std(cluster_std, mst_clusters, self._mst)
        exact_dupes = sorted_union_find(exact_indices)
        near_dupes = sorted_union_find(near_indices)

        return exact_dupes, near_dupes  # type: ignore

    # TODO: Move data input to evaluate from class
    @set_metadata(state=["data"])
    def evaluate(self, return_trees: bool = False) -> ClustererOutput:
        """Finds and flags indices of the data for Outliers and :term:`duplicates<Duplicates>`

        Returns
        -------
        ClustererOutput
            The Outliers and duplicate indices found in the data

        Example
        -------
        >>> import sklearn.datasets as dsets
        >>> embeddings = dsets.make_blobs(
        ...     n_samples=50, centers=np.array([(-1, -1), (1, 1)]), cluster_std=0.5, random_state=33
        ... )[0]
        >>> embeddings[9] = embeddings[24]
        >>> embeddings[23] = embeddings[48] + 1e-5
        >>> clustering = Clusterer(embeddings)
        >>> clustering.evaluate()
        ClustererOutput(clusters=array([ 2,  0,  0,  0,  0,  0,  4,  0,  3,  1,  1,  0,  2,  0,  0,  0,  0,
                4,  2,  0,  0,  1,  2,  0,  1,  3,  0,  3,  3,  4,  0,  0,  3,  0,
                3, -1,  0,  0,  2,  4,  3,  4,  0,  1,  0, -1,  3,  0,  0,  0]), outliers=array([35, 45]), duplicates=[[9, 24], [23, 48]], potential_duplicates=[[1, 11], [2, 7, 19, 20], [3, 4], [9, 10, 43], [12, 38], [25, 40], [26, 47], [30, 37, 49], [32, 46]], mst=None, linkage_tree=None, condensed_tree=None)
        """  # noqa: E501

        clusters = self.create_clusters()
        outliers = self.find_outliers(clusters)
        duplicates, potential_duplicates = self.find_duplicates(clusters)

        if return_trees:
            return ClustererOutput(
                clusters,
                outliers,
                duplicates,
                potential_duplicates,
                self._mst,
                self._linkage_tree,
                to_numpy(self._condensed_tree),
            )
        return ClustererOutput(clusters, outliers, duplicates, potential_duplicates, None, None, None)
