from __future__ import annotations

__all__ = ["ClustererOutput", "Clusterer"]

from dataclasses import dataclass
from typing import Any, Iterable, NamedTuple, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from fast_hdbscan.cluster_trees import (
    cluster_epsilon_search,
    cluster_tree_from_condensed_tree,
    condense_tree,
    extract_eom_clusters,
    extract_leaves,
    get_cluster_label_vector,
    get_point_membership_strength_vector,
    mst_to_linkage_tree,
    CondensedTree,
)

from dataeval.interop import to_numpy
from dataeval.output import OutputMetadata, set_metadata
from dataeval.utils.shared import flatten
from fast_mst import calculate_neighbor_distances, minimum_spanning_tree
from fast_cluster_utils import _cluster_variance, _compare_links_to_std, _group_mst_by_clusters


@dataclass(frozen=True)
class ClustererOutput(OutputMetadata):
    """
    Output class for :class:`Clusterer` lint detector

    Attributes
    ----------
    clusters: NDArray[int]
        Assigned clusters
    outliers : NDArray[int]
        Indices that do not fall within a cluster
    duplicates : NDArray[int]
        Groups of indices that are exact :term:`duplicates<Duplicates>`
    potential_duplicates : NDArray[int]
        Groups of indices which are not exact but closely related data points
    """

    clusters: NDArray[np.integer[Any]]
    outliers: NDArray[np.integer[Any]]
    duplicates: NDArray[np.integer[Any]]
    potential_duplicates: NDArray[np.integer[Any]]

class Clusterer:
    """
    Uses hierarchical clustering to flag dataset properties of interest like Outliers and :term:`duplicates<Duplicates>`

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
    If flattening a CxHxW image results in a dimension larger than 500, then it is recommended to reduce the dimensions.

    Example
    -------
    Initialize the Clusterer class:

    >>> cluster = Clusterer(dataset)
    """

    def __init__(self, dataset: ArrayLike) -> None:
        # Allows an update to dataset to reset the state rather than instantiate a new class
        self._on_init(dataset)

    def _on_init(self, dataset: ArrayLike):
        self._data: NDArray[Any] = flatten(to_numpy(dataset))
        self._num_samples = len(self._data)

        # Attributes that may shift to parameters
        min_num = int(self._num_samples * 0.05)
        self._min_cluster_size: int = min(max(2, min_num), 100)
        self._cluster_selection_method = "eom"
        self._cluster_selection_epsilon = 0.0
        self._return_trees=False
        
        # Calculated attributes
        self._kneighbors, self._kdistances = calculate_neighbor_distances(self._data, 20)
        self._unsorted_mst: NDArray[np.floating[Any]] = minimum_spanning_tree(self._data, self._kneighbors, self._kdistances)
        self._mst = self._unsorted_mst[np.argsort(self._unsorted_mst.T[2])]
        self._linkage_tree: NDArray[np.floating[Any]] = mst_to_linkage_tree(self._mst)
        self._condensed_tree: CondensedTree = condense_tree(self._linkage_tree, self._min_cluster_size, None)

    @property
    def data(self) -> NDArray[Any]:
        return self._data

    @data.setter
    def data(self, x: ArrayLike) -> None:
        self._on_init(x)

    def create_clusters(self) -> NDArray[np.integer[Any]]:
        """Generates clusters based on condensed tree"""
        cluster_tree = cluster_tree_from_condensed_tree(self._condensed_tree)
        
        if self._cluster_selection_method == 'eom':
            selected_clusters = extract_eom_clusters(
                self._condensed_tree, cluster_tree, allow_single_cluster=False
            )
        else:
            selected_clusters = extract_leaves(
                self._condensed_tree, allow_single_cluster=False
            )

        if len(selected_clusters) > 1 and self._cluster_selection_epsilon > 0.0:
            selected_clusters = cluster_epsilon_search(
                selected_clusters,
                cluster_tree,
                min_persistence=self._cluster_selection_epsilon,
            )

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
    
    def get_outliers(self, clusters) -> NDArray[np.integer[Any]]:
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
        return np.nonzero(clusters==-1)[0]
    
    def find_duplicates(self, clusters) -> tuple[NDArray[np.integer[Any]], NDArray[np.integer[Any]]]:
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
        mst_clusters = _group_mst_by_clusters(self._mst, clusters)
        
        cluster_std = _cluster_variance(clusters, mst_clusters, self._mst)

        exact_dupes, near_dupes = _compare_links_to_std(cluster_std, mst_clusters, self._mst)

        return exact_dupes, near_dupes

    @set_metadata(["data"])
    def evaluate(self) -> ClustererOutput:
        """Finds and flags indices of the data for Outliers and :term:`duplicates<Duplicates>`

        Returns
        -------
        ClustererOutput
            The Outliers and duplicate indices found in the data

        Example
        -------
        >>> cluster.evaluate()
        ClustererOutput(outliers=[18, 21, 34, 35, 45], potential_outliers=[13, 15, 42], duplicates=[[9, 24], [23, 48]], potential_duplicates=[[1, 11]])
        """  # noqa: E501

        clusters = self.create_clusters()
        outliers = self.get_outliers(clusters)
        duplicates, potential_duplicates = self.find_duplicates(clusters)

        return ClustererOutput(clusters, outliers, duplicates, potential_duplicates)
