"""
This module contains the implementation of Dp Divergence
using the First Nearest Neighbor and Minimum Spanning Tree algorithms
"""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors

from daml._internal.metrics.base import Metric
from daml._internal.metrics.outputs import DivergenceOutput


class _DpDivergence(Metric, ABC):
    """
    For more information about this divergence, its formal definition,
    and its associated estimators
    see https://arxiv.org/abs/1412.6534.
    """

    def __init__(self) -> None:
        """Constructor method"""
        super().__init__()

    @abstractmethod
    def calculate_errors(self, data: np.ndarray, labels: np.ndarray) -> int:
        """"""
        raise NotImplementedError

    def evaluate(
        self,
        dataset_a: np.ndarray,
        dataset_b: np.ndarray,
    ) -> DivergenceOutput:
        """
        Calculates the divergence and any errors between two datasets

        Parameters
        ----------
        dataset_a, dataset_b : np.ndarray
            Datasets to calculate the divergence between

        Returns
        -------
        DivergenceOutput
            Dataclass containing the dp divergence and errors during calculation

        Note
        ----
        A and B must be 2 dimensions, and equivalent in size on the second dimension
        """
        data = np.vstack((dataset_a, dataset_b))
        N = dataset_a.shape[0]
        M = dataset_b.shape[0]
        labels = np.vstack([np.zeros([N, 1]), np.ones([M, 1])])
        errors = self.calculate_errors(data, labels)
        dp = 1 - ((M + N) / (2 * M * N)) * errors
        return DivergenceOutput(dpdivergence=dp, error=errors)


class DpDivergenceMST(_DpDivergence):
    """
    A minimum spanning tree implementation of dp divergence

    Warning
    -------
        MST is very slow in this implementation, this is unlike matlab where
        they have comparable speeds
        Overall, MST takes ~25x LONGER!!
        Source of slowdown:
        conversion to and from CSR format adds ~10% of the time diff between
        1nn and scipy mst function the remaining 90%
    """

    # TODO
    # - validate the input algorithm
    # - improve speed for MST, requires a fast mst implementation
    # mst is at least 10x slower than knn approach

    def calculate_errors(self, data: np.ndarray, labels: np.ndarray) -> int:
        """
        Returns the divergence between two datasets using a minimum spanning tree

        Parameters
        ----------
        data : np.ndarray
            Array containing images from two datasets
        labels : np.ndarray
            Array showing which dataset each image belonged to

        Returns
        -------
        int
            Number of edges connecting the two datasets

        Note
        ----
        We add a small constant to the distance matrix to ensure scipy interprets
        the input graph as fully-connected.
        """
        dense_eudist = squareform(pdist(data)) + 1e-4
        eudist_csr = csr_matrix(dense_eudist)
        mst = minimum_spanning_tree(eudist_csr)
        mst = mst.toarray()
        edgelist = np.transpose(np.nonzero(mst))
        errors = np.sum(labels[edgelist[:, 0]] != labels[edgelist[:, 1]])
        return errors


class DpDivergenceFNN(_DpDivergence):
    """
    A first nearest neighbor implementation of dp divergence
    """

    def _compute_neighbors(
        self,
        A: np.ndarray,
        B: np.ndarray,
        k: int = 1,
        algorithm: Literal["auto", "ball_tree", "kd_tree"] = "auto",
    ) -> np.ndarray:
        """
        For each sample in A, compute the nearest neighbor in B

        Parameters
        ----------
        A, B : np.ndarray
            The n_samples and n_features respectively
        k : int, default 1
            The number of neighbors to find
        algorithm : Literal, default auto
            Tree method for nearest neighbor (auto, ball_tree or kd_tree)

        Note
        ----
        Do not use kd_tree if n_features > 20

        Returns
        -------
        np.ndarray
            Closest point in B to each point in A

        See Also
        --------
        sklearn.neighbors.NearestNeighbors
        """

        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm=algorithm).fit(B)
        nns = nbrs.kneighbors(A)[1]
        nns = nns[:, 1]

        return nns

    def calculate_errors(self, data: np.ndarray, labels: np.ndarray) -> int:
        """
        Returns the divergence between two datasets using first nearest neighbor

        Parameters
        ----------
        data : np.ndarray
            Array containing images from two datasets
        labels : np.ndarray
            Array showing which dataset each image belonged to

        Returns
        -------
        int
            Number of edges connecting the two datasets
        """
        nn_indices = self._compute_neighbors(data, data)
        errors = np.sum(np.abs(labels[nn_indices] - labels))
        return errors
