"""
This module contains the implementation of Dp Divergence
using the First Nearest Neighbor and Minimum Spanning Tree algorithms
"""

from abc import ABC
from typing import Any, Dict, Literal

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors

from daml._internal.MetricClasses import Divergence, Metrics


class DpDivergence(Divergence, ABC):
    def __init__(self) -> None:
        """Constructor method"""

        super().__init__()

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
        k : int
            The number of neighbors to find
        algorithm : str
            Tree method for nearest neighbor (ball_tree or kd_tree)

        .. note::
            Do not use kd_tree if n_features > 20

        Returns
        -------
        List:
            Closest points to each point in A and B

        See Also
        --------
        :func:`sklearn.neighbors.NearestNeighbors`
        """

        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm=algorithm).fit(B)
        nns = nbrs.kneighbors(A)[1]
        nns = nns[:, 1]

        return nns

    def evaluate(
        self,
        dataset_a: np.ndarray,
        dataset_b: np.ndarray,
        algorithm: str = Metrics.Algorithm.MinimumSpanningTree,
    ) -> Dict[str, Any]:
        """
        Returns the divergence between two datasets

        Parameters
        ----------
        dataset_a : np.ndarray
        dataset_b : np.ndarray
        algorithm : str, default "mst"

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the dp divergence and the errors during calculation

        Raises
        ------
        ValueError
            If unsupported method is given

        .. note::
            - A and B must be the same number of dimensions
            - Only algorithm "mst" is supported at the current time

        .. warning::
            WARNING!!!
            MST is very slow in this implementation, this is unlike matlab where
            they have comparable speeds
            Overall, MST takes ~25x LONGER!!
            Source of slowdown:
            conversion to and from CSR format adds ~10% of the time diff between
            1nn and scipy mst function the remaining 90%

        .. todo::
            - validate the input algorithm
            - improve speed for MST, requires a fast mst implementation
            mst is at least 10x slower than knn approach
        """

        data = np.vstack((dataset_a, dataset_b))
        N = dataset_a.shape[0]
        M = dataset_b.shape[0]
        labels = np.vstack([np.zeros([N, 1]), np.ones([M, 1])])
        results = dict()
        Dp = None

        if algorithm == Metrics.Algorithm.FirstNearestNeighbor:
            nn_indices = self._compute_neighbors(data, data)
            # import pdb
            # pdb.set_trace()
            errors = np.sum(np.abs(labels[nn_indices] - labels))
            # print('Errors '+str(errors))
            Dp = 1 - ((M + N) / (2 * M * N)) * errors

        elif algorithm == Metrics.Algorithm.MinimumSpanningTree:
            dense_eudist = squareform(pdist(data))
            eudist_csr = csr_matrix(dense_eudist)
            mst = minimum_spanning_tree(eudist_csr)
            mst = mst.toarray()
            edgelist = np.transpose(np.nonzero(mst))

            errors = np.sum(labels[edgelist[:, 0]] != labels[edgelist[:, 1]])

            Dp = 1 - ((M + N) / (2 * M * N)) * errors
        else:
            raise ValueError(
                f"""
                Unsupported algorithm detected!
                Valid Options: {
                    [
                        Metrics.Algorithm.FirstNearestNeighbor,
                        Metrics.Algorithm.MinimumSpanningTree
                    ]
                }
                """
            )
        results.update(
            {
                Metrics.Method.DpDivergence: Dp,
                "Error": errors,
            },
        )
        return results
