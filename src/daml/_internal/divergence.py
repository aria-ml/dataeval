from abc import ABC
from typing import Any, Dict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors

from daml._internal.MetricClasses import Divergence, Metrics


class DpDivergence(Divergence, ABC):
    def __init__(self, method: str) -> None:
        super().__init__()
        self.method = method

    def _compute_neighbors(self, A, B, k=1, algorithm="auto") -> None:
        """
        For each sample in A, compute the nearest neighbor in B
        :inputs:
        A and B - both (n_samples x n_features)
        algorithm - look @ scipy NearestNeighbors nocumentation for this
        (ball_tree or kd_tree)
                    dont use kd_tree if n_features>~20 or 30
        :return:
        a list of the closest points to each point in A and B
        """
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm=algorithm).fit(B)
        nns = nbrs.kneighbors(A)[1]
        nns = nns[:, 1]

        # exit()
        return nns

    def evaluate(
        self,
        dataset_a: np.ndarray,
        dataset_b: np.ndarray,
        algorithm: str = Metrics.Algorithm.MinimumSpanningTree,
    ) -> Dict[str, Any]:
        """
        Requires A and B to be the same number of dimensions
        *******
        WARNING!!!
        MST is very slow in this implementation, this is unlike matlab where
        they have comparable speeds
        Overall, MST takes ~25x LONGER!!
        Source of slowdown:
        conversion to and from CSR format adds ~10% of the time diff between
        1nn and scipy mst function the remaining 90%
        *******
        """
        # TODO: validate the input algorithm.

        data = np.vstack((dataset_a, dataset_b))
        print(type(dataset_a))
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

        # TODO: improve speed for MST, requires a fast mst implementation
        # mst is at least 10x slower than knn approach
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
        # errors=0
        # Cij = errors
        return results
