from abc import ABC
from itertools import combinations
from typing import Any, Dict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

from daml._internal.MetricClasses import BER, Metrics


class MultiClassBER(BER, ABC):
    def __init__(self) -> None:
        super().__init__()

    def _calc_FR_stat(self, X, y):
        labels = np.unique(y)
        lbl1_locs = np.nonzero(y == labels[0])[0]
        lbl2_locs = np.nonzero(y == labels[1])[0]
        if len(labels) != 2:
            raise ValueError("y (labels) must be an array with 2 classes")
        # All features belong on second dimension
        X = X.reshape((X.shape[0], -1))
        # Sparse matrix of pairwise distances between each feature vector
        Xdist = csr_matrix(np.triu(squareform(pdist(X)), 1))
        tree = minimum_spanning_tree(Xdist, overwrite=True).toarray()
        # get all elements containing opposing classes
        lbl1_x = tree[lbl1_locs]
        lbl2_x = tree[lbl2_locs]
        lbl1_lbl2 = lbl1_x[:, lbl2_locs]
        lbl2_lbl1 = lbl2_x[:, lbl1_locs]
        n_diff_edges = np.sum(lbl1_lbl2 != 0) + np.sum(lbl2_lbl1 != 0)
        return n_diff_edges

    def _multiclass_ber(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        classes, counts = np.unique(y, return_counts=True)
        M = len(classes)
        if M < 2:
            raise ValueError("Label vector contains less than 2 classes!")
        if M > 10:
            raise ValueError("Label vector contains more than 10 classes!")
        p = counts / counts.sum()
        ber = 0
        for i, j in combinations(classes, 2):
            p_i, N_i = zip(
                *[(v[0], v[1]) for v in zip(p, counts, classes) if v[2] in [i, j]]
            )
            X_i = X[[k in [i, j] for k in y]]
            y_i = y[[k in [i, j] for k in y]]
            R_i = self._calc_FR_stat(X_i, y_i)
            ber += sum(p_i) * R_i / sum(N_i)

        return ber

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """
        TODO: Add Metric description for documentation.
        https://gitlab.jatic.net/jatic/aria/daml/-/issues/83
        """
        return {
            Metrics.BER: self._multiclass_ber(X, y),
        }
