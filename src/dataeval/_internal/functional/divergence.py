import numpy as np

from .utils import compute_neighbors, minimum_spanning_tree


def divergence_mst(data: np.ndarray, labels: np.ndarray) -> int:
    mst = minimum_spanning_tree(data).toarray()
    edgelist = np.transpose(np.nonzero(mst))
    errors = np.sum(labels[edgelist[:, 0]] != labels[edgelist[:, 1]])
    return errors


def divergence_fnn(data: np.ndarray, labels: np.ndarray) -> int:
    nn_indices = compute_neighbors(data, data)
    errors = np.sum(np.abs(labels[nn_indices] - labels))
    return errors
