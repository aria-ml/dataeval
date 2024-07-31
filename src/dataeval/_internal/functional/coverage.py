import math

import numpy as np
from scipy.spatial.distance import pdist, squareform


def naive_radius(crit: np.ndarray, d: int, n: int, k: int):
    rho = (1 / math.sqrt(math.pi)) * ((2 * k * math.gamma(d / 2 + 1)) / (n)) ** (1 / d)
    return np.where(crit > rho)[0]


def adaptive_radius(crit: np.ndarray, percent: np.float64, n: int):
    cutoff = int(n * percent)
    return np.argsort(crit)[::-1][:cutoff]


def sort_neighbors(embeddings: np.ndarray, k: int):
    # Calculate distance matrix, look at the (k+1)th farthest neighbor for each image.

    mat = squareform(pdist(embeddings))
    sorted_dists = np.sort(mat, axis=1)
    crit = sorted_dists[:, k + 1]

    return crit


def coverage(embeddings: np.ndarray, k: int, radius_type: str, percent: np.float64):
    n = len(embeddings)
    if n <= k:
        raise ValueError("Number of observations less than or equal to the specified number of neighbors.")

    crit = sort_neighbors(embeddings=embeddings, k=k)
    d = np.shape(embeddings)[1]
    pvals = naive_radius(crit, d, n, k) if radius_type == "naive" else adaptive_radius(crit, percent, n)

    return pvals, crit
