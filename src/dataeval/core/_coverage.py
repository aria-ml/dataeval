from __future__ import annotations

__all__ = []

import math

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform

from dataeval.typing import Array
from dataeval.utils._array import ensure_embeddings, flatten


def _validate_inputs(embeddings: Array, num_observations: int) -> Array:
    embeddings = ensure_embeddings(embeddings, dtype=np.float64, unit_interval=True)
    if len(embeddings) <= num_observations:
        raise ValueError(
            f"Length of embeddings ({len(embeddings)}) is less than or equal to the specified number of \
                observations ({num_observations})."
        )
    return embeddings


def _calculate_critical_value_radii(embeddings: Array, num_observations: int) -> NDArray[np.float64]:
    embeddings_matrix = squareform(pdist(flatten(embeddings))).astype(np.float64)
    sorted_dists = np.sort(embeddings_matrix, axis=1)
    return sorted_dists[:, num_observations]


def coverage_naive(
    embeddings: Array,
    num_observations: int,
) -> tuple[NDArray[np.intp], NDArray[np.float64], float]:
    embeddings = _validate_inputs(embeddings, num_observations)
    critical_value_radii = _calculate_critical_value_radii(embeddings, num_observations)

    # Calculate distance matrix, look at the (num_observations + 1)th farthest neighbor for each image.
    coverage_radius = float(
        (1 / math.sqrt(math.pi))
        * ((2 * num_observations * math.gamma(embeddings.shape[1] / 2 + 1)) / (len(embeddings)))
        ** (1 / embeddings.shape[1])
    )
    uncovered_indices = np.where(critical_value_radii > coverage_radius)[0]
    return uncovered_indices, critical_value_radii, coverage_radius


def coverage_adaptive(
    embeddings: Array,
    num_observations: int,
    percent: float,
) -> tuple[NDArray[np.intp], NDArray[np.float64], float]:
    embeddings = _validate_inputs(embeddings, num_observations)
    critical_value_radii = _calculate_critical_value_radii(embeddings, num_observations)

    # Use data adaptive cutoff as coverage_radius
    selection = int(max(len(embeddings) * percent, 1))
    uncovered_indices = np.argsort(critical_value_radii)[::-1][:selection]
    coverage_radius = float(np.mean(np.sort(critical_value_radii)[::-1][selection - 1 : selection + 1]))

    return uncovered_indices, critical_value_radii, coverage_radius
