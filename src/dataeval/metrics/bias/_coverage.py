from __future__ import annotations

__all__ = []

import math
from typing import Literal

import numpy as np
from scipy.spatial.distance import pdist, squareform

from dataeval.outputs import CoverageOutput
from dataeval.outputs._base import set_metadata
from dataeval.typing import Array
from dataeval.utils._array import ensure_embeddings, flatten


@set_metadata
def coverage(
    embeddings: Array,
    radius_type: Literal["adaptive", "naive"] = "adaptive",
    num_observations: int = 20,
    percent: float = 0.01,
) -> CoverageOutput:
    """
    Class for evaluating :term:`coverage<Coverage>` and identifying images/samples that are in undercovered regions.

    Parameters
    ----------
    embeddings : ArrayLike, shape - (N, P)
        Dataset embeddings as unit interval [0, 1].
        Function expects the data to have 2 dimensions, N number of observations in a P-dimensional space.
    radius_type : {"adaptive", "naive"}, default "adaptive"
        The function used to determine radius.
    num_observations : int, default 20
        Number of observations required in order to be covered.
        [1] suggests that a minimum of 20-50 samples is necessary.
    percent : float, default 0.01
        Percent of observations to be considered uncovered. Only applies to adaptive radius.

    Returns
    -------
    CoverageOutput
        Array of uncovered indices, critical value radii, and the radius for coverage

    Raises
    ------
    ValueError
        If embeddings are not unit interval [0-1]
    ValueError
        If length of :term:`embeddings<Embeddings>` is less than or equal to num_observations
    ValueError
        If radius_type is unknown

    Note
    ----
    Embeddings should be on the unit interval [0-1].

    Example
    -------
    >>> results = coverage(embeddings)
    >>> results.uncovered_indices
    array([447, 412,   8,  32,  63])
    >>> results.coverage_radius
    0.17592147193757596

    Reference
    ---------
    This implementation is based on https://dl.acm.org/doi/abs/10.1145/3448016.3457315.

    [1] Seymour Sudman. 1976. Applied sampling. Academic Press New York (1976).
    """

    # Calculate distance matrix, look at the (num_observations + 1)th farthest neighbor for each image.
    embeddings = ensure_embeddings(embeddings, dtype=np.float64, unit_interval=True)
    len_embeddings = len(embeddings)
    if len_embeddings <= num_observations:
        raise ValueError(
            f"Length of embeddings ({len_embeddings}) is less than or equal to the specified number of \
                observations ({num_observations})."
        )
    embeddings_matrix = squareform(pdist(flatten(embeddings))).astype(np.float64)
    sorted_dists = np.sort(embeddings_matrix, axis=1)
    critical_value_radii = sorted_dists[:, num_observations + 1]

    d = embeddings.shape[1]
    if radius_type == "naive":
        coverage_radius = (1 / math.sqrt(math.pi)) * (
            (2 * num_observations * math.gamma(d / 2 + 1)) / (len_embeddings)
        ) ** (1 / d)
        uncovered_indices = np.where(critical_value_radii > coverage_radius)[0]
    elif radius_type == "adaptive":
        # Use data adaptive cutoff as coverage_radius
        selection = int(max(len_embeddings * percent, 1))
        uncovered_indices = np.argsort(critical_value_radii)[::-1][:selection]
        coverage_radius = float(np.mean(np.sort(critical_value_radii)[::-1][selection - 1 : selection + 1]))
    else:
        raise ValueError(f"{radius_type} is an invalid radius type. Expected 'adaptive' or 'naive'")
    return CoverageOutput(uncovered_indices, critical_value_radii, coverage_radius)
