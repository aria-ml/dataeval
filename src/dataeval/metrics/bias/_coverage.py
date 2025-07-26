from __future__ import annotations

__all__ = []

from typing import Literal

from dataeval.core._coverage import coverage_adaptive, coverage_naive
from dataeval.outputs import CoverageOutput
from dataeval.outputs._base import set_metadata
from dataeval.typing import Array


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
    array([447, 412,   8,  32,  13])
    >>> results.coverage_radius
    0.1703530195830698

    Reference
    ---------
    This implementation is based on https://dl.acm.org/doi/abs/10.1145/3448016.3457315.

    [1] Seymour Sudman. 1976. Applied sampling. Academic Press New York (1976).
    """

    # Calculate distance matrix, look at the (num_observations + 1)th farthest neighbor for each image.
    if radius_type == "naive":
        return CoverageOutput(*coverage_naive(embeddings, num_observations))
    if radius_type == "adaptive":
        return CoverageOutput(*coverage_adaptive(embeddings, num_observations, percent))

    raise ValueError(f"{radius_type} is an invalid radius type. Expected 'adaptive' or 'naive'")
