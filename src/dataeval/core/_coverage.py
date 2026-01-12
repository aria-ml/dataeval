__all__ = []

import logging
import math
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform

from dataeval.types import Array2D
from dataeval.utils.arrays import as_numpy, ensure_embeddings, flatten_samples

_logger = logging.getLogger(__name__)


class CoverageResult(TypedDict):
    """
    Type definition for coverage output.

    Attributes
    ----------
    uncovered_indices : NDArray[np.intp]
        Array of indices for uncovered observations
    critical_value_radii : NDArray[np.float64]
        Array of critical value radii for each observation
    coverage_radius : float
        The radius threshold for coverage
    """

    uncovered_indices: NDArray[np.intp]
    critical_value_radii: NDArray[np.float64]
    coverage_radius: float


def _validate_inputs(embeddings: NDArray[np.float64], num_observations: int) -> NDArray[np.float64]:
    embeddings = ensure_embeddings(embeddings, dtype=np.float64, unit_interval=True)
    if len(embeddings) <= num_observations:
        raise ValueError(
            f"Length of embeddings ({len(embeddings)}) is less than or equal to the specified number of \
                observations ({num_observations})."
        )
    return embeddings


def _calculate_critical_value_radii(embeddings: NDArray[np.float64], num_observations: int) -> NDArray[np.float64]:
    embeddings_matrix = squareform(pdist(flatten_samples(embeddings))).astype(np.float64)
    sorted_dists = np.sort(embeddings_matrix, axis=1)
    return sorted_dists[:, num_observations]


def coverage_naive(
    embeddings: Array2D[float],
    num_observations: int,
) -> CoverageResult:
    """
    Evaluate :term:`coverage<Coverage>` using a naive radius calculation method.

    This method calculates a fixed coverage radius based on the dimensionality of the
    embedding space and the desired number of observations per covered region.

    Parameters
    ----------
    embeddings : Array2D[float]
        Dataset image embeddings as unit interval [0, 1]. Can be a 2D list, array-like
        object, or tensor. Function expects the data to have 2 dimensions, N number of
        observations in a P-dimensional space.
    num_observations : int
        Number of observations required in order to be covered.
        [1] suggests that a minimum of 20-50 samples is necessary.

    Returns
    -------
    CoverageResult
        Mapping with keys:

        - uncovered_indices: NDArray[np.intp] - Array of indices for uncovered observations
        - critical_value_radii: NDArray[np.float64] - Array of critical value radii for each observation
        - coverage_radius: float - The radius threshold for coverage

    Raises
    ------
    ValueError
        If embeddings are not unit interval [0-1]
    ValueError
        If length of :term:`embeddings<Embeddings>` is less than or equal to num_observations

    Notes
    -----
    Embeddings should be on the unit interval [0-1].

    The naive method calculates a fixed radius based on the formula:
    r = (1/√π) * ((2 * k * Γ(d/2 + 1)) / n)^(1/d)
    where k is num_observations, d is the dimensionality, and n is the number of samples.

    Reference
    ---------
    This implementation is based on https://dl.acm.org/doi/abs/10.1145/3448016.3457315.

    [1] Seymour Sudman. 1976. Applied sampling. Academic Press New York (1976).
    """
    _logger.info("Starting coverage_naive calculation with num_observations=%d", num_observations)

    embeddings_np = _validate_inputs(as_numpy(embeddings, dtype=np.float64, required_ndim=2), num_observations)
    _logger.debug("Embeddings shape: %s", embeddings_np.shape)

    critical_value_radii = _calculate_critical_value_radii(embeddings_np, num_observations)

    # Calculate distance matrix, look at the (num_observations + 1)th farthest neighbor for each image.
    coverage_radius = float(
        (1 / math.sqrt(math.pi))
        * ((2 * num_observations * math.gamma(embeddings_np.shape[1] / 2 + 1)) / (len(embeddings_np)))
        ** (1 / embeddings_np.shape[1])
    )
    uncovered_indices = np.where(critical_value_radii > coverage_radius)[0]

    _logger.info(
        "Coverage_naive complete: radius=%.4f, %d uncovered (%.1f%%)",
        coverage_radius,
        len(uncovered_indices),
        100 * len(uncovered_indices) / len(embeddings_np),
    )

    return {
        "uncovered_indices": uncovered_indices,
        "critical_value_radii": critical_value_radii,
        "coverage_radius": coverage_radius,
    }


def coverage_adaptive(
    embeddings: Array2D[float],
    num_observations: int,
    percent: float,
) -> CoverageResult:
    """
    Evaluate :term:`coverage<Coverage>` using an adaptive radius calculation method.

    This method calculates a data-adaptive coverage radius based on the distribution
    of critical value radii, selecting the top percentage of observations as uncovered.

    Parameters
    ----------
    embeddings : Array2D[float]
        Dataset embeddings as unit interval [0, 1]. Can be a 2D list, array-like object,
        or tensor. Function expects the data to have 2 dimensions, N number of
        observations in a P-dimensional space.
    num_observations : int
        Number of observations required in order to be covered.
        [1] suggests that a minimum of 20-50 samples is necessary.
    percent : float
        Percent of observations to be considered uncovered. Should be between 0 and 1.

    Returns
    -------
    CoverageResult
        Mapping with keys:

        - uncovered_indices: NDArray[np.intp] - Array of indices for uncovered observations
        - critical_value_radii: NDArray[np.float64] - Array of critical value radii for each observation
        - coverage_radius: float - The adaptive radius threshold for coverage

    Raises
    ------
    ValueError
        If embeddings are not unit interval [0-1]
    ValueError
        If length of :term:`embeddings<Embeddings>` is less than or equal to num_observations

    Notes
    -----
    Embeddings should be on the unit interval [0-1].

    The adaptive method determines the coverage radius based on the data distribution,
    selecting the top `percent` of observations with the largest critical value radii
    as uncovered. This approach is more flexible than the naive method and adapts to
    the actual distribution of the data.

    Reference
    ---------
    This implementation is based on https://dl.acm.org/doi/abs/10.1145/3448016.3457315.

    [1] Seymour Sudman. 1976. Applied sampling. Academic Press New York (1976).
    """
    _logger.info(
        "Starting coverage_adaptive calculation with num_observations=%d, percent=%.2f", num_observations, percent
    )

    embeddings = _validate_inputs(as_numpy(embeddings, dtype=np.float64, required_ndim=2), num_observations)
    _logger.debug("Embeddings shape: %s", embeddings.shape)

    critical_value_radii = _calculate_critical_value_radii(embeddings, num_observations)

    # Use data adaptive cutoff as coverage_radius
    selection = int(max(len(embeddings) * percent, 1))
    uncovered_indices = np.argsort(critical_value_radii)[::-1][:selection]
    coverage_radius = float(np.mean(np.sort(critical_value_radii)[::-1][selection - 1 : selection + 1]))

    _logger.info(
        "Coverage_adaptive complete: radius=%.4f, %d uncovered (%.1f%%)",
        coverage_radius,
        len(uncovered_indices),
        100 * len(uncovered_indices) / len(embeddings),
    )

    return {
        "uncovered_indices": uncovered_indices,
        "critical_value_radii": critical_value_radii,
        "coverage_radius": coverage_radius,
    }
