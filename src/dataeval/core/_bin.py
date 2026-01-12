__all__ = []

import logging
from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import wasserstein_distance

_logger = logging.getLogger(__name__)

DISCRETE_MIN_WD = 0.054
CONTINUOUS_MIN_SAMPLE_SIZE = 20


def get_counts(data: NDArray[np.intp], min_num_bins: int | None = None) -> NDArray[np.intp]:
    """
    Returns columnwise unique counts for discrete data.

    Parameters
    ----------
    data : NDArray
        Array containing integer values for metadata factors
    min_num_bins : int | None, default None
        Minimum number of bins for bincount, helps force consistency across runs

    Returns
    -------
    NDArray[np.int]
        Bin counts per column of data.
    """
    max_value = data.max() + 1 if min_num_bins is None else min_num_bins
    cnt_array = np.zeros((max_value, data.shape[1]), dtype=np.intp)
    for idx in range(data.shape[1]):
        cnt_array[:, idx] = np.bincount(data[:, idx], minlength=max_value)

    return cnt_array


def digitize_data(data: list[Any] | NDArray[Any], bins: int | Iterable[float]) -> NDArray[np.intp]:
    """
    Digitizes a list of values into a given number of bins.

    Parameters
    ----------
    data : list | NDArray
        The values to be digitized.
    bins : int | Iterable[float]
        The number of bins or list of bin edges for the discrete values that data will be digitized into.

    Returns
    -------
    NDArray[np.intp]
        The digitized values
    """

    if not np.all([np.issubdtype(type(n), np.number) for n in data]):
        raise TypeError(
            "Encountered a data value with non-numeric type when digitizing a factor. "
            "Ensure all occurrences of continuous factors are numeric types."
        )
    if isinstance(bins, int):
        _, bin_edges = np.histogram(data, bins=bins)
        bin_edges[-1] = np.inf
        bin_edges[0] = -np.inf
    else:
        bin_edges = list(bins)
    return np.digitize(data, bin_edges)


def bin_data(data: NDArray[Any], bin_method: str) -> NDArray[np.intp]:
    """
    Bins continuous data through either equal width bins, equal amounts in each bin, or by clusters.
    """
    if bin_method == "clusters":
        bin_edges = _bin_by_clusters(data)

    else:
        counts, bin_edges = np.histogram(data, bins="auto")
        n_bins = counts.size
        if counts[counts > 0].min() < 10:
            counter = 20
            while counts[counts > 0].min() < 10 and n_bins >= 2 and counter > 0:
                counter -= 1
                n_bins -= 1
                counts, bin_edges = np.histogram(data, bins=n_bins)

        if bin_method == "uniform_count":
            quantiles = np.linspace(0, 100, n_bins + 1)
            bin_edges = np.asarray(np.percentile(data, quantiles))

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    return np.digitize(data, bin_edges)


def is_continuous(data: NDArray[np.number[Any]], image_indices: NDArray[np.number[Any]] | None = None) -> bool:
    """
    Determines whether the data is continuous or discrete using the Wasserstein distance.

    Given a 1D sample, we consider the intervals between adjacent points. For a continuous distribution,
    a point is equally likely to lie anywhere in the interval bounded by its two neighbors. Furthermore,
    we can put all "between neighbor" locations on the same scale of 0 to 1 by subtracting the smaller
    neighbor and dividing out the length of the interval. (Duplicates are either assigned to zero or
    ignored, depending on context). These normalized locations will be much more uniformly distributed
    for continuous data than for discrete, and this gives us a way to distinguish them. Call this the
    Normalized Near Neighbor distribution (NNN), defined on the interval [0,1].

    The Wasserstein distance is available in scipy.stats.wasserstein_distance. We can use it to measure
    how close the NNN is to a uniform distribution over [0,1]. We found that as long as a sample has at
    least 20 points, and furthermore at least half as many points as there are discrete values, we can
    reliably distinguish discrete from continuous samples by testing that the Wasserstein distance
    measured from a uniform distribution is greater or less than 0.054, respectively.
    """
    # Check if the metadata is image specific
    if image_indices is not None:
        _, data_indices_unsorted = np.unique(data, return_index=True)
        if data_indices_unsorted.size == image_indices.size:
            data_indices = np.sort(data_indices_unsorted)
            if (data_indices == image_indices).all():
                data = data[data_indices]

    n_examples = len(data)

    if n_examples < CONTINUOUS_MIN_SAMPLE_SIZE:
        _logger.warning(f"All samples look discrete with so few data points (< {CONTINUOUS_MIN_SAMPLE_SIZE})")
        return False

    # Require at least 3 unique values before bothering with NNN
    xu = np.unique(data)
    if xu.size < 3:
        return False

    Xs = np.sort(data)

    X0, X1 = Xs[0:-2], Xs[2:]  # left and right neighbors

    dx = np.zeros(n_examples - 2)  # no dx at end points
    gtz = (X1 - X0) > 0  # check for dups; dx will be zero for them
    dx[np.logical_not(gtz)] = 0.0

    dx[gtz] = (Xs[1:-1] - X0)[gtz] / (X1 - X0)[gtz]  # the core idea: dx is NNN samples.

    shift = wasserstein_distance(dx, np.linspace(0, 1, dx.size))  # how far is dx from uniform, for this feature?

    return bool(shift < DISCRETE_MIN_WD)  # if NNN is close enough to uniform, consider the sample continuous.


def _bin_by_clusters(data: NDArray[np.number[Any]]) -> NDArray[np.float64]:
    """
    Bins continuous data by using the Clusterer to identify clusters
    and incorporates outliers by adding them to the nearest bin.
    """
    # Delay load numba compiled functions
    from dataeval.core._clusterer import cluster

    # Create initial clusters
    c = cluster(data)

    # Create bins from clusters
    bin_edges = np.zeros(c["clusters"].max() + 2)
    for group in range(c["clusters"].max() + 1):
        points = np.nonzero(c["clusters"] == group)[0]
        bin_edges[group] = data[points].min()

    # Get the outliers
    outliers = np.nonzero(c["clusters"] == -1)[0]

    # Identify non-outlier neighbors
    nbrs = c["k_neighbors"][outliers]
    nbrs = np.where(np.isin(nbrs, outliers), -1, nbrs)

    # Find the nearest non-outlier neighbor for each outlier
    nn = np.full(outliers.size, -1, dtype=np.int32)
    for row in range(outliers.size):
        non_outliers = nbrs[row, nbrs[row] != -1]
        if non_outliers.size > 0:
            nn[row] = non_outliers[0]

    # Group outliers by their neighbors
    unique_nnbrs, same_nbr, counts = np.unique(nn, return_inverse=True, return_counts=True)

    # Adjust bin_edges based on each unique neighbor group
    extend_bins = []
    for i, nnbr in enumerate(unique_nnbrs):
        outlier_indices = np.nonzero(same_nbr == i)[0]
        min2add = data[outliers[outlier_indices]].min()
        if counts[i] >= 4:
            extend_bins.append(min2add)
        else:
            if min2add < data[nnbr]:
                clusters = c["clusters"][nnbr]
                bin_edges[clusters] = min2add
    if extend_bins:
        bin_edges = np.concatenate([bin_edges, extend_bins])

    return np.sort(bin_edges)
