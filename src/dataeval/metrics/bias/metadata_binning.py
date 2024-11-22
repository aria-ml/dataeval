from __future__ import annotations

__all__ = ["MetadataOutput", "metadata_binning"]

import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import wasserstein_distance as wd

from dataeval.interop import as_numpy, to_numpy
from dataeval.output import OutputMetadata, set_metadata
from dataeval.utils.metadata import merge_metadata

TNum = TypeVar("TNum", int, float)
DISCRETE_MIN_WD = 0.054
CONTINUOUS_MIN_SAMPLE_SIZE = 20
CLASS_LABEL = "class_labels"


@dataclass(frozen=True)
class MetadataOutput(OutputMetadata):
    """
    Output class for :func:`metadata_binning` function

    Attributes
    ----------
    discrete : Mapping[str, Any]
        Dictionary containing original data that was discrete and the binned continuous data
    continuous : Mapping[str, Any]
        Dictionary containing the original continuous data
    contingency_table : NDArray[np.integer]
    class_labels : NDArray[np.integer]
        Numerical class labels for the images/objects
    class_names : NDArray[Any]
        Array of unique class names (for use with plotting)
    """

    discrete: Mapping[str, Any]
    continuous: Mapping[str, Any]
    contingency_table: NDArray[np.integer]
    class_labels: NDArray[np.integer]
    class_names: NDArray[Any]


@set_metadata()
def metadata_binning(
    raw_metadata: Iterable[Mapping[str, Any]],
    class_labels: ArrayLike | str,
    number_of_images: int,
    continuous_factor_bins: Mapping[str, int | list[tuple[TNum, TNum]]] | None = None,
    auto_bin_method: Literal["uniform_width", "uniform_count", "clusters"] = "uniform_width",
) -> MetadataOutput:
    """
    Calculates various :term:`statistics` for each image

    Parameters
    ----------
    raw_metadata : Iterable[Mapping[str, Any]]
        Iterable collection of metadata dictionaries to flatten and merge.
    class_labels : ArrayLike or string or None
        If arraylike, expects the labels for each image (image classification) or each object (object detection).
        If the labels are included in the metadata dictionary, pass in the key value.
    number_of_images : int
        Number of images (NOT objects) in dataset
    continuous_factor_bins : Mapping[str, int] or Mapping[str, list[tuple[TNum, TNum]]] or None, default None
        User provided dictionary specifying how to bin the continuous metadata factors
    auto_bin_method : "uniform_width" or "uniform_count" or "clusters", default "uniform_width"
        Method by which the function will automatically bin continuous metadata factors. It is recommended
        that the user provide the bins through the `continuous_factor_bins`.

    Returns
    -------
    MetadataOutput
        Output class containing the binned metadata
    NDArray[intp]
        Integer array of the class_labels
    """
    # Transform metadata into single, flattened dictionary
    metadata = merge_metadata(raw_metadata)

    # Get the class label array in numeric form
    class_array = as_numpy(metadata.pop(class_labels)) if isinstance(class_labels, str) else as_numpy(class_labels)
    if not np.issubdtype(class_array.dtype, np.integer):
        unique_classes, numerical_labels = np.unique(class_array, return_inverse=True)
    else:
        numerical_labels = np.asarray(class_array, dtype=np.intp)
        unique_classes = np.unique(class_array)

    # Bin according to user supplied bins
    continuous_metadata = {}
    discrete_metadata = {}
    if continuous_factor_bins is not None and continuous_factor_bins != {}:
        for factor, grouping in continuous_factor_bins.items():
            discrete_metadata[factor] = _user_defined_bin(metadata[factor], grouping)
            continuous_metadata[factor] = metadata[factor]

    # Determine category of the rest of the keys
    remaining_keys = set(metadata.keys()) - set(continuous_metadata.keys())
    for key in remaining_keys:
        data = to_numpy(metadata[key])
        if np.issubdtype(data.dtype, np.number):
            result = _is_continuous(data)
            if result:
                warnings.warn(
                    f"A user defined binning was not provided for {key}.\n \
                    Using the {auto_bin_method} method to discretize the data.\n \
                    It is recommended that the user rerun and supply the desired \
                    bins using the continuous_factor_bins parameter.",
                    UserWarning,
                )
                continuous_metadata[key] = data
                discrete_metadata[key] = _binning_function(data, auto_bin_method)
            else:
                _, discrete_metadata[key] = np.unique(data, return_inverse=True)
        else:
            _, discrete_metadata[key] = np.unique(data, return_inverse=True)

    # creating contingency table from discrete metadata
    contingency_table = np.array(discrete_metadata)  # This actually needs to be worked out, currently a place holder

    return MetadataOutput(discrete_metadata, continuous_metadata, contingency_table, numerical_labels, unique_classes)


def _binning_function(data: NDArray[Any], bin_method: str) -> NDArray[np.intp]:
    """
    Bins continuous data through either equal width bins, equal amounts in each bin, or by clusters.
    """
    if bin_method == "clusters":
        # bin_edges = _binning_by_clusters(data)
        bin_method = "uniform_width"

    if bin_method != "clusters":
        n_bins = 10
        rounds = 0
        while True:
            bin_edges = np.linspace(data.min(), data.max(), n_bins + 1)
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
            equal_widths = np.digitize(data, bin_edges)
            _, counts = np.unique(equal_widths, return_counts=True)
            if counts < 10:
                n_bins -= 1
            elif counts.min() / data.size < 0.01:
                n_bins += 1
            else:
                break

            if rounds >= 20:
                break
            rounds += 1

        if bin_method == "uniform_count":
            quantiles = np.linspace(0, 100, n_bins + 1)
            bin_edges = np.asarray(np.percentile(data, quantiles))

    bin_edges[0] = -np.inf  # type: ignore # until the clusters speed up is merged
    bin_edges[-1] = np.inf  # type: ignore
    return np.digitize(data, bin_edges)  # type: ignore


def _user_defined_bin(
    metadata: NDArray[Any] | list[Any], binning: int | list[tuple[TNum, TNum]]
) -> Mapping[str, int | list[tuple[TNum, TNum]]]:
    return {"two": 3}


def _is_continuous(X: NDArray[np.number]) -> bool:
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
    reliably distinguish discrete from continuous samples by testing that the Wasserstein distance is
    greater or less than 0.054, respectively.
    """
    n_examples = len(X)

    if n_examples < CONTINUOUS_MIN_SAMPLE_SIZE:
        print(f"All samples look discrete with so few data points (< {CONTINUOUS_MIN_SAMPLE_SIZE})")
        return False

    # Require at least 3 unique values before bothering with NNN
    xu = np.unique(X, axis=None)
    if xu.size < 3:
        return False

    Xs = np.sort(X)

    X0, X1 = Xs[0:-2], Xs[2:]  # left and right neighbors

    dx = np.zeros(n_examples - 2)  # no dx at end points
    gtz = (X1 - X0) > 0  # check for dups; dx will be zero for them
    dx[np.logical_not(gtz)] = 0.0

    dx[gtz] = (Xs[1:-1] - X0)[gtz] / (X1 - X0)[gtz]  # the core idea: dx is NNN samples.

    shift = wd(dx, np.linspace(0, 1, dx.size))  # how far is dx from uniform, for this feature?

    return shift < DISCRETE_MIN_WD  # if NNN is close enough to uniform, consider the sample continuous.


# def _binning_by_clusters(data: NDArray[np.number]):
#     """
#     Bins continuous data by using the Clusterer to identify clusters
#     and incorporates outliers by adding them to the nearest bin.
#     """
#     # Create initial clusters
#     groupings = Clusterer(data)
#     clusters = groupings.create_clusters()

#     # Create bins from clusters
#     bin_edges = np.zeros(clusters.max() + 2)
#     for group in range(clusters.max() + 1):
#         points = np.nonzero(clusters == group)[0]
#         bin_edges[group] = data[points].min()

#     # Get the outliers
#     outliers = np.nonzero(clusters == -1)[0]

#     # Identify non-outlier neighbors
#     nbrs = groupings._kneighbors[outliers]
#     nbrs = np.where(np.isin(nbrs, outliers), -1, nbrs)

#     # Find the nearest non-outlier neighbor for each outlier
#     nn = np.full(outliers.size, -1, dtype=np.int32)
#     for row in range(outliers.size):
#         non_outliers = nbrs[row, nbrs[row] != -1]
#         if non_outliers.size > 0:
#             nn[row] = non_outliers[0]

#     # Group outliers by their neighbors
#     unique_nnbrs, same_nbr, counts = np.unique(nn, return_inverse=True, return_counts=True)

#     # Adjust bin_edges based on each unique neighbor group
#     extend_bins = []
#     for i, nnbr in enumerate(unique_nnbrs):
#         outlier_indices = np.nonzero(same_nbr == i)[0]
#         min2add = data[outliers[outlier_indices]].min()
#         if counts[i] >= 4:
#             extend_bins.append(min2add)
#         else:
#             if min2add < data[nnbr]:
#                 cluster = clusters[nnbr]
#                 bin_edges[cluster] = min2add
#     if extend_bins:
#         bin_edges = np.concatenate([bin_edges, extend_bins])

#     bin_edges = np.sort(bin_edges)
#     return bin_edges
