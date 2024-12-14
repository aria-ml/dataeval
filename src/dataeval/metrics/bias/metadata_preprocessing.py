from __future__ import annotations

__all__ = ["MetadataOutput", "metadata_preprocessing"]

import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import wasserstein_distance as wd

from dataeval.interop import as_numpy, to_numpy
from dataeval.output import Output, set_metadata
from dataeval.utils.metadata import merge_metadata

TNum = TypeVar("TNum", int, float)
DISCRETE_MIN_WD = 0.054
CONTINUOUS_MIN_SAMPLE_SIZE = 20


@dataclass(frozen=True)
class MetadataOutput(Output):
    """
    Output class for :func:`metadata_binning` function

    Attributes
    ----------
    discrete_factor_names : list[str]
        List containing factor names for the original data that was discrete and the binned continuous data
    discrete_data : NDArray[np.int]
        Array containing values for the original data that was discrete and the binned continuous data
    continuous_factor_names : list[str]
        List containing factor names for the original continuous data
    continuous_data : NDArray[np.int or np.double] | None
        Array containing values for the original continuous data or None if there was no continuous data
    class_labels : NDArray[np.int]
        Numerical class labels for the images/objects
    class_names : NDArray[Any]
        Array of unique class names (for use with plotting)
    total_num_factors : int
        Sum of discrete_factor_names and continuous_factor_names plus 1 for class
    """

    discrete_factor_names: list[str]
    discrete_data: NDArray[np.int_]
    continuous_factor_names: list[str]
    continuous_data: NDArray[np.int_ | np.double] | None
    class_labels: NDArray[np.int_]
    class_names: NDArray[Any]
    total_num_factors: int


@set_metadata
def metadata_preprocessing(
    raw_metadata: Iterable[Mapping[str, Any]],
    class_labels: ArrayLike | str,
    continuous_factor_bins: Mapping[str, int | list[tuple[TNum, TNum]]] | None = None,
    auto_bin_method: Literal["uniform_width", "uniform_count", "clusters"] = "uniform_width",
    exclude: Iterable[str] | None = None,
) -> MetadataOutput:
    """
    Restructures the metadata to be in the correct format for the bias functions.

    This identifies whether the incoming metadata is discrete or continuous,
    and whether the data is already binned or still needs binning.
    It accepts a list of dictionaries containing the per image metadata and
    automatically adjusts for multiple targets in an image.

    Parameters
    ----------
    raw_metadata : Iterable[Mapping[str, Any]]
        Iterable collection of metadata dictionaries to flatten and merge.
    class_labels : ArrayLike or string or None
        If arraylike, expects the labels for each image (image classification) or each object (object detection).
        If the labels are included in the metadata dictionary, pass in the key value.
    continuous_factor_bins : Mapping[str, int] or Mapping[str, list[tuple[TNum, TNum]]] or None, default None
        User provided dictionary specifying how to bin the continuous metadata factors
    auto_bin_method : "uniform_width" or "uniform_count" or "clusters", default "uniform_width"
        Method by which the function will automatically bin continuous metadata factors. It is recommended
        that the user provide the bins through the `continuous_factor_bins`.
    exclude : Iterable[str] or None, default None
        User provided collection of metadata keys to exclude when processing metadata.

    Returns
    -------
    MetadataOutput
        Output class containing the binned metadata
    """
    # Transform metadata into single, flattened dictionary
    metadata, image_repeats = merge_metadata(raw_metadata)

    # Drop any excluded metadata keys
    if exclude:
        for k in list(metadata):
            if k in exclude:
                metadata.pop(k)

    # Get the class label array in numeric form
    class_array = as_numpy(metadata.pop(class_labels)) if isinstance(class_labels, str) else as_numpy(class_labels)
    if class_array.ndim > 1:
        raise ValueError(
            f"Got class labels with {class_array.ndim}-dimensional "
            f"shape {class_array.shape}, but expected a 1-dimensional array."
        )
    if not np.issubdtype(class_array.dtype, np.int_):
        unique_classes, numerical_labels = np.unique(class_array, return_inverse=True)
    else:
        numerical_labels = class_array
        unique_classes = np.unique(class_array)

    # Bin according to user supplied bins
    continuous_metadata = {}
    discrete_metadata = {}
    if continuous_factor_bins is not None and continuous_factor_bins != {}:
        invalid_keys = set(continuous_factor_bins.keys()) - set(metadata.keys())
        if invalid_keys:
            raise KeyError(
                f"The keys - {invalid_keys} - are present in the `continuous_factor_bins` dictionary "
                "but are not keys in the `metadata` dictionary. Delete these keys from `continuous_factor_bins` "
                "or add corresponding entries to the `metadata` dictionary."
            )
        for factor, grouping in continuous_factor_bins.items():
            discrete_metadata[factor] = _user_defined_bin(metadata[factor], grouping)
            continuous_metadata[factor] = metadata[factor]

    # Determine category of the rest of the keys
    remaining_keys = set(metadata.keys()) - set(continuous_metadata.keys())
    for key in remaining_keys:
        data = to_numpy(metadata[key])
        if np.issubdtype(data.dtype, np.number):
            result = _is_continuous(data, image_repeats)
            if result:
                continuous_metadata[key] = data
            unique_samples, ordinal_data = np.unique(data, return_inverse=True)
            if unique_samples.size <= np.max([20, data.size * 0.01]):
                discrete_metadata[key] = ordinal_data
            else:
                warnings.warn(
                    f"A user defined binning was not provided for {key}. "
                    f"Using the {auto_bin_method} method to discretize the data. "
                    "It is recommended that the user rerun and supply the desired "
                    "bins using the continuous_factor_bins parameter.",
                    UserWarning,
                )
                discrete_metadata[key] = _binning_function(data, auto_bin_method)
        else:
            _, discrete_metadata[key] = np.unique(data, return_inverse=True)

    # splitting out the dictionaries into the keys and values
    discrete_factor_names = list(discrete_metadata.keys())
    discrete_data = np.stack(list(discrete_metadata.values()), axis=-1)
    continuous_factor_names = list(continuous_metadata.keys())
    continuous_data = np.stack(list(continuous_metadata.values()), axis=-1) if continuous_metadata else None
    total_num_factors = len(discrete_factor_names + continuous_factor_names) + 1

    return MetadataOutput(
        discrete_factor_names,
        discrete_data,
        continuous_factor_names,
        continuous_data,
        numerical_labels,
        unique_classes,
        total_num_factors,
    )


def _user_defined_bin(data: list[Any] | NDArray[Any], binning: int | list[tuple[TNum, TNum]]) -> NDArray[np.intp]:
    """
    Digitizes a list of values into a given number of bins.

    Parameters
    ----------
    data : list | NDArray
        The values to be digitized.
    binning :  int | list[tuple[TNum, TNum]]
        The number of bins for the discrete values that data will be digitized into.

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
    if type(binning) is int:
        _, bin_edges = np.histogram(data, bins=binning)
        bin_edges[-1] = np.inf
        bin_edges[0] = -np.inf
    else:
        bin_edges = binning
    return np.digitize(data, bin_edges)


def _binning_function(data: NDArray[Any], bin_method: str) -> NDArray[np.int_]:
    """
    Bins continuous data through either equal width bins, equal amounts in each bin, or by clusters.
    """
    if bin_method == "clusters":
        # bin_edges = _binning_by_clusters(data)
        warnings.warn(
            "Binning by clusters is currently unavailable until changes to the clustering function go through.",
            UserWarning,
        )
        bin_method = "uniform_width"

    if bin_method != "clusters":
        counts, bin_edges = np.histogram(data, bins="auto")
        n_bins = counts.size
        if counts[counts > 0].min() < 10:
            for _ in range(20):
                n_bins -= 1
                counts, bin_edges = np.histogram(data, bins=n_bins)
                if counts[counts > 0].min() >= 10 or n_bins < 2:
                    break

        if bin_method == "uniform_count":
            quantiles = np.linspace(0, 100, n_bins + 1)
            bin_edges = np.asarray(np.percentile(data, quantiles))

    bin_edges[0] = -np.inf  # type: ignore # until the clusters speed up is merged
    bin_edges[-1] = np.inf  # type: ignore # and the _binning_by_clusters can be uncommented
    return np.digitize(data, bin_edges)  # type: ignore


def _is_continuous(data: NDArray[np.number], image_indicies: NDArray[np.number]) -> bool:
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
    _, data_indicies_unsorted = np.unique(data, return_index=True)
    if data_indicies_unsorted.size == image_indicies.size:
        data_indicies = np.sort(data_indicies_unsorted)
        if (data_indicies == image_indicies).all():
            data = data[data_indicies]

    # OLD METHOD
    # uvals = np.unique(data)
    # pct_unique = uvals.size / data.size
    # return pct_unique < threshold

    n_examples = len(data)

    if n_examples < CONTINUOUS_MIN_SAMPLE_SIZE:
        warnings.warn(
            f"All samples look discrete with so few data points (< {CONTINUOUS_MIN_SAMPLE_SIZE})", UserWarning
        )
        return False

    # Require at least 3 unique values before bothering with NNN
    xu = np.unique(data, axis=None)
    if xu.size < 3:
        return False

    Xs = np.sort(data)

    X0, X1 = Xs[0:-2], Xs[2:]  # left and right neighbors

    dx = np.zeros(n_examples - 2)  # no dx at end points
    gtz = (X1 - X0) > 0  # check for dups; dx will be zero for them
    dx[np.logical_not(gtz)] = 0.0

    dx[gtz] = (Xs[1:-1] - X0)[gtz] / (X1 - X0)[gtz]  # the core idea: dx is NNN samples.

    shift = wd(dx, np.linspace(0, 1, dx.size))  # how far is dx from uniform, for this feature?

    return shift < DISCRETE_MIN_WD  # if NNN is close enough to uniform, consider the sample continuous.
