from __future__ import annotations

__all__ = ["MetadataOutput", "metadata_binning"]

import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, TypeVar

import numpy as np
from numpy.typing import NDArray

from dataeval.output import OutputMetadata, set_metadata
from dataeval.utils.metadata import merge_metadata

TNum = TypeVar("TNum", int, float)


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
    contingency_table : NDArray[np.int64]
    """

    discrete: Mapping[str, Any]
    continuous: Mapping[str, Any]
    contingency_table: NDArray[np.int64]


@set_metadata()
def metadata_binning(
    raw_metadata: Iterable[Mapping[str, Any]],
    continuous_factor_bins: Mapping[str, int | list[tuple[TNum, TNum]]] | None = None,
    auto_bin_method: Literal["uniform_width", "uniform_count", "clusters"] = "clusters",
) -> MetadataOutput:
    """
    Calculates various :term:`statistics<Statistics>` for each image

    Parameters
    ----------
    raw_metadata :
    continuous_factor_bins :

    Returns
    -------
    MetadataOutput
        Output class containing the binned metadata

    """
    # use merge_metadata to transform metadata into desired format
    metadata = merge_metadata(raw_metadata)

    # use user supplied bins
    continuous_metadata = {}
    discrete_metadata = {}
    if continuous_factor_bins is not None and continuous_factor_bins != {}:
        for factor, grouping in continuous_factor_bins.items():
            discrete_metadata[factor] = user_defined_bin(metadata[factor], grouping)
            continuous_metadata[factor] = metadata[factor]

    # determine category of the rest of the keys
    remaining_keys = set(metadata.keys()) - set(continuous_metadata.keys())
    for key in remaining_keys:
        result = is_continuous(metadata[key])
        if result:
            warnings.warn(
                f"A user defined binning was not provided for {key}.\n \
                Using the {auto_bin_method} method to discretize the data.",
                UserWarning,
            )
            continuous_metadata[key] = metadata[key]
            discrete_metadata[key] = binning_function(metadata[key], auto_bin_method)
        else:
            discrete_metadata[key] = metadata[key]

    # creating contingency table from discrete metadata
    contingency_table = np.array(discrete_metadata)  # This actually needs to be worked out, currently a place holder

    return MetadataOutput(discrete_metadata, continuous_metadata, contingency_table)


# have option for user to specify type of binning like KBinsDiscretizer
# density based binning using knn to get centers (potentially use MST to define centers)
# need to determine difference between returning mean vs ordinal (might not be one)
def binning_function(metadata, bin_method):
    return


# (continuous_values: NDArray[Any], bins: int, factor_name: str) -> NDArray[np.intp]:
def user_defined_bin(metadata: NDArray[Any], binning: int) -> NDArray[np.intp]:
    """
    Digitizes a list of values into a given number of bins.

    Parameters
    ----------
    metadata : NDArray
        The values to be digitized.
    binning : int
        The number of bins for the discrete values that metadata will be digitized into.

    Returns
    -------
    NDArray[np.intp]
        The digitized values
    """

    if not np.all([np.issubdtype(type(n), np.number) for n in metadata]):
        raise TypeError(
            "Encountered a metadata value with non-numeric type when digitizing a factor.",
            " Ensure all occurrences of continuous factors are numeric types.",
        )

    _, bin_edges = np.histogram(metadata, bins=binning)
    bin_edges[-1] = np.inf
    bin_edges[0] = -np.inf
    return np.digitize(metadata, bin_edges)


def is_continuous(metadata):
    return
