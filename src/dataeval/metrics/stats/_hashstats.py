from __future__ import annotations

__all__ = []

from typing import Any

from dataeval.core._processor import process
from dataeval.core.processors._hashstats import HashStatsProcessor
from dataeval.metrics.stats._base import convert_output, unzip_dataset
from dataeval.outputs import HashStatsOutput
from dataeval.outputs._base import set_metadata
from dataeval.typing import ArrayLike, Dataset


@set_metadata
def hashstats(
    dataset: Dataset[ArrayLike] | Dataset[tuple[ArrayLike, Any, Any]],
    *,
    per_box: bool = False,
) -> HashStatsOutput:
    """
    Calculates hashes for each image.

    This function computes hashes from the images including exact hashes and perception-based
    hashes. These hash values can be used to determine if images are exact or near matches.

    Parameters
    ----------
    dataset : Dataset
        Dataset to perform calculations on.
    per_box : bool, default False
        If True, perform calculations on each bounding box.

    Returns
    -------
    HashStatsOutput
        A dictionary-like object containing the computed hashes for each image.

    See Also
    --------
    Duplicates

    Examples
    --------
    Calculate the hashes of a dataset of images, whose shape is (C, H, W)

    >>> results = hashstats(dataset)
    >>> print(results.xxhash[:5])
    ['66a93f556577c086', 'd8b686fb405c4105', '7ffdb4990ad44ac6', '42cd4c34c80f6006', 'c5519e36ac1f8839']
    >>> print(results.pchash[:5])
    ['e666999999266666', 'e666999999266666', 'e666999966666299', 'e666999999266666', '96e91656e91616e9']
    """
    stats = process(*unzip_dataset(dataset, per_box), HashStatsProcessor)
    return convert_output(HashStatsOutput, stats)
