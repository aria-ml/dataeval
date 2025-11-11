from __future__ import annotations

__all__ = []

from typing import Any

from dataeval.core._calculate import calculate
from dataeval.core.flags import ImageStats
from dataeval.metrics.stats._base import convert_output
from dataeval.outputs import HashStatsOutput
from dataeval.protocols import ArrayLike, Dataset
from dataeval.types import set_metadata
from dataeval.utils import unzip_dataset


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
    >>> print(results.xxhash[:3])
    ['66a93f556577c086' 'd8b686fb405c4105' '7ffdb4990ad44ac6']
    >>> print(results.pchash[:3])
    ['e666999999266666' 'e666999999266666' 'e666999966666299']
    """
    stats = calculate(
        *unzip_dataset(dataset, per_box),
        stats=ImageStats.HASH,
        per_image=not per_box,
        per_target=per_box,
    )
    return convert_output(HashStatsOutput, stats)
