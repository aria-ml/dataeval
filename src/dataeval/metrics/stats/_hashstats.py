from __future__ import annotations

__all__ = []

from collections.abc import Callable
from typing import Any

from dataeval.core._hash import pchash, xxhash
from dataeval.metrics.stats._base import StatsProcessor, run_stats
from dataeval.outputs import HashStatsOutput
from dataeval.outputs._base import set_metadata
from dataeval.typing import ArrayLike, Dataset


class HashStatsProcessor(StatsProcessor[HashStatsOutput]):
    output_class: type = HashStatsOutput
    image_function_map: dict[str, Callable[[StatsProcessor[HashStatsOutput]], str]] = {
        "xxhash": lambda x: xxhash(x.image),
        "pchash": lambda x: pchash(x.image),
    }


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
    ['69b50a5f06af238c', '5a861d7a23d1afe7', '7ffdb4990ad44ac6', '4f0c366a3298ceac', 'c5519e36ac1f8839']
    >>> print(results.pchash[:5])
    ['e666999999266666', 'e666999999266666', 'e666999966666299', 'e666999999266666', '96e91656e91616e9']
    """
    return run_stats(dataset, per_box, False, [HashStatsProcessor])[0]
