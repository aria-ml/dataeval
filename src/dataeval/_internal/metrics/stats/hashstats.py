from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from numpy.typing import ArrayLike

from dataeval._internal.metrics.stats.base import BaseStatsOutput, StatsProcessor, run_stats
from dataeval._internal.metrics.utils import pchash, xxhash
from dataeval._internal.output import set_metadata


@dataclass(frozen=True)
class HashStatsOutput(BaseStatsOutput):
    """
    Output class for :func:`hashstats` stats metric

    Attributes
    ----------
    xxhash : List[str]
        xxHash hash of the images as a hex string
    pchash : List[str]
        Perception hash of the images as a hex string
    """

    xxhash: list[str]
    pchash: list[str]


class HashStatsProcessor(StatsProcessor[HashStatsOutput]):
    output_class = HashStatsOutput
    image_function_map = {
        "xxhash": lambda x: xxhash(x.image),
        "pchash": lambda x: pchash(x.image),
    }


@set_metadata("dataeval.metrics")
def hashstats(
    images: Iterable[ArrayLike],
    bboxes: Iterable[ArrayLike] | None = None,
) -> HashStatsOutput:
    """
    Calculates hashes for each image

    This function computes hashes from the images including exact hashes and perception-based
    hashes. These hash values can be used to determine if images are exact or near matches.

    Parameters
    ----------
    images : ArrayLike
        Images to hashing
    bboxes : Iterable[ArrayLike] or None
        Bounding boxes in `xyxy` format for each image

    Returns
    -------
    HashStatsOutput
        A dictionary-like object containing the computed hashes for each image.

    See Also
    --------
    Duplicates

    Examples
    --------
    Calculating the statistics on the images, whose shape is (C, H, W)

    >>> results = hashstats(images)
    >>> print(results.xxhash)
    ['a72434443d6e7336', 'efc12c2f14581d79', '4a1e03483a27d674', '3a3ecedbcf814226']
    >>> print(results.pchash)
    ['8f25506af46a7c6a', '8000808000008080', '8e71f18e0ef18e0e', 'a956d6a956d6a928']
    """
    return run_stats(images, bboxes, False, [HashStatsProcessor])[0]
