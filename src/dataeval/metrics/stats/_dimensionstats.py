from __future__ import annotations

__all__ = []

from typing import Any

from dataeval.core._processor import process
from dataeval.core.processors._dimensionstats import DimensionStatsProcessor
from dataeval.metrics.stats._base import convert_output, unzip_dataset
from dataeval.outputs import DimensionStatsOutput
from dataeval.outputs._base import set_metadata
from dataeval.typing import ArrayLike, Dataset


@set_metadata
def dimensionstats(
    dataset: Dataset[ArrayLike] | Dataset[tuple[ArrayLike, Any, Any]],
    *,
    per_box: bool = False,
) -> DimensionStatsOutput:
    """
    Calculates dimension :term:`statistics<Statistics>` for each image.

    This function computes various dimensional metrics (e.g., width, height, channels)
    on the images or individual bounding boxes for each image.

    Parameters
    ----------
    dataset : Dataset
        Dataset to perform calculations on.
    per_box : bool, default False
        If True, perform calculations on each bounding box.

    Returns
    -------
    DimensionStatsOutput
        A dictionary-like object containing the computed dimension statistics for each image or bounding
        box. The keys correspond to the names of the statistics (e.g., 'width', 'height'), and the values
        are lists of results for each image or :term:NumPy` arrays when the results are multi-dimensional.

    See Also
    --------
    imagestats, Outliers

    Examples
    --------
    Calculate the dimension statistics of a dataset of 8 images, whose shape is (C, H, W).

    >>> results = dimensionstats(dataset)
    >>> print(results.aspect_ratio)
    [1.    1.    1.333 1.    0.667 1.    1.    1.   ]
    >>> print(results.channels)
    [3 3 1 3 1 3 3 3]
    """
    stats = process(*unzip_dataset(dataset, per_box), DimensionStatsProcessor)
    return convert_output(DimensionStatsOutput, stats)
