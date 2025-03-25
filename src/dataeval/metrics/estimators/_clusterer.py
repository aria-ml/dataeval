from __future__ import annotations

__all__ = []


from dataeval.outputs import ClustererOutput
from dataeval.typing import ArrayLike
from dataeval.utils._array import as_numpy


def clusterer(data: ArrayLike) -> ClustererOutput:
    """
    Uses hierarchical clustering on the flattened data and returns clustering
    information.

    Parameters
    ----------
    data : ArrayLike, shape - (N, ...)
        A dataset in an ArrayLike format. Function expects the data to have 2
        or more dimensions which will flatten to (N, P) where N number of
        observations in a P-dimensional space.

    Returns
    -------
    :class:`.ClustererOutput`

    Note
    ----
    The clusterer works best when the length of the feature dimension, P, is
    less than 500. If flattening a CxHxW image results in a dimension larger
    than 500, then it is recommended to reduce the dimensions.

    Example
    -------
    >>> clusterer(clusterer_images).clusters
    array([ 2,  0,  0,  0,  0,  0,  4,  0,  3,  1,  1,  0,  2,  0,  0,  0,  0,
            4,  2,  0,  0,  1,  2,  0,  1,  3,  0,  3,  3,  4,  0,  0,  3,  0,
            3, -1,  0,  0,  2,  4,  3,  4,  0,  1,  0, -1,  3,  0,  0,  0])
    """
    # Delay load numba compiled functions
    from dataeval.utils._clusterer import cluster

    c = cluster(data)
    return ClustererOutput(c.clusters, c.mst, c.linkage_tree, as_numpy(c.condensed_tree), c.membership_strengths)
