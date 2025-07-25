from __future__ import annotations

__all__ = []


from dataeval.core._completeness import completeness as _completeness
from dataeval.outputs import CompletenessOutput
from dataeval.outputs._base import set_metadata
from dataeval.typing import Array


@set_metadata
def completeness(embeddings: Array, quantiles: int) -> CompletenessOutput:
    """
    Calculate the fraction of boxes in a grid defined by quantiles that
    contain at least one data point.
    Also returns the center coordinates of each empty box.

    Parameters
    ----------
    embeddings : Array
        Embedded dataset (or other low-dimensional data) (nxp)
    quantiles : int
        number of quantile values to use for partitioning each dimension
        e.g., 1 would create a grid of 2^p boxes, 2, 3^p etc..

    Returns
    -------
    CompletenessOutput
        - fraction_filled: float - Fraction of boxes that contain at least one
          data point
        - empty_box_centers: List[np.ndarray] - List of coordinates for centers of empty
          boxes

    Raises
    ------
    ValueError
        If embeddings are too high-dimensional (>10)
    ValueError
        If there are too many quantiles (>2)
    ValueError
        If embedding is invalid shape

    Example
    -------
    >>> embs = np.array([[1, 0], [0, 1], [1, 1]])
    >>> quantiles = 1
    >>> result = completeness(embs, quantiles)
    >>> result.fraction_filled
    0.75

    Reference
    ---------
    This implementation is based on https://arxiv.org/abs/2002.03147.

    [1] Byun, Taejoon, and Sanjai Rayadurgam. “Manifold for Machine Learning Assurance.”
    Proceedings of the ACM/IEEE 42nd International Conference on Software Engineering
    """
    return CompletenessOutput(*_completeness(embeddings, quantiles))
