"""
This module contains the implementation of the
FR Test Statistic based estimate for the :term:`upper-bound
average precision<Upper-Bound Average Precision (UAP)>` using empirical mean precision
"""

from __future__ import annotations

__all__ = []


from sklearn.metrics import average_precision_score

from dataeval.outputs import UAPOutput
from dataeval.outputs._base import set_metadata
from dataeval.typing import ArrayLike
from dataeval.utils._array import as_numpy


@set_metadata
def uap(labels: ArrayLike, scores: ArrayLike) -> UAPOutput:
    """
    FR Test Statistic based estimate of the empirical mean precision for the \
    upperbound average precision.

    Parameters
    ----------
    labels : ArrayLike
        A term:`NumPy` array of n_samples of class labels with M unique classes.
    scores : ArrayLike
        A 2D array of class probabilities per image

    Returns
    -------
    UAPOutput
        The empirical mean precision estimate, float

    Raises
    ------
    ValueError
        If unique classes M < 2

    Note
    ----
    This function calculates the empirical mean precision using the
    ``average_precision_score`` from scikit-learn, weighted by the class distribution.

    Examples
    --------
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> uap(y_true, y_scores)
    UAPOutput(uap=0.8333333333333333)

    >>> y_true = np.array([0, 0, 1, 1, 2, 2])
    >>> y_scores = np.array(
    ...     [
    ...         [0.7, 0.2, 0.1],
    ...         [0.4, 0.3, 0.3],
    ...         [0.1, 0.8, 0.1],
    ...         [0.2, 0.3, 0.5],
    ...         [0.4, 0.4, 0.2],
    ...         [0.1, 0.2, 0.7],
    ...     ]
    ... )
    >>> uap(y_true, y_scores)
    UAPOutput(uap=0.7777777777777777)
    """

    precision = float(average_precision_score(as_numpy(labels), as_numpy(scores), average="weighted"))
    return UAPOutput(precision)
