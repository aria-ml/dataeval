"""
This module contains the implementation of the
FR Test Statistic based estimate for the upperbound
average precision using empirical mean precision
"""

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import ArrayLike
from sklearn.metrics import average_precision_score

from dataeval._internal.interop import as_numpy
from dataeval._internal.output import OutputMetadata, set_metadata


@dataclass(frozen=True)
class UAPOutput(OutputMetadata):
    """
    Output class for :func:`uap` estimator metric

    Attributes
    ----------
    uap : float
        The empirical mean precision estimate
    """

    uap: float


@set_metadata("dataeval.metrics")
def uap(labels: ArrayLike, scores: ArrayLike) -> UAPOutput:
    """
    FR Test Statistic based estimate of the empirical mean precision for
    the upperbound average precision

    Parameters
    ----------
    labels : ArrayLike
        A numpy array of n_samples of class labels with M unique classes.
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
