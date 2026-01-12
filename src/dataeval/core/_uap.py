"""
This module contains the implementation of the
FR Test Statistic based estimate for the :term:`upper-bound
average precision<Upper-Bound Average Precision (UAP)>` using empirical mean precision
"""

__all__ = []

import logging

from sklearn.metrics import average_precision_score

from dataeval.types import Array2D
from dataeval.utils.arrays import as_numpy

_logger = logging.getLogger(__name__)


def uap(labels: Array2D[int], scores: Array2D[float]) -> float:
    """
    FR Test Statistic based estimate of the empirical mean precision for the \
    upperbound average precision.

    Parameters
    ----------
    labels : ArrayLike
        A 2D array of n_samples of class labels with M unique classes.
    scores : ArrayLike
        A 2D array of class probabilities per image.

    Returns
    -------
    float
        The empirical mean precision estimate.

    Raises
    ------
    ValueError
        If unique classes M < 2.

    Notes
    -----
    This function calculates the empirical mean precision using the
    ``average_precision_score`` from scikit-learn, weighted by the class distribution.

    Examples
    --------
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> uap(y_true, y_scores)
    0.8333333333333333

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
    0.7777777777777777
    """
    _logger.info("Starting UAP calculation")

    labels_np = as_numpy(labels)
    scores_np = as_numpy(scores)

    _logger.debug("Labels shape: %s, Scores shape: %s", labels_np.shape, scores_np.shape)

    avg_precision = float(average_precision_score(labels_np, scores_np, average="weighted"))

    _logger.info("UAP calculation complete: uap=%.4f", avg_precision)

    return avg_precision
