"""
FR Test Statistic based estimate for :term:`upper-bound average precision<Upper-Bound Average Precision (UAP)>`.

Uses empirical mean precision to estimate the upper-bound average precision.
"""

__all__ = []


from sklearn.metrics import average_precision_score

from dataeval._experimental import experimental
from dataeval._log import get_logger
from dataeval.protocols import ArrayLike
from dataeval.utils._array import as_numpy

_logger = get_logger(__name__)


@experimental
def uap(labels: ArrayLike, scores: ArrayLike) -> float:
    """
    Estimate the empirical mean precision for the upperbound average precision.

    .. warning::
       This feature is experimental and may change or be removed in future releases.

    Uses the FR Test Statistic based approach.

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

    References
    ----------
    [1] Empirical upper bound in object detection and more.
        Borji, A., & Iranmanesh, S. M. (2019). arXiv preprint arXiv:1911.12451.
        https://arxiv.org/abs/1911.12451

    Examples
    --------
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> uap(y_true, y_scores)
    0.8333333333333333

    >>> y_true = np.array([0, 0, 1, 1, 2, 2])
    >>> y_scores = np.array([
    ...     [0.7, 0.2, 0.1],
    ...     [0.4, 0.3, 0.3],
    ...     [0.1, 0.8, 0.1],
    ...     [0.2, 0.3, 0.5],
    ...     [0.4, 0.4, 0.2],
    ...     [0.1, 0.2, 0.7],
    ... ])
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
