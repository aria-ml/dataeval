"""
This module contains the implementation of the
FR Test Statistic based estimate for the upperbound
average precision using empirical mean precision
"""

from dataclasses import dataclass

from numpy.typing import ArrayLike
from sklearn.metrics import average_precision_score

from dataeval._internal.interop import to_numpy
from dataeval._internal.output import OutputMetadata, set_metadata


@dataclass(frozen=True)
class UAPOutput(OutputMetadata):
    """
    Attributes
    ----------
    uap : float
        The empirical mean precision estimate
    """

    uap: float


@set_metadata("dataeval.metrics.uap")
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
    Dict[str, float]
        uap : The empirical mean precision estimate

    Raises
    ------
    ValueError
        If unique classes M < 2
    """

    precision = float(average_precision_score(to_numpy(labels), to_numpy(scores), average="weighted"))
    return UAPOutput(precision)
