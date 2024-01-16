"""This module contains dataclasses for each metric output"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class OutlierDetectorOutput:
    """
    Dataclass to store outputs from Outlier Detection metrics

    Attributes
    ----------
    is_outlier : List[bool]
        Indicates which elements of the dataset are marked as outliers
    feature_score : np.ndarray
        Feature level outlier scores. Shape = (B, H, W, C)
    instance_score : np.ndarray
        Instance (image) level outlier scores. Shape = (B, )
    """

    is_outlier: Iterable[bool]
    feature_score: np.ndarray
    instance_score: np.ndarray


@dataclass
class DivergenceOutput:
    """
    Dataclass to store outputs from Divergence metrics

    Attributes
    ----------
    dpdivergence : float
        Measure of the distance between two distributions (or datasets).

        For more information about this divergence, its formal definition,
        and its associated estimators
        visit https://arxiv.org/abs/1412.6534.

    error : float
        Number of edges connecting the two datasets
    """

    dpdivergence: float
    error: float

    def __eq__(self, other):
        if isinstance(other, DivergenceOutput):
            if self.dpdivergence == other.dpdivergence and self.error == other.error:
                return True
        return False


@dataclass
class BEROutput:
    """
    Dataclass to store output from BER metrics

    Attributes
    ----------
    ber : float
        The minimum error rate that can be achieved by a classifier
        based on the Bayes decision rule. This is the upper bound.
    ber_lower: float
        Lower bound for the above.
    """

    ber: float
    ber_lower: float


@dataclass
class UAPOutput:
    """
    Dataclass to store output from UAP metrics

    Attributes
    ----------
    uap : float
        The maximum average precision that can be achieved by a detector.
    """

    uap: float
