"""This module contains dataclasses for each metric output"""

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class OutlierDetectorOutput:
    """
    Dataclass to store outputs from Outlier Detection

    Attributes
    ----------
    is_outlier : Iterable[bool]
        Indicates which elements of the dataset are marked as outliers
    feature_score : Any
        TODO
    instance_score : Any
        TODO
    """

    is_outlier: Iterable[bool]
    feature_score: Any
    instance_score: Any


@dataclass
class DivergenceOutput:
    """
    Dataclass to store outputs from Divergence metrics

    Attributes
    ----------
    dpdivergence : float
        TODO
    error : float
        TODO
    """

    dpdivergence: float
    error: float


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
