"""This module contains dataclasses for each metric output"""

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class OutlierDetectorOutput:
    """
    dataclass to store outputs from an Alibi-Detect Outlier Detector
    """

    is_outlier: Iterable[bool]
    feature_score: Any
    instance_score: Any


@dataclass
class DivergenceOutput:
    """
    dataclass to store outputs from Dp Divergence
    """

    dpdivergence: float
    error: float


@dataclass
class BEROutput:
    """
    dataclass to store output from a BER calculation
    """

    ber: float
