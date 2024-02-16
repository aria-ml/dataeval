from dataclasses import dataclass
from enum import Enum


class ThresholdType(Enum):
    """
    Enum of threshold types for outlier detection
    """

    VALUE = "value"
    PERCENTAGE = "percentage"


@dataclass
class Threshold:
    """
    Dataclass to specify the threshold value and type for outlier detection
    """

    value: float
    type: ThresholdType
