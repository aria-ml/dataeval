from dataclasses import dataclass
from enum import Enum


class OutlierType(str, Enum):
    INSTANCE = "instance"
    FEATURE = "feature"

    def __str__(self) -> str:
        return str.__str__(self)  # pragma: no cover


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
