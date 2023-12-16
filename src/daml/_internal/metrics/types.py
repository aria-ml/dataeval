from dataclasses import dataclass
from enum import Enum


class ThresholdType(Enum):
    VALUE = "value"
    PERCENTAGE = "percentage"


@dataclass
class Threshold:
    value: float
    type: ThresholdType
