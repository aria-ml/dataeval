"""This module contains dataclasses for each metric output"""

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class AlibiOutlierDetectorOutput:
    is_outlier: Iterable[bool]
    feature_score: Any
    instance_score: Any
