# Max Bright
# Return types for data metrics

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class AlibiOutlierDetectorOutput:
    is_outlier: Iterable[bool]
    feature_score: Any
    instance_score: Any
