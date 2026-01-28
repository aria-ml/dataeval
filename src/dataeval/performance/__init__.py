"""
Determine whether a problem is feasible and how much data is needed.
"""

__all__ = [
    "Sufficiency",
    "SufficiencyOutput",
    "schedules",
]

from . import schedules
from ._output import SufficiencyOutput
from ._sufficiency import Sufficiency
