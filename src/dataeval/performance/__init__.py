"""Determine whether a problem is feasible and how much data is needed."""

__all__ = [
    "Allowances",
    "Feasibility",
    "FeasibilityOutput",
    "Sufficiency",
    "SufficiencyOutput",
    "TargetSpec",
    "schedules",
]

from . import schedules
from ._feasibility import Feasibility, FeasibilityOutput
from ._feasibility_target import Allowances, TargetSpec
from ._output import SufficiencyOutput
from ._sufficiency import Sufficiency
