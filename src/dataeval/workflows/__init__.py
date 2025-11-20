"""
Workflows perform a sequence of actions to analyze the dataset and make predictions.
"""

__all__ = ["Sufficiency", "SufficiencyConfig", "SufficiencyOutput"]

from dataeval.workflows._output import SufficiencyOutput
from dataeval.workflows.sufficiency import Sufficiency, SufficiencyConfig
