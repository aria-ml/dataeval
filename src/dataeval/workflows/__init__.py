"""
Workflows perform a sequence of actions to analyze the dataset and make predictions.
"""

__all__ = ["Sufficiency", "SufficiencyOutput"]

from dataeval.outputs._workflows import SufficiencyOutput
from dataeval.workflows.sufficiency import Sufficiency
