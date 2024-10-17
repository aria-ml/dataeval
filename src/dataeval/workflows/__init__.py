"""
Workflows perform a sequence of actions to analyze the dataset and make predictions.
"""

from dataeval import _IS_TORCH_AVAILABLE

if _IS_TORCH_AVAILABLE:  # pragma: no cover
    from dataeval._internal.workflows.sufficiency import Sufficiency, SufficiencyOutput

    __all__ = ["Sufficiency", "SufficiencyOutput"]
