"""
Detectors can determine if a dataset or individual images in a dataset are indicative of a specific issue.
"""

from dataeval import _IS_TENSORFLOW_AVAILABLE
from dataeval.detectors import drift, linters

__all__ = ["drift", "linters"]

if _IS_TENSORFLOW_AVAILABLE:
    from dataeval.detectors import ood

    __all__ += ["ood"]

del _IS_TENSORFLOW_AVAILABLE
