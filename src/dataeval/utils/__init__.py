"""
The utility classes and functions are provided by DataEval to assist users \
in setting up data and architectures that are guaranteed to work with applicable \
DataEval metrics.
"""

from dataeval.utils import collate
from dataeval.utils._merge import flatten, merge
from dataeval.utils._validate import validate_dataset

__all__ = [
    "collate",
    "flatten",
    "merge",
    "validate_dataset",
]
