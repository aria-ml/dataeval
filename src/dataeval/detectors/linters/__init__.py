"""
Linters help identify potential issues in training and test data and are an important aspect of data cleaning.
"""

__all__ = [
    "Duplicates",
    "DuplicatesOutput",
    "Outliers",
    "OutliersOutput",
]

from dataeval.detectors.linters.duplicates import Duplicates
from dataeval.detectors.linters.outliers import Outliers
from dataeval.outputs._linters import DuplicatesOutput, OutliersOutput
