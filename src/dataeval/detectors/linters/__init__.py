"""
Linters help identify potential issues in training and test data and are an important aspect of data cleaning.
"""

__all__ = [
    "Duplicates",
    "DuplicatesOutput",
    "Outliers",
    "OutliersOutput",
]

from dataeval.detectors.linters.duplicates import Duplicates, DuplicatesOutput
from dataeval.detectors.linters.outliers import Outliers, OutliersOutput
