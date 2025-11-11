"""
The utility classes and functions are provided by DataEval to assist users \
in setting up data and architectures that are guaranteed to work with applicable \
DataEval metrics.
"""

from dataeval.utils import models
from dataeval.utils._merge import flatten, merge
from dataeval.utils._split_dataset import split_dataset
from dataeval.utils._unzip_dataset import unzip_dataset

__all__ = ["flatten", "merge", "models", "split_dataset", "unzip_dataset"]
