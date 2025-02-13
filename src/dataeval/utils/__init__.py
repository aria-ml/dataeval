"""
The utility classes and functions are provided by DataEval to assist users \
in setting up data and architectures that are guaranteed to work with applicable \
DataEval metrics.
"""

__all__ = ["data", "metadata", "torch", "Targets", "Metadata"]

from dataeval.utils import data, metadata, torch
from dataeval.utils._targets import Targets
from dataeval.utils.metadata import Metadata
