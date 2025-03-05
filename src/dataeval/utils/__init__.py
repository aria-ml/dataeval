"""
The utility classes and functions are provided by DataEval to assist users \
in setting up data and architectures that are guaranteed to work with applicable \
DataEval metrics.
"""

__all__ = ["data", "metadata", "torch", "Metadata", "Targets"]

from dataeval.utils._targets import Targets
from dataeval.utils.metadata import Metadata

from . import data, metadata, torch
