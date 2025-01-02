"""
The utility classes and functions are provided by DataEval to assist users
in setting up architectures that are guaranteed to work with applicable DataEval
metrics. Currently DataEval supports both :term:`TensorFlow` and PyTorch backends.
"""

__all__ = ["merge_metadata", "split_dataset", "torch"]

from dataeval.utils import torch
from dataeval.utils.metadata import merge_metadata
from dataeval.utils.split_dataset import split_dataset
