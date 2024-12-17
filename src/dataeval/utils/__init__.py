"""
The utility classes and functions are provided by DataEval to assist users
in setting up architectures that are guaranteed to work with applicable DataEval
metrics. Currently DataEval supports both :term:`TensorFlow` and PyTorch backends.
"""

from dataeval import _IS_TORCH_AVAILABLE
from dataeval.utils.metadata import merge_metadata
from dataeval.utils.split_dataset import split_dataset

__all__ = ["split_dataset", "merge_metadata"]

if _IS_TORCH_AVAILABLE:
    from dataeval.utils import torch

    __all__ += ["torch"]

del _IS_TORCH_AVAILABLE
