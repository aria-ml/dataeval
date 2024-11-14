"""
The utility classes and functions are provided by DataEval to assist users
in setting up architectures that are guaranteed to work with applicable DataEval
metrics. Currently DataEval supports both :term:`TensorFlow` and PyTorch backends.
"""

from dataeval import _IS_TENSORFLOW_AVAILABLE, _IS_TORCH_AVAILABLE
from dataeval.utils.metadata import merge_metadata
from dataeval.utils.split_dataset import split_dataset

__all__ = ["split_dataset", "merge_metadata"]

if _IS_TORCH_AVAILABLE:  # pragma: no cover
    from dataeval.utils import torch

    __all__ += ["torch"]

if _IS_TENSORFLOW_AVAILABLE:  # pragma: no cover
    from dataeval.utils import tensorflow

    __all__ += ["tensorflow"]

del _IS_TENSORFLOW_AVAILABLE
del _IS_TORCH_AVAILABLE
