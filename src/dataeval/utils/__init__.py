"""
The utility classes and functions are provided by DataEval to assist users
in setting up architectures that are guaranteed to work with applicable DataEval
metrics. Currently DataEval supports both Tensorflow and PyTorch backends.
"""

from dataeval import _IS_TENSORFLOW_AVAILABLE, _IS_TORCH_AVAILABLE

__all__ = []

if _IS_TORCH_AVAILABLE:  # pragma: no cover
    from . import torch

    __all__ += ["torch"]

if _IS_TENSORFLOW_AVAILABLE:  # pragma: no cover
    from . import tensorflow

    __all__ += ["tensorflow"]
