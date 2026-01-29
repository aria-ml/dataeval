"""
DataEval utilities organized by domain.

The utility classes and functions are provided by DataEval to assist users
in setting up data and architectures that are guaranteed to work with applicable
DataEval metrics.
"""

from dataeval.utils import arrays, data, losses, models, onnx, preprocessing, training

__all__ = [
    "arrays",
    "data",
    "losses",
    "models",
    "onnx",
    "preprocessing",
    "training",
]
