"""
DataEval utilities organized by domain.

The utility classes and functions are provided by DataEval to assist users
in setting up data and architectures that are guaranteed to work with applicable
DataEval metrics.
"""

from dataeval.utils import data, losses, models, onnx, preprocessing, thresholds, training
from dataeval.utils._array import as_numpy, flatten_samples, to_numpy

__all__ = [
    # Array conversion. Exported because downstream projects were reaching into a private
    # module for them, and a rename there broke those callers at call time rather than at
    # import -- the failure a public name exists to prevent.
    "as_numpy",
    "data",
    "flatten_samples",
    "losses",
    "models",
    "onnx",
    "preprocessing",
    "thresholds",
    "to_numpy",
    "training",
]
