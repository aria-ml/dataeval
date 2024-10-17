"""
PyTorch is the primary backend for metrics that require neural networks.

While these metrics can take in custom models, DataEval provides utility classes
to create a seamless integration between custom models and DataEval's metrics.
"""

from dataeval._internal.utils import read_dataset

from . import models, trainer

__all__ = ["read_dataset", "models", "trainer"]
