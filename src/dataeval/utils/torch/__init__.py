"""
PyTorch is the primary backend for metrics that require neural networks.

While these metrics can take in custom models, DataEval provides utility classes
to create a seamless integration between custom models and DataEval's metrics.
"""

__all__ = ["models", "trainer"]

from dataeval.utils.torch import models, trainer
