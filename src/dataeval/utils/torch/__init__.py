"""
PyTorch is the primary backend for metrics that require neural networks.

While these metrics can take in custom models, DataEval provides utility classes
to create a seamless integration between custom models and DataEval's metrics.
"""

from dataeval import _IS_TORCH_AVAILABLE, _IS_TORCHVISION_AVAILABLE

__all__ = []

if _IS_TORCH_AVAILABLE:
    from dataeval.utils.torch import models, trainer
    from dataeval.utils.torch.utils import read_dataset

    __all__ += ["read_dataset", "models", "trainer"]

if _IS_TORCHVISION_AVAILABLE:
    from dataeval.utils.torch import datasets

    __all__ += ["datasets"]


del _IS_TORCH_AVAILABLE
del _IS_TORCHVISION_AVAILABLE
