"""
Provide access to common Torch datasets used for computer vision
"""

from dataeval import _IS_TORCHVISION_AVAILABLE

__all__ = []

if _IS_TORCHVISION_AVAILABLE:
    from dataeval._internal.datasets import CIFAR10, MNIST, VOCDetection

    __all__ += ["CIFAR10", "MNIST", "VOCDetection"]
