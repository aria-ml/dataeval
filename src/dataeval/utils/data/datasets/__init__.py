"""Provides access to common Computer Vision datasets."""

from dataeval.utils.data.datasets._cifar10 import CIFAR10
from dataeval.utils.data.datasets._milco import MILCO
from dataeval.utils.data.datasets._mnist import MNIST
from dataeval.utils.data.datasets._ships import Ships
from dataeval.utils.data.datasets._voc import VOCDetection, VOCDetectionTorch, VOCSegmentation

__all__ = [
    "MNIST",
    "Ships",
    "CIFAR10",
    "MILCO",
    "VOCDetection",
    "VOCDetectionTorch",
    "VOCSegmentation",
]
