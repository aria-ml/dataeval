"""Provides access to common Computer Vision datasets."""

from dataeval.utils.datasets._antiuav import AntiUAVDetection
from dataeval.utils.datasets._cifar10 import CIFAR10
from dataeval.utils.datasets._milco import MILCO
from dataeval.utils.datasets._mnist import MNIST
from dataeval.utils.datasets._ships import Ships
from dataeval.utils.datasets._voc import VOCDetection, VOCDetectionTorch, VOCSegmentation

__all__ = [
    "MNIST",
    "Ships",
    "CIFAR10",
    "AntiUAVDetection",
    "MILCO",
    "VOCDetection",
    "VOCDetectionTorch",
    "VOCSegmentation",
]
