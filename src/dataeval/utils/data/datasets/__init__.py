"""Provides access to common Computer Vision datasets."""

from dataeval.utils.data.datasets._ic import CIFAR10, MNIST
from dataeval.utils.data.datasets._od import VOCDetection

__all__ = ["MNIST", "CIFAR10", "VOCDetection"]
