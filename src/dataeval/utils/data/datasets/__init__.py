"""Provides access to common Computer Vision datasets."""

from dataeval.utils.data.datasets._ic import CIFAR10, MNIST, ShipDataset
from dataeval.utils.data.datasets._od import VOCDetection

__all__ = ["MNIST", "ShipDataset", "CIFAR10", "VOCDetection"]
