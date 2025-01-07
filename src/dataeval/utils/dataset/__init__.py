"""Provides utility functions for interacting with Computer Vision datasets."""

__all__ = ["datasets", "read_dataset", "SplitDatasetOutput", "split_dataset"]

from dataeval.utils.dataset import datasets
from dataeval.utils.dataset.read import read_dataset
from dataeval.utils.dataset.split import SplitDatasetOutput, split_dataset
