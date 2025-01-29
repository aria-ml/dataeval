"""Provides utility functions for interacting with Computer Vision datasets."""

__all__ = ["datasets", "read_dataset", "SplitDatasetOutput", "split_dataset"]

from dataeval.utils.data import datasets
from dataeval.utils.data._read import read_dataset
from dataeval.utils.data._split import SplitDatasetOutput, split_dataset
