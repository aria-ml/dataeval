"""Provides utility functions for interacting with Computer Vision datasets."""

__all__ = ["datasets", "collate", "SplitDatasetOutput", "split_dataset"]

from dataeval.utils.data import datasets
from dataeval.utils.data._collate import collate
from dataeval.utils.data._split import SplitDatasetOutput, split_dataset
