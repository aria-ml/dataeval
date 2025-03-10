"""Provides utility functions for interacting with Computer Vision datasets."""

__all__ = [
    "datasets",
    "Embeddings",
    "Images",
    "Metadata",
    "SplitDatasetOutput",
    "Targets",
    "split_dataset",
]

from dataeval.utils.data._embeddings import Embeddings
from dataeval.utils.data._images import Images
from dataeval.utils.data._metadata import Metadata
from dataeval.utils.data._split import SplitDatasetOutput, split_dataset
from dataeval.utils.data._targets import Targets

from . import datasets
