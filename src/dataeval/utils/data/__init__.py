"""Provides utility functions for interacting with Computer Vision datasets."""

__all__ = [
    "collate",
    "datasets",
    "Embeddings",
    "Images",
    "Metadata",
    "Select",
    "SplitDatasetOutput",
    "Targets",
    "split_dataset",
    "to_image_classification_dataset",
    "to_object_detection_dataset",
]

from dataeval.outputs._utils import SplitDatasetOutput
from dataeval.utils.data._dataset import to_image_classification_dataset, to_object_detection_dataset
from dataeval.utils.data._embeddings import Embeddings
from dataeval.utils.data._images import Images
from dataeval.utils.data._metadata import Metadata
from dataeval.utils.data._selection import Select
from dataeval.utils.data._split import split_dataset
from dataeval.utils.data._targets import Targets

from . import collate, datasets
