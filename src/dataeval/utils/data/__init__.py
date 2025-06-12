"""Provides access to common Computer Vision datasets."""

from dataeval.utils.data import collate, metadata
from dataeval.utils.data._dataset import to_image_classification_dataset, to_object_detection_dataset
from dataeval.utils.data._validate import validate_dataset

__all__ = [
    "collate",
    "metadata",
    "to_image_classification_dataset",
    "to_object_detection_dataset",
    "validate_dataset",
]
