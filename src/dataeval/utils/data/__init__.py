"""Provides access to common Computer Vision datasets."""

from dataeval.utils.data import collate
from dataeval.utils.data._merge import flatten, merge
from dataeval.utils.data._validate import validate_dataset

__all__ = [
    "collate",
    "flatten",
    "merge",
    "validate_dataset",
]
