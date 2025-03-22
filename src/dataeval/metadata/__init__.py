"""Explanatory functions using metadata and additional features such as ood or drift"""

__all__ = ["most_deviated_factors", "metadata_distance"]

from dataeval.metadata._distance import metadata_distance
from dataeval.metadata._ood import most_deviated_factors
