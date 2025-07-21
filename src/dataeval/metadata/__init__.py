"""Explanatory functions using metadata and additional features such as ood or drift"""

__all__ = ["find_ood_predictors", "find_most_deviated_factors"]

from dataeval.metadata._ood import find_most_deviated_factors, find_ood_predictors
