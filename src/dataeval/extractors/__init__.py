"""
Feature extractors for drift detection and quality metrics.

This module provides feature extraction implementations that convert various
data types into arrays suitable for drift detection and quality analysis.

The primary feature extractors are :class:`~dataeval.Embeddings` and
:class:`~dataeval.Metadata`, which implement the :class:`~dataeval.protocols.FeatureExtractor`
protocol and can be used directly with drift detectors.

This submodule provides additional specialized extractors:

- :class:`BoVWExtractor`: Computes Bag of Visual Words using SIFT for feature comparison
- :class:`UncertaintyFeatureExtractor`: Computes model prediction uncertainty (entropy)
"""

__all__ = ["BoVWExtractor", "UncertaintyFeatureExtractor"]

from dataeval.extractors._bovw import BoVWExtractor
from dataeval.extractors._uncertainty import UncertaintyFeatureExtractor
