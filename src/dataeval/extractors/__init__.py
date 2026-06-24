"""
Feature extractors that transform input data into arrays.

All extractors implement the :class:`~dataeval.protocols.FeatureExtractor` protocol
(``__call__(data) -> Array``) and can be used standalone or passed to
:class:`~dataeval.Embeddings` for batching, caching, and memory-mapped storage.
"""

__all__ = [
    "DetectionGeometryExtractor",
    "FlattenExtractor",
    "OnnxExtractor",
    "TorchExtractor",
    "BoVWExtractor",
    "UncertaintyExtractor",
    "ClasswiseUncertaintyExtractor",
    "ClassifierUncertaintyExtractor",
    "ScoresExtractor",
]

from dataeval.extractors._bovw import BoVWExtractor
from dataeval.extractors._flatten import FlattenExtractor
from dataeval.extractors._geometry import DetectionGeometryExtractor
from dataeval.extractors._onnx import OnnxExtractor
from dataeval.extractors._scores import ScoresExtractor
from dataeval.extractors._torch import TorchExtractor
from dataeval.extractors._uncertainty import (
    ClassifierUncertaintyExtractor,
    ClasswiseUncertaintyExtractor,
    UncertaintyExtractor,
)
