"""
Embedding encoders for extracting features from datasets.

This module provides pluggable encoder implementations for different backends.
"""

__all__ = ["NumpyFlattenEncoder", "OnnxEncoder", "TorchEmbeddingEncoder"]

from dataeval.encoders._numpy import NumpyFlattenEncoder
from dataeval.encoders._onnx import OnnxEncoder
from dataeval.encoders._torch import TorchEmbeddingEncoder
