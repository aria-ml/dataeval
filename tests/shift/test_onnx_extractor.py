"""Tests for OnnxExtractor batching."""

import pytest

from dataeval.extractors import OnnxExtractor


@pytest.mark.required
def test_onnx_extractor_has_batch_size_property():
    """batch_size is stored and exposed without needing onnxruntime loaded."""
    ex = OnnxExtractor("nonexistent.onnx", batch_size=8)
    assert ex.batch_size == 8
    assert OnnxExtractor("nonexistent.onnx").batch_size is None
