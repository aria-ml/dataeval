"""Verify that feature extractors produce embeddings of the correct shape.

Maps to meta repo test cases:
  - TC-7.1: Feature extraction (FlattenExtractor and optional backends)
"""

import numpy as np
import pytest


@pytest.mark.test_case("7-1")
class TestFeatureExtraction:
    """Verify FlattenExtractor and optional extractor availability."""

    def test_flatten_extractor_produces_embeddings(self):
        from dataeval.extractors import FlattenExtractor

        extractor = FlattenExtractor()
        images = np.random.default_rng(0).random((10, 3, 16, 16)).astype(np.float32)
        result = extractor(images)
        assert result is not None
        assert result.shape[0] == 10

    def test_flatten_extractor_flattens_to_1d_per_image(self):
        from dataeval.extractors import FlattenExtractor

        extractor = FlattenExtractor()
        images = np.random.default_rng(0).random((5, 3, 8, 8)).astype(np.float32)
        result = extractor(images)
        assert result.shape == (5, 3 * 8 * 8)

    def test_torch_extractor_importable(self):
        pytest.importorskip("torch")
        from dataeval.extractors import TorchExtractor  # noqa: F401

    def test_onnx_extractor_importable(self):
        pytest.importorskip("onnxruntime")
        from dataeval.extractors import OnnxExtractor  # noqa: F401

    def test_bovw_extractor_importable(self):
        pytest.importorskip("cv2")
        from dataeval.extractors import BoVWExtractor  # noqa: F401

    def test_all_extractors_listed_in_module(self):
        from dataeval import extractors

        expected = {"FlattenExtractor", "TorchExtractor", "OnnxExtractor", "BoVWExtractor"}
        available = {name for name in dir(extractors) if name.endswith("Extractor")}
        assert expected.issubset(available)
