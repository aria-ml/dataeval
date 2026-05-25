"""End-to-end verification of ONNX-derived embeddings flowing into a downstream evaluator.

Demonstrates the supported IR-3-S-1 path:

    ONNX model file --> OnnxExtractor --> embeddings --> drift evaluator

Maps to meta repo test cases:
  - TC-7.1: Feature extraction (OnnxExtractor end-to-end)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from verification.helpers import build_simple_onnx_model


@pytest.fixture(scope="module")
def onnx_encoder_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a small ONNX encoder once per module (output_dim=16)."""
    path = tmp_path_factory.mktemp("onnx_e2e") / "encoder.onnx"
    return build_simple_onnx_model(path, output_dim=16)


@pytest.fixture(scope="module")
def onnx_extractor(onnx_encoder_path: Path):
    from dataeval.extractors import OnnxExtractor

    return OnnxExtractor(onnx_encoder_path)


@pytest.mark.test_case("7-1")
class TestOnnxEndToEnd:
    """Verify the full ONNX -> Embeddings -> downstream evaluator path."""

    def test_onnx_extractor_produces_embeddings_matching_model_output(self, tmp_path: Path):
        from dataeval.extractors import OnnxExtractor

        model_path = build_simple_onnx_model(tmp_path / "encoder.onnx", output_dim=32)
        extractor = OnnxExtractor(model_path)

        images = np.random.default_rng(0).random((8, 3, 16, 16)).astype(np.float32)
        embeddings = extractor(images)

        assert embeddings.shape == (8, 32)
        assert isinstance(embeddings, np.ndarray)
        assert np.issubdtype(embeddings.dtype, np.floating)

    @pytest.mark.parametrize("detector_name", ["DriftMMD", "DriftKNeighbors"])
    def test_onnx_embeddings_feed_drift_detector(self, onnx_extractor, detector_name: str):
        """ONNX-derived embeddings must be consumable by a downstream evaluator."""
        import dataeval.shift as shift

        detector_cls = getattr(shift, detector_name)
        kwargs = {"n_permutations": 20} if detector_name == "DriftMMD" else {}

        rng = np.random.default_rng(42)
        ref_embeddings = onnx_extractor(rng.random((40, 3, 16, 16)).astype(np.float32))
        test_embeddings = onnx_extractor(rng.random((40, 3, 16, 16)).astype(np.float32) + 10.0)

        result = detector_cls(**kwargs).fit(ref_embeddings).predict(test_embeddings)

        assert result.drifted is True
