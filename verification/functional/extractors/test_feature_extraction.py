"""Verify that feature extractors produce embeddings of the correct shape.

Maps to meta repo test cases:
  - TC-7.1: Feature extraction (FlattenExtractor and optional backends)
"""

import numpy as np
import pytest


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

    def test_torch_extractor_named_layer_embeddings(self):
        """TorchExtractor returns one row per image from the requested layer."""
        torch = pytest.importorskip("torch")
        from dataeval.extractors import TorchExtractor

        model = torch.nn.Sequential(
            torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 5)
        )
        images = np.random.default_rng(0).random((10, 3, 8, 8)).astype(np.float32)

        hidden = TorchExtractor(model, device="cpu", layer_name="1")(images)
        output = TorchExtractor(model, device="cpu")(images)

        assert np.asarray(hidden).shape == (10, 16)  # the Linear(192, 16) layer, not the final output
        assert np.asarray(output).shape == (10, 5)

    def test_bovw_extractor_one_vector_per_image(self):
        pytest.importorskip("cv2")
        from dataeval.extractors import BoVWExtractor

        images = (np.random.default_rng(0).random((6, 3, 64, 64)) * 255).astype(np.uint8)
        extractor = BoVWExtractor(vocab_size=8).fit(images)
        result = np.asarray(extractor(images))
        assert result.shape == (6, 8)
        assert np.isfinite(result).all()

    def test_uncertainty_extractors_with_stub_model(self):
        from dataeval.extractors import ClasswiseUncertaintyExtractor, UncertaintyExtractor

        class StubScores:
            """Stand-in model producing fixed (n, n_classes) logits."""

            def __call__(self, data):
                return np.random.default_rng(1).standard_normal((10, 4)).astype(np.float32)

        per_instance = UncertaintyExtractor(StubScores())(None)
        assert per_instance.shape == (10, 1)
        assert np.isfinite(per_instance).all()

        per_class = ClasswiseUncertaintyExtractor(StubScores())(None)
        assert isinstance(per_class, dict)
        assert set(per_class) <= {0, 1, 2, 3}
        assert all(v.ndim == 2 and v.shape[1] == 1 for v in per_class.values())
        assert sum(len(v) for v in per_class.values()) >= 10  # every detection lands in at least one class
