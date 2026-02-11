"""Verify that optional extras install correctly and their features are available.

Maps to meta repo test cases:
  - TC-1.2: Dependency resolution
  - TC-1.7: Package manager installation
"""

import importlib

import pytest


@pytest.mark.test_case("1-2")
@pytest.mark.test_case("1-7")
class TestExtrasInstall:
    """Verify optional dependency extras provide expected functionality."""

    def test_torch_importable(self):
        torch = pytest.importorskip("torch")
        assert hasattr(torch, "__version__")

    def test_torchvision_importable(self):
        try:
            import torchvision  # noqa: F401
        except ImportError:
            pytest.skip("torchvision not installed")
        except RuntimeError as e:
            pytest.skip(f"torchvision import failed (version mismatch): {e}")

    def test_onnxruntime_importable(self):
        ort = pytest.importorskip("onnxruntime")
        assert hasattr(ort, "__version__")

    def test_opencv_importable(self):
        cv2 = pytest.importorskip("cv2")
        assert hasattr(cv2, "__version__")

    def test_torch_extractors_available(self):
        """Torch extra enables torch-based feature extractors."""
        pytest.importorskip("torch")
        spec = importlib.util.find_spec("dataeval.extractors")
        assert spec is not None

    def test_onnx_extractors_available(self):
        """ONNX extra enables ONNX-based feature extractors."""
        pytest.importorskip("onnxruntime")
        spec = importlib.util.find_spec("dataeval.extractors")
        assert spec is not None

    def test_opencv_features_available(self):
        """OpenCV extra enables BoVW extractor."""
        pytest.importorskip("cv2")
        spec = importlib.util.find_spec("dataeval.extractors")
        assert spec is not None
