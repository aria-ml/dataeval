"""Verify that DataEval installs correctly via pip/uv and core functionality is available.

Maps to meta repo test cases:
  - TC-1.1: Python version compatibility
  - TC-1.7: Package manager installation
"""

import pytest


@pytest.mark.test_case("1-1")
@pytest.mark.test_case("1-7")
class TestPipInstall:
    """Verify the package is importable and functional after pip installation."""

    def test_import_dataeval(self):
        import dataeval  # noqa: F401

    def test_version_is_set(self):
        from dataeval import __version__

        assert __version__ != "unknown"

    def test_core_modules_importable(self):
        from dataeval import config, flags, protocols, types  # noqa: F401

    def test_subpackages_importable(self):
        from dataeval import bias, core, data, quality, shift  # noqa: F401

    def test_basic_smoke_test(self):
        """Smoke test: a simple end-to-end calculation completes without error."""
        import numpy as np

        from dataeval.core import label_stats

        labels = np.array([0, 0, 1, 1, 2, 2])
        result = label_stats(labels)
        assert result is not None

    def test_optional_dependencies_graceful_degradation_bovw(self, monkeypatch):
        """BoVWExtractor degrades gracefully with actionable ImportError when cv2 is missing."""
        import sys

        monkeypatch.setitem(sys.modules, "cv2", None)
        from dataeval.extractors import BoVWExtractor

        with pytest.raises(ImportError, match=r"BoVWExtractor requires 'opencv-python'"):
            BoVWExtractor()

    def test_optional_dependencies_graceful_degradation_onnx(self, monkeypatch):
        """OnnxExtractor degrades gracefully with actionable ImportError when onnxruntime is missing."""
        import sys

        monkeypatch.setitem(sys.modules, "onnxruntime", None)
        from dataeval.extractors._onnx import _get_ort

        with pytest.raises(ImportError, match=r"onnxruntime is required for OnnxExtractor"):
            _get_ort()

    def test_optional_dependencies_graceful_degradation_torchvision(self, monkeypatch):
        """TorchvisionTransform degrades gracefully with actionable ImportError when torchvision is missing."""
        import sys

        monkeypatch.setitem(sys.modules, "torchvision", None)
        from dataeval.data._torchvision import _import_torchvision

        with pytest.raises(ImportError, match=r"TorchvisionTransform requires torchvision"):
            _import_torchvision()

    def test_optional_dependencies_graceful_degradation_ontology(self, monkeypatch):
        """Ontology.from_rdf degrades gracefully with actionable ImportError when rdflib is missing."""
        import sys

        monkeypatch.setitem(sys.modules, "rdflib", None)
        from dataeval._ontology import Ontology

        with pytest.raises(ImportError, match=r"Ontology\.from_rdf requires the optional 'rdflib' dependency"):
            Ontology.from_rdf("<rdf/>")

    def test_optional_dependencies_graceful_degradation_onnx_utils(self, monkeypatch):
        """ONNX utilities degrade gracefully with actionable ImportError when onnx is missing."""
        import sys

        monkeypatch.setitem(sys.modules, "onnx", None)
        from dataeval.utils import onnx

        with pytest.raises(ImportError, match=r"onnx is required for ONNX model utilities"):
            onnx.to_encoding_model(b"dummy")

    def test_optional_dependencies_graceful_degradation_litert(self, monkeypatch, tmp_path):
        """LiteRtBackend degrades gracefully with an ImportError naming the `litert` extra."""
        import sys

        from dataeval.models._backends import _LITERT_MODULES, LiteRtBackend

        for module_name in _LITERT_MODULES:
            monkeypatch.setitem(sys.modules, module_name, None)
        model_path = tmp_path / "model.tflite"
        model_path.write_bytes(b"")
        with pytest.raises(ImportError, match=r"pip install 'dataeval\[litert\]'"):
            LiteRtBackend(model_path)  # type: ignore
