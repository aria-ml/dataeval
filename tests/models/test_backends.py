"""Tests for runtime backends."""

from pathlib import Path

import numpy as np
import pytest

from dataeval.models._backends import LiteRtBackend, OnnxBackend, make_backend


def test_onnx_backend_runs_and_returns_named_outputs(onnx_classifier: Path):
    backend = OnnxBackend(onnx_classifier)
    tensor = np.zeros((2, 3, 8, 8), dtype=np.float32)
    out = backend.run(tensor)
    assert "scores" in out
    assert out["scores"].shape == (2, 4)
    assert out["scores"].dtype == np.float32


def test_onnx_backend_detector_outputs(onnx_detector: Path):
    backend = OnnxBackend(onnx_detector)
    out = backend.run(np.zeros((1, 3, 8, 8), dtype=np.float32))
    assert out["boxes"].shape == (1, 5, 4)
    assert out["scores"].shape == (1, 5, 4)


def test_onnx_backend_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        OnnxBackend("/nonexistent/model.onnx")


def test_make_backend_unknown_extension_raises():
    with pytest.raises(ValueError, match="unsupported model extension"):
        make_backend("model.pt")


def test_litert_backend_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        LiteRtBackend(tmp_path / "model.tflite")
