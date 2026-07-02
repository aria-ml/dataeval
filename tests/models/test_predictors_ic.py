"""Tests for the opinionated image-classification predictor."""

import json
from pathlib import Path

import numpy as np
import pytest

from dataeval.models import LiteRtImageClassifier, OnnxImageClassifier
from dataeval.models._predictors import _BaseImageClassifier


def _meta(tmp_path: Path, n_classes: int = 4, interface: str = "IMAGE_CLASSIFICATION") -> Path:
    p = tmp_path / "model-metadata.json"
    p.write_text(
        json.dumps({
            "interface": {"name": "JATIC_ONNX", "version": "v1"},
            "io": {
                "batchSize": -1,
                "interface": interface,
                "input": {"channels": "RGB", "height": 8, "width": 8},
                "output": {"nClasses": n_classes},
            },
        }),
        encoding="utf-8",
    )
    return p


class _FakeBackend:
    """A runtime backend returning fixed outputs, bypassing real inference."""

    def __init__(self, outputs: dict) -> None:
        self._outputs = outputs

    def run(self, tensor) -> dict:
        return self._outputs


def test_classifier_returns_per_image_scores(onnx_classifier: Path, tmp_path: Path):
    model = OnnxImageClassifier(onnx_classifier, _meta(tmp_path))
    batch = [np.zeros((3, 16, 16), dtype=np.uint8), np.full((3, 16, 16), 255, dtype=np.uint8)]
    preds = model(batch)
    assert len(preds) == 2
    assert np.asarray(preds[0]).shape == (4,)
    assert np.all(np.asarray(preds[0]) >= 0)
    assert np.all(np.asarray(preds[0]) <= 1)


def test_classifier_has_maite_metadata(onnx_classifier: Path, tmp_path: Path):
    model = OnnxImageClassifier(onnx_classifier, _meta(tmp_path))
    assert "id" in model.metadata
    assert isinstance(model.metadata["id"], str)


def test_onnx_classifier_is_a_base_image_classifier(onnx_classifier: Path, tmp_path: Path):
    model = OnnxImageClassifier(onnx_classifier, _meta(tmp_path))
    assert isinstance(model, _BaseImageClassifier)


def test_base_image_classifier_is_abstract(tmp_path: Path):
    with pytest.raises(TypeError):
        _BaseImageClassifier("model.onnx", _meta(tmp_path))  # type: ignore[abstract]


def test_wrong_task_metadata_raises(tmp_path: Path):
    meta = _meta(tmp_path, interface="IMAGE_OBJECT_DETECTION")
    with pytest.raises(ValueError, match="not IMAGE_CLASSIFICATION"):
        OnnxImageClassifier("unused.onnx", meta)


def test_missing_scores_output_raises(onnx_classifier: Path, tmp_path: Path):
    model = OnnxImageClassifier(onnx_classifier, _meta(tmp_path))
    model._backend = _FakeBackend({})  # type: ignore # no "scores" key
    with pytest.raises(ValueError, match="not found"):
        model([np.zeros((3, 8, 8), dtype=np.uint8)])


def test_non_2d_scores_raises(onnx_classifier: Path, tmp_path: Path):
    model = OnnxImageClassifier(onnx_classifier, _meta(tmp_path))
    model._backend = _FakeBackend({"scores": np.zeros((2, 3, 4), dtype=np.float32)})  # type: ignore # 3-D
    with pytest.raises(ValueError, match="must be 2-D"):
        model([np.zeros((3, 8, 8), dtype=np.uint8)])


def test_litert_classifier_make_backend_loads_tflite(tmp_path: Path):
    # the LiteRT subclass reaches its backend, which rejects the missing model file
    with pytest.raises(FileNotFoundError):
        LiteRtImageClassifier(tmp_path / "missing.tflite", _meta(tmp_path))
