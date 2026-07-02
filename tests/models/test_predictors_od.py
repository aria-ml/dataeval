"""Tests for the opinionated object-detection predictor."""

import json
from pathlib import Path

import numpy as np
import pytest

from dataeval.models import LiteRtObjectDetector, OnnxObjectDetector
from dataeval.models._predictors import _BaseObjectDetector
from dataeval.protocols import ObjectDetectionTarget


def _meta(tmp_path: Path, interface: str = "IMAGE_OBJECT_DETECTION") -> Path:
    p = tmp_path / "model-metadata.json"
    p.write_text(
        json.dumps({
            "interface": {"name": "JATIC_ONNX", "version": "v1"},
            "io": {
                "batchSize": -1,
                "interface": interface,
                "input": {"channels": "RGB", "height": 8, "width": 8},
                "output": {"nBoxes": 5, "nClasses": 4},
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


def test_detector_returns_targets_per_image(onnx_detector: Path, tmp_path: Path):
    model = OnnxObjectDetector(onnx_detector, _meta(tmp_path))
    preds = model([np.zeros((3, 8, 8), dtype=np.uint8), np.full((3, 8, 8), 255, dtype=np.uint8)])
    assert len(preds) == 2
    t = preds[0]
    assert isinstance(t, ObjectDetectionTarget)
    assert np.asarray(t.boxes).shape == (5, 4)
    assert np.asarray(t.labels).shape == (5,)
    assert np.asarray(t.scores).shape == (5, 4)


def test_detector_boxes_are_normalized(onnx_detector: Path, tmp_path: Path):
    model = OnnxObjectDetector(onnx_detector, _meta(tmp_path))
    t = model([np.zeros((3, 8, 8), dtype=np.uint8)])[0]
    boxes = np.asarray(t.boxes)
    assert boxes.min() >= 0.0
    assert boxes.max() <= 1.0


def test_onnx_detector_is_a_base_object_detector(onnx_detector: Path, tmp_path: Path):
    model = OnnxObjectDetector(onnx_detector, _meta(tmp_path))
    assert isinstance(model, _BaseObjectDetector)


def test_base_object_detector_is_abstract(tmp_path: Path):
    with pytest.raises(TypeError):
        _BaseObjectDetector("model.onnx", _meta(tmp_path))  # type: ignore[abstract]


def test_wrong_task_metadata_raises(tmp_path: Path):
    meta = _meta(tmp_path, interface="IMAGE_CLASSIFICATION")
    with pytest.raises(ValueError, match="not IMAGE_OBJECT_DETECTION"):
        OnnxObjectDetector("unused.onnx", meta)


def test_missing_output_raises(onnx_detector: Path, tmp_path: Path):
    model = OnnxObjectDetector(onnx_detector, _meta(tmp_path))
    model._backend = _FakeBackend({"boxes": np.zeros((1, 5, 4), dtype=np.float32)})  # type: ignore # no "scores"
    with pytest.raises(ValueError, match="not found"):
        model([np.zeros((3, 8, 8), dtype=np.uint8)])


def test_bad_boxes_shape_raises(onnx_detector: Path, tmp_path: Path):
    model = OnnxObjectDetector(onnx_detector, _meta(tmp_path))
    # boxes last dim is 3, not 4
    model._backend = _FakeBackend({
        "boxes": np.zeros((1, 5, 3), dtype=np.float32),
        "scores": np.zeros((1, 5, 4), dtype=np.float32),
    })  # type: ignore
    with pytest.raises(ValueError, match=r"boxes must be \(B, nBoxes, 4\)"):
        model([np.zeros((3, 8, 8), dtype=np.uint8)])


def test_non_3d_scores_raises(onnx_detector: Path, tmp_path: Path):
    model = OnnxObjectDetector(onnx_detector, _meta(tmp_path))
    model._backend = _FakeBackend({
        "boxes": np.zeros((1, 5, 4), dtype=np.float32),
        "scores": np.zeros((1, 5), dtype=np.float32),  # 2-D
    })  # type: ignore
    with pytest.raises(ValueError, match=r"scores must be \(B, nBoxes, nClasses\)"):
        model([np.zeros((3, 8, 8), dtype=np.uint8)])


def test_litert_detector_make_backend_loads_tflite(tmp_path: Path):
    # the LiteRT subclass reaches its backend, which rejects the missing model file
    with pytest.raises(FileNotFoundError):
        LiteRtObjectDetector(tmp_path / "missing.tflite", _meta(tmp_path))
