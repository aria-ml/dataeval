"""Tests for the opinionated object-detection predictor."""

import json
from pathlib import Path

import numpy as np
import pytest

from dataeval.models import OnnxObjectDetector
from dataeval.models._predictors import _BaseObjectDetector
from dataeval.protocols import ObjectDetectionTarget


def _meta(tmp_path: Path) -> Path:
    p = tmp_path / "model-metadata.json"
    p.write_text(
        json.dumps({
            "interface": {"name": "JATIC_ONNX", "version": "v1"},
            "io": {
                "batchSize": -1,
                "interface": "IMAGE_OBJECT_DETECTION",
                "input": {"channels": "RGB", "height": 8, "width": 8},
                "output": {"nBoxes": 5, "nClasses": 4},
            },
        }),
        encoding="utf-8",
    )
    return p


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
