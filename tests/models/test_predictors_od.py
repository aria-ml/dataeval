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


def test_missing_output_raises(stub_model_file, fake_onnxruntime, tmp_path: Path):
    # boxes but no scores -- the pair is checked before either is decoded
    fake_onnxruntime({"boxes": np.zeros((1, 5, 4), dtype=np.float32)})
    model = OnnxObjectDetector(stub_model_file("detector.onnx"), _meta(tmp_path))
    with pytest.raises(ValueError, match="not found"):
        model([np.zeros((3, 8, 8), dtype=np.uint8)])


def test_bad_boxes_shape_raises(stub_model_file, fake_onnxruntime, tmp_path: Path):
    # boxes last dim is 3, not 4
    fake_onnxruntime({
        "boxes": np.zeros((1, 5, 3), dtype=np.float32),
        "scores": np.zeros((1, 5, 4), dtype=np.float32),
    })
    model = OnnxObjectDetector(stub_model_file("detector.onnx"), _meta(tmp_path))
    with pytest.raises(ValueError, match=r"boxes must be \(B, nBoxes, 4\)"):
        model([np.zeros((3, 8, 8), dtype=np.uint8)])


def test_non_3d_scores_raises(stub_model_file, fake_onnxruntime, tmp_path: Path):
    fake_onnxruntime({
        "boxes": np.zeros((1, 5, 4), dtype=np.float32),
        "scores": np.zeros((1, 5), dtype=np.float32),  # 2-D
    })
    model = OnnxObjectDetector(stub_model_file("detector.onnx"), _meta(tmp_path))
    with pytest.raises(ValueError, match=r"scores must be \(B, nBoxes, nClasses\)"):
        model([np.zeros((3, 8, 8), dtype=np.uint8)])


def test_litert_detector_make_backend_loads_tflite(tmp_path: Path):
    # the LiteRT subclass reaches its backend, which rejects the missing model file
    with pytest.raises(FileNotFoundError):
        LiteRtObjectDetector(tmp_path / "missing.tflite", _meta(tmp_path))


def test_detector_runs_with_stubbed_runtime(stub_model_file, fake_onnxruntime, tmp_path: Path):
    """The full construct-preprocess-decode path, without needing a real ONNX runtime."""
    scores = np.tile(np.array([0.1, 0.2, 0.6, 0.1], dtype=np.float32), (5, 1))
    fake_onnxruntime({
        "boxes": lambda t: np.zeros((t.shape[0], 5, 4), dtype=np.float32),
        "scores": lambda t: np.tile(scores, (t.shape[0], 1, 1)),
    })

    model = OnnxObjectDetector(stub_model_file("detector.onnx"), _meta(tmp_path))
    assert model.metadata["id"] == "dataeval-onnx-detector:detector.onnx"

    preds = model([np.zeros((3, 8, 8), dtype=np.uint8), np.full((3, 8, 8), 255, dtype=np.uint8)])
    assert len(preds) == 2
    t = preds[0]
    assert isinstance(t, ObjectDetectionTarget)
    assert np.asarray(t.boxes).shape == (5, 4)
    assert np.asarray(t.scores).shape == (5, 4)
    # labels are the argmax over each detection's class scores
    np.testing.assert_array_equal(np.asarray(t.labels), np.full(5, 2))


def test_detector_honors_image_size_override(stub_model_file, fake_onnxruntime, tmp_path: Path):
    """An explicit image_size resizes the batch instead of the size declared in metadata."""
    session = fake_onnxruntime({
        "boxes": lambda t: np.zeros((t.shape[0], 5, 4), dtype=np.float32),
        "scores": lambda t: np.zeros((t.shape[0], 5, 4), dtype=np.float32),
    })

    model = OnnxObjectDetector(stub_model_file("detector.onnx"), _meta(tmp_path), image_size=(4, 4))
    model([np.zeros((3, 8, 8), dtype=np.uint8)])

    assert session.last_feed is not None
    assert session.last_feed["image"].shape == (1, 3, 4, 4)
