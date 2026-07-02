"""Tests for per-detection geometry features feeding drift."""

import json
from pathlib import Path

import numpy as np

from dataeval.extractors import DetectionGeometryExtractor
from dataeval.models import OnnxObjectDetector
from dataeval.protocols import FeatureExtractor


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


def test_geometry_extractor_shape(onnx_detector: Path, tmp_path: Path):
    model = OnnxObjectDetector(onnx_detector, _meta(tmp_path))
    ext = DetectionGeometryExtractor(model)
    assert isinstance(ext, FeatureExtractor)
    out = np.asarray(ext([np.zeros((3, 8, 8), dtype=np.uint8)] * 2))
    assert out.shape == (10, 6)  # 2 imgs x 5 boxes, 6 geom features


def test_geometry_features_are_finite(onnx_detector: Path, tmp_path: Path):
    model = OnnxObjectDetector(onnx_detector, _meta(tmp_path))
    out = np.asarray(DetectionGeometryExtractor(model)([np.zeros((3, 8, 8), dtype=np.uint8)]))
    assert np.isfinite(out).all()


def test_geometry_feeds_drift(onnx_detector: Path, tmp_path: Path):
    from dataeval.shift import DriftUnivariate

    model = OnnxObjectDetector(onnx_detector, _meta(tmp_path))
    ext = DetectionGeometryExtractor(model)
    ref = np.asarray(ext([np.zeros((3, 8, 8), dtype=np.uint8)] * 4))
    det = DriftUnivariate().fit(ref)
    result = det.predict(np.asarray(ext([np.full((3, 8, 8), 255, dtype=np.uint8)] * 4)))
    assert result is not None


class _Prediction:
    def __init__(self, boxes, scores):
        self.boxes = boxes
        self.scores = scores


class _FakeDetector:
    """A minimal detection model returning fixed predictions (no ONNX)."""

    def __init__(self, predictions):
        self._predictions = predictions

    def __call__(self, images):
        return self._predictions


def test_repr_renders_model_class_name():
    ext = DetectionGeometryExtractor(_FakeDetector([]), confidence=0.5)
    assert repr(ext) == "DetectionGeometryExtractor(model=_FakeDetector, confidence=0.5)"


def test_low_confidence_detections_are_dropped():
    # two detections; only the first clears the confidence floor
    preds = [_Prediction(np.array([[0, 0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]]), np.array([[0.9, 0.1], [0.2, 0.1]]))]
    ext = DetectionGeometryExtractor(_FakeDetector(preds), confidence=0.5)
    out = ext([np.zeros((3, 8, 8), dtype=np.uint8)])
    assert out.shape == (1, 6)


def test_no_kept_detections_returns_empty():
    preds = [_Prediction(np.zeros((0, 4)), np.zeros((0, 2)))]
    out = DetectionGeometryExtractor(_FakeDetector(preds))([np.zeros((3, 8, 8), dtype=np.uint8)])
    assert out.shape == (0, 6)
