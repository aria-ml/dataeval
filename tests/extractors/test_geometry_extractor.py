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
