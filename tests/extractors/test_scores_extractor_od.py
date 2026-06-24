"""Detection predictions flatten to per-detection rows and feed classwise uncertainty."""

import json
from pathlib import Path

import numpy as np

from dataeval.extractors import ClasswiseUncertaintyExtractor, ScoresExtractor
from dataeval.models import OnnxObjectDetector


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


def test_detection_scores_flatten_per_detection(onnx_detector: Path, tmp_path: Path):
    model = OnnxObjectDetector(onnx_detector, _meta(tmp_path))
    ext = ScoresExtractor(model)
    out = np.asarray(ext([np.zeros((3, 8, 8), dtype=np.uint8)] * 2))
    # 2 images x 5 boxes = 10 detections, 4 classes
    assert out.shape == (10, 4)


def test_detection_scores_feed_classwise_uncertainty(onnx_detector: Path, tmp_path: Path):
    model = OnnxObjectDetector(onnx_detector, _meta(tmp_path))
    unc = ClasswiseUncertaintyExtractor(ScoresExtractor(model), preds_type="probs")
    result = unc([np.zeros((3, 8, 8), dtype=np.uint8)] * 2)
    assert isinstance(result, dict)
    assert len(result) > 0
    for arr in result.values():
        assert np.asarray(arr).ndim == 2
