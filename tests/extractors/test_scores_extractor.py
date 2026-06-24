"""Tests for the Model -> scores FeatureExtractor adapter."""

import json
from pathlib import Path

import numpy as np

from dataeval.extractors import ScoresExtractor, UncertaintyExtractor
from dataeval.models import OnnxImageClassifier
from dataeval.protocols import FeatureExtractor


def _meta(tmp_path: Path) -> Path:
    p = tmp_path / "model-metadata.json"
    p.write_text(
        json.dumps({
            "interface": {"name": "JATIC_ONNX", "version": "v1"},
            "io": {
                "batchSize": -1,
                "interface": "IMAGE_CLASSIFICATION",
                "input": {"channels": "RGB", "height": 8, "width": 8},
                "output": {"nClasses": 4},
            },
        }),
        encoding="utf-8",
    )
    return p


def test_scores_extractor_stacks_classification_scores(onnx_classifier: Path, tmp_path: Path):
    model = OnnxImageClassifier(onnx_classifier, _meta(tmp_path))
    ext = ScoresExtractor(model)
    assert isinstance(ext, FeatureExtractor)
    out = np.asarray(ext([np.zeros((3, 8, 8), dtype=np.uint8)] * 3))
    assert out.shape == (3, 4)


def test_scores_extractor_feeds_uncertainty(onnx_classifier: Path, tmp_path: Path):
    model = OnnxImageClassifier(onnx_classifier, _meta(tmp_path))
    unc = UncertaintyExtractor(ScoresExtractor(model), preds_type="probs")
    out = np.asarray(unc([np.zeros((3, 8, 8), dtype=np.uint8)] * 2))
    assert out.shape == (2, 1)
