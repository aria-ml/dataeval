"""Tests for the opinionated image-classification predictor."""

import json
from pathlib import Path

import numpy as np
import pytest

from dataeval.models import OnnxImageClassifier
from dataeval.models._predictors import _BaseImageClassifier


def _meta(tmp_path: Path, n_classes: int = 4) -> Path:
    p = tmp_path / "model-metadata.json"
    p.write_text(
        json.dumps({
            "interface": {"name": "JATIC_ONNX", "version": "v1"},
            "io": {
                "batchSize": -1,
                "interface": "IMAGE_CLASSIFICATION",
                "input": {"channels": "RGB", "height": 8, "width": 8},
                "output": {"nClasses": n_classes},
            },
        }),
        encoding="utf-8",
    )
    return p


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
