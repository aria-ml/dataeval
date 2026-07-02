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


class _DetectionPrediction:
    def __init__(self, scores):
        self.scores = scores


class _FakeModel:
    """A minimal model returning fixed predictions (no ONNX)."""

    def __init__(self, predictions):
        self._predictions = predictions

    def __call__(self, images):
        return self._predictions


def test_repr_renders_model_class_name():
    assert repr(ScoresExtractor(_FakeModel([]))) == "ScoresExtractor(model=_FakeModel)"


def test_detection_one_dimensional_scores_get_a_class_axis():
    # a single detection with a 1-D score vector -> one row of that width
    preds = [_DetectionPrediction(np.array([0.1, 0.9, 0.2]))]
    out = ScoresExtractor(_FakeModel(preds))([np.zeros((3, 8, 8), dtype=np.uint8)])
    assert out.shape == (3, 1)


def test_detection_two_dimensional_scores_flatten_per_detection():
    # a single image with two detections, each a 4-class score row -> two output rows
    preds = [_DetectionPrediction(np.array([[0.1, 0.9, 0.0, 0.0], [0.2, 0.2, 0.5, 0.1]]))]
    out = ScoresExtractor(_FakeModel(preds))([np.zeros((3, 8, 8), dtype=np.uint8)])
    assert out.shape == (2, 4)


def test_empty_predictions_return_empty():
    out = ScoresExtractor(_FakeModel([]))([np.zeros((3, 8, 8), dtype=np.uint8)])
    assert out.shape == (0, 0)
