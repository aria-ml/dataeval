"""Tests for model-metadata.json parsing (IR-3.1)."""

import json
from pathlib import Path

import pytest

from dataeval.models import ModelIOSpec, read_model_metadata


def _write(tmp_path: Path, payload: dict) -> Path:
    p = tmp_path / "model-metadata.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_reads_classification_metadata(tmp_path: Path):
    p = _write(
        tmp_path,
        {
            "interface": {"name": "JATIC_ONNX", "version": "v1"},
            "io": {
                "batchSize": 5,
                "interface": "IMAGE_CLASSIFICATION",
                "input": {"channels": "RGB", "height": -1, "width": 500},
                "output": {"nClasses": 10},
            },
        },
    )
    spec = read_model_metadata(p)
    assert isinstance(spec, ModelIOSpec)
    assert spec.task == "IMAGE_CLASSIFICATION"
    assert spec.channels == "RGB"
    assert spec.height == -1
    assert spec.width == 500
    assert spec.batch_size == 5
    assert spec.n_classes == 10
    assert spec.n_boxes is None


def test_reads_detection_metadata(tmp_path: Path):
    p = _write(
        tmp_path,
        {
            "interface": {"name": "JATIC_ONNX", "version": "v1"},
            "io": {
                "batchSize": -1,
                "interface": "IMAGE_OBJECT_DETECTION",
                "input": {"channels": "GRAYSCALE", "height": -1, "width": -1},
                "output": {"nBoxes": 100, "nClasses": 10},
            },
        },
    )
    spec = read_model_metadata(p)
    assert spec.task == "IMAGE_OBJECT_DETECTION"
    assert spec.n_boxes == 100
    assert spec.n_classes == 10


def test_missing_field_raises_naming_field(tmp_path: Path):
    p = _write(
        tmp_path,
        {
            "interface": {"name": "JATIC_ONNX"},
            "io": {"interface": "IMAGE_CLASSIFICATION", "batchSize": 1, "input": {}, "output": {"nClasses": 10}},
        },
    )
    with pytest.raises(ValueError, match="channels"):
        read_model_metadata(p)


def test_unknown_task_raises(tmp_path: Path):
    p = _write(
        tmp_path,
        {
            "io": {
                "interface": "SEGMENTATION",
                "batchSize": 1,
                "input": {"channels": "RGB", "height": 1, "width": 1},
                "output": {"nClasses": 1},
            }
        },
    )
    with pytest.raises(ValueError, match="interface"):
        read_model_metadata(p)
