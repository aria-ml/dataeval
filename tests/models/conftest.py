"""Shared fixtures: tiny ONNX classifier + detector exported via torch."""

import warnings
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _require_onnx_runtime() -> None:
    """Skip the calling test when the ONNX toolchain is missing.

    Deliberately not at module scope: a module-level ``importorskip`` in a conftest skips
    every test in that conftest's directory, and tests/extractors/conftest.py imports this
    module for its two fixtures -- which spread the skip across a second tree whose tests
    have nothing to do with ONNX. Guarding inside the fixtures keeps it to the tests that
    actually build an ONNX model.
    """
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")


try:
    from torch import Tensor
except ImportError:
    from typing import Any as Tensor


class _TinyClassifier(torch.nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.n_classes = n_classes

    def forward(self, x: Tensor) -> Tensor:  # x: (B,3,H,W)
        pooled = x.flatten(1).mean(dim=1, keepdim=True)  # (B,1)
        logits = pooled.repeat(1, self.n_classes) * torch.arange(self.n_classes).float()
        return torch.softmax(logits, dim=1)  # (B, n_classes) in [0,1]


class _TinyDetector(torch.nn.Module):
    def __init__(self, n_boxes: int, n_classes: int) -> None:
        super().__init__()
        self.n_boxes = n_boxes
        self.n_classes = n_classes

    def forward(self, x: Tensor):  # x: (B,3,H,W)
        b = x.shape[0]
        seed = x.flatten(1).mean(dim=1).reshape(b, 1, 1)  # (B,1,1)
        boxes = torch.sigmoid(seed.repeat(1, self.n_boxes, 4))  # (B,nBoxes,4) in [0,1]
        scores = torch.softmax(seed.repeat(1, self.n_boxes, self.n_classes), dim=2)
        return boxes, scores


@pytest.fixture
def onnx_classifier(tmp_path: Path) -> Path:
    _require_onnx_runtime()
    path = tmp_path / "classifier.onnx"
    model = _TinyClassifier(n_classes=4).eval()
    dummy = torch.zeros(1, 3, 8, 8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
            dummy,
            str(path),
            input_names=["image"],
            output_names=["scores"],
            dynamic_axes={"image": {0: "batch"}, "scores": {0: "batch"}},
            opset_version=13,
        )
    return path


@pytest.fixture
def onnx_detector(tmp_path: Path) -> Path:
    _require_onnx_runtime()
    path = tmp_path / "detector.onnx"
    model = _TinyDetector(n_boxes=5, n_classes=4).eval()
    dummy = torch.zeros(1, 3, 8, 8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
            dummy,
            str(path),
            input_names=["image"],
            output_names=["boxes", "scores"],
            dynamic_axes={"image": {0: "batch"}, "boxes": {0: "batch"}, "scores": {0: "batch"}},
            opset_version=13,
        )
    return path
