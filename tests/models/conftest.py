"""Shared fixtures: tiny ONNX classifier + detector exported via torch, plus a stubbed runtime."""

import sys
import types
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
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


class _FakeOnnxSession:
    """Stand-in for ``onnxruntime.InferenceSession``, returning canned outputs.

    Each output is either an array or a callable of the input tensor, so a test can
    return shapes that follow the batch it was handed.
    """

    def __init__(self, outputs: dict[str, Any]) -> None:
        self._outputs = outputs
        self.model_path: str | None = None
        self.providers: list[str] = []
        self.last_feed: dict[str, np.ndarray] | None = None

    def get_inputs(self) -> list[types.SimpleNamespace]:
        return [types.SimpleNamespace(name="image")]

    def get_outputs(self) -> list[types.SimpleNamespace]:
        return [types.SimpleNamespace(name=name) for name in self._outputs]

    def run(self, output_names: list[str], feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.last_feed = feed
        tensor = next(iter(feed.values()))
        outputs = (self._outputs[name] for name in output_names)
        return [np.asarray(out(tensor) if callable(out) else out, dtype=np.float32) for out in outputs]


@pytest.fixture
def fake_onnxruntime(monkeypatch: pytest.MonkeyPatch) -> Callable[..., _FakeOnnxSession]:
    """Install a stub ``onnxruntime`` module and return the session it hands out.

    ``OnnxBackend`` imports onnxruntime inside ``__init__``, so every ONNX path is
    skipped wherever ``dataeval[onnx]`` is missing -- which is every CI job installing
    only the required dependencies. Only the inference call itself belongs to the
    runtime; provider selection, tensor naming, dtype casting, and output decoding are
    ours, so stubbing the module keeps them exercised without the optional dependency.
    The fixtures above still run the same paths against the real runtime wherever it is
    installed.

    Returns a factory taking the model's outputs (keyed by tensor name) and yielding the
    single session every ``InferenceSession`` call resolves to.
    """

    def install(outputs: dict[str, Any], available_providers: list[str] | None = None) -> _FakeOnnxSession:
        session = _FakeOnnxSession(outputs)

        def _inference_session(model_path: str, providers: list[str] | None = None, **kwargs: Any) -> _FakeOnnxSession:
            session.model_path = model_path
            session.providers = list(providers or [])
            return session

        module = types.ModuleType("onnxruntime")
        module.InferenceSession = _inference_session  # type: ignore[attr-defined]
        module.get_available_providers = lambda: list(  # type: ignore[attr-defined]
            available_providers if available_providers is not None else ["CPUExecutionProvider"]
        )
        monkeypatch.setitem(sys.modules, "onnxruntime", module)
        return session

    return install


@pytest.fixture
def stub_model_file(tmp_path: Path) -> Callable[[str], Path]:
    """Write a placeholder model file that only has to exist for a stubbed runtime."""

    def make(name: str = "model.onnx") -> Path:
        path = tmp_path / name
        path.write_bytes(b"stub")
        return path

    return make


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
