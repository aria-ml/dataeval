"""Tests for runtime backends."""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

import dataeval.models._backends as backends
from dataeval.models._backends import LiteRtBackend, OnnxBackend, make_backend


class _FakeInterpreter:
    """Stand-in for a LiteRT interpreter, avoiding a real tflite runtime."""

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path
        self.set_input: np.ndarray | None = None
        self.resized_shape: tuple[int, ...] | None = None

    def allocate_tensors(self) -> None:
        pass

    def get_input_details(self) -> list[dict]:
        return [{"name": "image", "index": 0}]

    def get_output_details(self) -> list[dict]:
        return [{"name": "scores", "index": 1}]

    def resize_tensor_input(self, index: int, shape: tuple[int, ...]) -> None:
        self.resized_shape = shape

    def set_tensor(self, index: int, tensor: np.ndarray) -> None:
        self.set_input = tensor

    def invoke(self) -> None:
        pass

    def get_tensor(self, index: int) -> np.ndarray:
        return np.ones((1, 4), dtype=np.float32)


def test_onnx_backend_runs_and_returns_named_outputs(onnx_classifier: Path):
    backend = OnnxBackend(onnx_classifier)
    tensor = np.zeros((2, 3, 8, 8), dtype=np.float32)
    out = backend.run(tensor)
    assert "scores" in out
    assert out["scores"].shape == (2, 4)
    assert out["scores"].dtype == np.float32


def test_onnx_backend_detector_outputs(onnx_detector: Path):
    backend = OnnxBackend(onnx_detector)
    out = backend.run(np.zeros((1, 3, 8, 8), dtype=np.float32))
    assert out["boxes"].shape == (1, 5, 4)
    assert out["scores"].shape == (1, 5, 4)


def test_onnx_backend_missing_file_raises(fake_onnxruntime):
    # The missing-file guard sits after the runtime import, so without onnxruntime
    # `OnnxBackend` raises ImportError before it ever looks at the path -- stub the
    # runtime rather than skip, so the guard is exercised everywhere.
    fake_onnxruntime({"scores": np.ones((1, 4), dtype=np.float32)})
    with pytest.raises(FileNotFoundError):
        OnnxBackend("/nonexistent/model.onnx")


def test_make_backend_unknown_extension_raises():
    with pytest.raises(ValueError, match="unsupported model extension"):
        make_backend("model.pt")


def test_make_backend_dispatches_onnx(onnx_classifier: Path):
    assert isinstance(make_backend(onnx_classifier), OnnxBackend)


def test_make_backend_dispatches_tflite(tmp_path):
    # .tflite dispatch reaches LiteRtBackend, which rejects the missing file
    with pytest.raises(FileNotFoundError):
        make_backend(tmp_path / "model.tflite")


def test_litert_backend_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        LiteRtBackend(tmp_path / "model.tflite")


def test_litert_backend_init_and_run_transposes_to_nhwc(tmp_path, monkeypatch):
    """With a stubbed interpreter, init reads tensor names and run transposes NCHW->NHWC."""
    model_path = tmp_path / "model.tflite"
    model_path.write_bytes(b"stub")
    interp = _FakeInterpreter()
    monkeypatch.setattr(backends, "_litert_interpreter", lambda path: interp)

    backend = LiteRtBackend(model_path)
    assert backend.input_name == "image"
    assert backend.output_names == ["scores"]

    out = backend.run(np.zeros((1, 3, 8, 8), dtype=np.float32))  # NCHW in
    assert out["scores"].shape == (1, 4)
    # the interpreter received the batch in NHWC layout
    assert interp.set_input is not None
    assert interp.set_input.shape == (1, 8, 8, 3)
    assert interp.resized_shape == (1, 8, 8, 3)


def test_litert_interpreter_raises_without_runtime(monkeypatch):
    # neither tflite_runtime nor tensorflow importable -> a helpful ImportError
    monkeypatch.setitem(sys.modules, "tflite_runtime", None)
    monkeypatch.setitem(sys.modules, "tensorflow", None)
    with pytest.raises(ImportError, match="tflite-runtime or tensorflow"):
        backends._litert_interpreter("model.tflite")


def test_litert_interpreter_uses_tflite_runtime(monkeypatch):
    # a present tflite_runtime is used to build the interpreter
    module = types.ModuleType("tflite_runtime")
    interpreter_module = types.ModuleType("tflite_runtime.interpreter")
    interpreter_module.Interpreter = _FakeInterpreter  # type: ignore
    module.interpreter = interpreter_module  # type: ignore
    monkeypatch.setitem(sys.modules, "tflite_runtime", module)
    monkeypatch.setitem(sys.modules, "tflite_runtime.interpreter", interpreter_module)

    interp = backends._litert_interpreter("model.tflite")
    assert isinstance(interp, _FakeInterpreter)
    assert interp.model_path == "model.tflite"


def test_onnx_backend_init_and_run_with_stubbed_runtime(stub_model_file, fake_onnxruntime):
    """With a stubbed runtime, init reads tensor names and run casts and keys the outputs."""
    session = fake_onnxruntime({"scores": np.ones((2, 4), dtype=np.float32)})

    backend = OnnxBackend(stub_model_file("model.onnx"))
    assert backend.input_name == "image"
    assert backend.output_names == ["scores"]

    out = backend.run(np.zeros((2, 3, 8, 8), dtype=np.uint8))
    assert out["scores"].shape == (2, 4)
    # the batch reaches the session under the model's own input name, cast to float32
    assert session.last_feed is not None
    assert session.last_feed["image"].dtype == np.float32


def test_onnx_backend_prefers_gpu_providers(stub_model_file, fake_onnxruntime):
    """Available accelerators are ordered ahead of CPU; unavailable ones are dropped."""
    session = fake_onnxruntime(
        {"scores": np.ones((1, 4), dtype=np.float32)},
        available_providers=["CPUExecutionProvider", "CUDAExecutionProvider"],
    )
    OnnxBackend(stub_model_file("model.onnx"))
    assert session.providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_make_backend_dispatches_onnx_with_stubbed_runtime(stub_model_file, fake_onnxruntime):
    fake_onnxruntime({"scores": np.ones((1, 4), dtype=np.float32)})
    assert isinstance(make_backend(stub_model_file("model.onnx")), OnnxBackend)
