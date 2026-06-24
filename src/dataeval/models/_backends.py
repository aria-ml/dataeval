"""Runtime backends for opinionated model inference (ONNX, LiteRT)."""

__all__ = ["RuntimeBackend", "OnnxBackend", "LiteRtBackend", "make_backend"]

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class RuntimeBackend(Protocol):
    """
    Loaded-model interface mapping an input tensor to named output arrays.

    A runtime-checkable protocol implemented by each concrete backend
    (:class:`OnnxBackend`, :class:`LiteRtBackend`). Decouples the predictors from
    any specific inference runtime: predictors hold a ``RuntimeBackend`` and only
    depend on its input/output tensor names and :meth:`run`.

    Attributes
    ----------
    input_name : str
        Name of the model's single input tensor.
    output_names : list[str]
        Names of the model's output tensors, in declaration order.
    """

    input_name: str
    output_names: list[str]

    def run(self, tensor: NDArray[Any]) -> dict[str, NDArray[Any]]:
        """
        Run one batch through the model.

        Parameters
        ----------
        tensor : NDArray[Any]
            Input batch in NCHW layout.

        Returns
        -------
        dict[str, NDArray[Any]]
            Output arrays keyed by tensor name (the keys in :attr:`output_names`).
        """
        ...


def _onnx_providers() -> list[str]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    preferred = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
    return [p for p in preferred if p in available] or ["CPUExecutionProvider"]


class OnnxBackend:
    """
    ONNX Runtime backend with NCHW input/output.

    Loads an ``.onnx`` model into an ONNX Runtime ``InferenceSession``, preferring
    GPU execution providers (CUDA, CoreML) when available and falling back to CPU.
    Implements :class:`RuntimeBackend`. Requires ``dataeval[onnx]``.

    Parameters
    ----------
    model_path : str or Path
        Path to the ``.onnx`` model file.

    Raises
    ------
    ImportError
        If ``onnxruntime`` is not installed.
    FileNotFoundError
        If ``model_path`` does not exist.
    """

    def __init__(self, model_path: str | Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover
            raise ImportError("onnxruntime is required for OnnxBackend. Install dataeval[onnx].") from e
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self._session = ort.InferenceSession(str(model_path), providers=_onnx_providers())
        self.input_name: str = self._session.get_inputs()[0].name
        self.output_names: list[str] = [o.name for o in self._session.get_outputs()]

    def run(self, tensor: NDArray[Any]) -> dict[str, NDArray[Any]]:
        """Run one NCHW batch and return outputs keyed by tensor name."""
        outputs = self._session.run(self.output_names, {self.input_name: tensor.astype(np.float32)})
        return {name: np.asarray(arr) for name, arr in zip(self.output_names, outputs, strict=True)}


def _litert_interpreter(model_path: str | Path) -> Any:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            from tensorflow.lite import Interpreter  # type: ignore[no-redef]
        except ImportError as e:
            raise ImportError("LiteRtBackend requires tflite-runtime or tensorflow. Install dataeval[tflite].") from e
    return Interpreter(model_path=str(model_path))


class LiteRtBackend:
    """
    LiteRT (TensorFlow Lite) backend with NCHW input.

    Loads a ``.tflite`` model via a LiteRT interpreter (``tflite-runtime`` if
    available, else ``tensorflow.lite``). Accepts NCHW input and transposes to the
    NHWC layout LiteRT expects internally, resizing the interpreter's input tensor
    per batch. Implements :class:`RuntimeBackend`. Requires ``dataeval[tflite]``.

    Parameters
    ----------
    model_path : str or Path
        Path to the ``.tflite`` model file.

    Raises
    ------
    ImportError
        If neither ``tflite-runtime`` nor ``tensorflow`` is installed.
    FileNotFoundError
        If ``model_path`` does not exist.
    """

    def __init__(self, model_path: str | Path) -> None:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self._interp = _litert_interpreter(model_path)
        self._interp.allocate_tensors()
        self._in = self._interp.get_input_details()[0]
        self._out = self._interp.get_output_details()
        self.input_name: str = str(self._in["name"])
        self.output_names: list[str] = [str(o["name"]) for o in self._out]

    def run(self, tensor: NDArray[Any]) -> dict[str, NDArray[Any]]:
        """Run one NCHW batch (transposed to NHWC) and return outputs by tensor name."""
        nhwc = np.transpose(tensor.astype(np.float32), (0, 2, 3, 1))  # NCHW -> NHWC
        self._interp.resize_tensor_input(self._in["index"], nhwc.shape)
        self._interp.allocate_tensors()
        self._interp.set_tensor(self._in["index"], nhwc)
        self._interp.invoke()
        return {str(o["name"]): np.asarray(self._interp.get_tensor(o["index"])) for o in self._out}


def make_backend(model_path: str | Path) -> RuntimeBackend:
    """
    Construct the runtime backend matching a model file's extension.

    Parameters
    ----------
    model_path : str or Path
        Path to a ``.onnx`` or ``.tflite`` model file.

    Returns
    -------
    RuntimeBackend
        An :class:`OnnxBackend` for ``.onnx`` files or a :class:`LiteRtBackend`
        for ``.tflite`` files.

    Raises
    ------
    ValueError
        If the file extension is neither ``.onnx`` nor ``.tflite``.
    """
    suffix = Path(model_path).suffix.lower()
    if suffix == ".onnx":
        return OnnxBackend(model_path)
    if suffix == ".tflite":
        return LiteRtBackend(model_path)
    raise ValueError(f"unsupported model extension {suffix!r}; expected .onnx or .tflite")
