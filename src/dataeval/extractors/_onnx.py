"""
ONNX Runtime-based feature extractor.
"""

__all__ = []

import logging
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from dataeval.protocols import Array, Transform
from dataeval.utils.arrays import as_numpy

if TYPE_CHECKING:
    from onnxruntime import InferenceSession
else:
    InferenceSession = Any

_logger = logging.getLogger(__name__)

_ort_import_error = ImportError(
    "onnxruntime is required for OnnxExtractor. "
    "Install it with: pip install onnxruntime (CPU) or pip install onnxruntime-gpu (GPU)"
)


def _get_ort() -> ModuleType:
    """Import onnxruntime and return InferenceSession class."""
    try:
        import onnxruntime

        return onnxruntime
    except ImportError as e:
        raise ImportError(
            "onnxruntime is required for OnnxExtractor. "
            "Install it with: pip install onnxruntime (CPU) or pip install onnxruntime-gpu (GPU)"
        ) from e


def _get_inference_session() -> type[InferenceSession]:
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise _ort_import_error from e

    return ort.InferenceSession


def _get_execution_providers() -> list[str]:
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise _ort_import_error from e

    """Get available ONNX Runtime execution providers with GPU fallback to CPU."""
    available = ort.get_available_providers()

    # Prefer GPU providers, fall back to CPU
    preferred = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
    providers = [p for p in preferred if p in available]

    if not providers:
        providers = ["CPUExecutionProvider"]

    return providers


class OnnxExtractor:
    """
    Extracts embeddings via ONNX Runtime with lazy model loading.

    Encapsulates ONNX-specific logic for feature extraction:

    - Model loading from ONNX files or in-memory bytes
    - Automatic GPU/CPU provider selection with fallback
    - Transform pipeline
    - Output layer selection for multi-output models

    Implements the :class:`~dataeval.protocols.FeatureExtractor` protocol.

    Parameters
    ----------
    model : str, Path, or bytes
        Path to the ONNX model file, or serialized model bytes from
        :func:`~dataeval.utils.onnx.to_encoding_model`.
    transforms : Transform or Sequence[Transform] or None, default None
        Preprocessing transforms to apply before encoding. When None, uses raw images.
    output_name : str or None, default None
        Name of the output to extract embeddings from. When None, uses the first output.
        Required for models with multiple outputs.
    flatten : bool, default True
        If True, flattens outputs with more than 2 dimensions to (N, D) shape.
        If False, preserves the original output shape.

    Example
    -------
    Basic usage with a model file:

    >>> from dataeval import Embeddings
    >>> from dataeval.extractors import OnnxExtractor
    >>>
    >>> extractor = OnnxExtractor("model.onnx")
    >>> embeddings = Embeddings(dataset, extractor=extractor, batch_size=32)

    Notes
    -----
    - The extractor expects images in CHW format (channels, height, width).
    - For models with multiple outputs, use ``output_name`` to specify which
      output contains embeddings.
    - The model is loaded lazily on first use.
    - Requires ``onnxruntime`` or ``onnxruntime-gpu`` to be installed.
    """

    def __init__(
        self,
        model: str | Path | bytes,
        transforms: Transform[NDArray[Any]] | Sequence[Transform[NDArray[Any]]] | None = None,
        output_name: str | None = None,
        flatten: bool = True,
    ) -> None:
        if isinstance(model, bytes):
            self._model_bytes: bytes | None = model
            self._model_path: Path | None = None
        else:
            self._model_bytes = None
            self._model_path = Path(model)

        self._transforms = self._normalize_transforms(transforms)
        self._output_name = output_name
        self._flatten = flatten

        # Lazy-loaded session
        self._session: InferenceSession | None = None
        self._input_name: str | None = None
        self._output_names: list[str] = []

    @property
    def output_name(self) -> str | None:
        """Return the output name for extraction, if set."""
        return self._output_name

    def _normalize_transforms(
        self, transforms: Transform[NDArray[Any]] | Sequence[Transform[NDArray[Any]]] | None
    ) -> list[Transform[NDArray[Any]]]:
        """Normalize transforms to a list."""
        if transforms is None:
            return []
        if isinstance(transforms, Transform):
            return [transforms]
        return list(transforms)

    def _load_model(self) -> None:
        """Load the ONNX model and validate configuration."""
        InferenceSession = _get_inference_session()
        providers = _get_execution_providers()

        if self._model_bytes is not None:
            _logger.debug(f"Loading ONNX model from bytes with providers: {providers}")
            self._session = InferenceSession(self._model_bytes, providers=providers)
            model_source = "bytes"
        else:
            if self._model_path is None or not self._model_path.exists():
                raise FileNotFoundError(f"Model not found: {self._model_path}")
            _logger.debug(f"Loading ONNX model from {self._model_path} with providers: {providers}")
            self._session = InferenceSession(str(self._model_path), providers=providers)
            model_source = str(self._model_path)

        if self._session is None:
            raise RuntimeError(f"Failed to create ONNX InferenceSession for model: {model_source}")

        # Get input/output metadata
        model_inputs = self._session.get_inputs()
        if not model_inputs:
            raise ValueError(f"ONNX model has no inputs: {model_source}")

        model_outputs = self._session.get_outputs()
        if not model_outputs:
            raise ValueError(f"ONNX model has no outputs: {model_source}")

        self._input_name = model_inputs[0].name
        self._output_names = [o.name for o in model_outputs]

        # Validate output_name if specified
        if self._output_name is not None and self._output_name not in self._output_names:
            raise ValueError(
                f"Specified output_name '{self._output_name}' not found in model outputs.\n"
                f"  Available outputs: {self._output_names}"
            )

        # Multi-output models require explicit output_name
        if len(self._output_names) > 1 and self._output_name is None:
            raise ValueError(
                f"Model has {len(self._output_names)} outputs: {self._output_names}.\n"
                f"Specify 'output_name' to indicate which output produces embeddings."
            )

        _logger.debug(f"ONNX model loaded. Input: {self._input_name}, Outputs: {self._output_names}")

    def _ensure_loaded(self) -> InferenceSession:
        """Ensure the model is loaded and return the session."""
        if self._session is None:
            self._load_model()
        if self._session is None:
            raise RuntimeError("Failed to load ONNX model")
        return self._session

    def _preprocess(self, image: NDArray[Any]) -> NDArray[np.floating[Any]]:
        """Apply preprocessing transforms to an image."""
        result = image
        for transform in self._transforms:
            result = transform(result)

        # Ensure float32 for ONNX
        return result.astype(np.float32)

    def __call__(self, data: Any) -> Array:
        """
        Extract features from a batch of images.

        Parameters
        ----------
        data : Any
            Iterable of images to extract features from. Each image should be
            array-like in CHW format.

        Returns
        -------
        Array
            Embeddings array of shape (n_images, embedding_dim).

        Raises
        ------
        FileNotFoundError
            If the model file does not exist.
        ImportError
            If onnxruntime is not installed.
        """
        session = self._ensure_loaded()

        # Collect and preprocess images
        batch_images: list[NDArray[Any]] = []
        for img in data:
            image_array = as_numpy(img)
            processed = self._preprocess(image_array)
            batch_images.append(processed)

        if not batch_images:
            return np.empty((0,), dtype=np.float32)

        # Stack into batch: (N, C, H, W)
        batch_array = np.stack(batch_images)

        # Run inference
        if self._input_name is None:
            raise RuntimeError("Model input name not set")

        outputs = session.run(self._output_names, {self._input_name: batch_array})

        # Select the appropriate output
        if self._output_name is not None:
            idx = self._output_names.index(self._output_name)
            result = outputs[idx]
        else:
            result = outputs[0]

        result = np.asarray(result)

        # Flatten spatial dimensions if present (N, C, H, W) -> (N, C*H*W) or (N, D) -> (N, D)
        if self._flatten and result.ndim > 2:
            result = result.reshape(result.shape[0], -1)

        return result

    def __repr__(self) -> str:
        output_info = f", output_name={self._output_name!r}" if self._output_name else ""
        flatten_info = "" if self._flatten else ", flatten=False"
        if self._model_bytes is not None:
            model_info = f"model=<{len(self._model_bytes)} bytes>"
        else:
            model_info = f"model={self._model_path!r}"
        return f"OnnxExtractor({model_info}{output_info}{flatten_info})"
