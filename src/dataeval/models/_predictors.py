"""Opinionated ONNX/LiteRT predictors implemented as MAITE Models.

Each task (image classification, object detection) has an abstract base that holds
the shared metadata/input/decoding pipeline and one concrete subclass per runtime
backend. The backend is the only thing that varies, so adding or removing support
for a format is a matter of adding or deleting a small subclass.
"""

__all__ = [
    "OnnxImageClassifier",
    "LiteRtImageClassifier",
    "OnnxObjectDetector",
    "LiteRtObjectDetector",
]

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from maite.protocols import ModelMetadata
from numpy.typing import ArrayLike, NDArray

from dataeval.models._backends import LiteRtBackend, OnnxBackend, RuntimeBackend
from dataeval.models._input import build_model_input
from dataeval.models._metadata import ModelIOSpec, read_model_metadata
from dataeval.protocols import ObjectDetectionTarget


class _BaseImageClassifier(ABC):
    """
    MAITE ``image_classification.Model`` over an opinionated ONNX/LiteRT model.

    Loads an exported image-classification model and its model-metadata.json
    contract, then maps a batch of CHW images to per-image class scores. Calling
    the instance preprocesses each image to the model's input contract (color
    layout, size, ``[0, 1]`` normalization), runs inference, and returns one score
    array per image.

    This abstract base holds the shared metadata/input/decoding pipeline; the
    runtime backend is the only thing that varies. It cannot be instantiated
    directly -- use a concrete subclass that selects a backend
    (:class:`OnnxImageClassifier`, :class:`LiteRtImageClassifier`), which document
    the constructor parameters.

    Notes
    -----
    Implements the MAITE ``image_classification.Model`` protocol: instances expose
    a :class:`~maite.protocols.ModelMetadata` ``metadata`` attribute and are
    callable on a batch of images.
    """

    #: Short backend identifier used in the :attr:`metadata` id; overridden per subclass.
    _backend_label: str = "model"

    def __init__(
        self,
        model_path: str | Path,
        metadata_path: str | Path,
        *,
        image_size: tuple[int, int] | None = None,
        scores_name: str = "scores",
    ) -> None:
        self._spec: ModelIOSpec = read_model_metadata(metadata_path)
        if self._spec.task != "IMAGE_CLASSIFICATION":
            raise ValueError(f"metadata declares {self._spec.task}, not IMAGE_CLASSIFICATION")
        self._backend: RuntimeBackend = self._make_backend(model_path)
        self._image_size = image_size
        self._scores_name = scores_name
        self.metadata: ModelMetadata = {"id": f"dataeval-{self._backend_label}-classifier:{Path(model_path).name}"}

    @abstractmethod
    def _make_backend(self, model_path: str | Path) -> RuntimeBackend:
        """Load the model file with this subclass's runtime backend."""
        ...

    def __call__(self, input_batch: Sequence[ArrayLike]) -> list[NDArray[Any]]:
        """
        Run inference on a batch of images and return per-image class scores.

        Parameters
        ----------
        input_batch : Sequence[ArrayLike]
            Images in CHW layout. Each is preprocessed to the model's input
            contract before inference.

        Returns
        -------
        list[NDArray[Any]]
            One ``(nClasses,)`` float32 score array per input image.

        Raises
        ------
        ValueError
            If the model output named ``scores_name`` is missing or the score
            array is not 2-D ``(B, nClasses)``.
        """
        h, w = self._image_size or (None, None)
        tensor = build_model_input(input_batch, self._spec, height=h, width=w)
        outputs = self._backend.run(tensor)
        if self._scores_name not in outputs:
            raise ValueError(f"model output {self._scores_name!r} not found; got {list(outputs)}")
        scores = np.asarray(outputs[self._scores_name], dtype=np.float32)
        if scores.ndim != 2:
            raise ValueError(f"classification scores must be 2-D (B, nClasses); got {scores.shape}")
        return [scores[i] for i in range(scores.shape[0])]


class OnnxImageClassifier(_BaseImageClassifier):
    """
    Opinionated image classifier backed by ONNX Runtime.

    Loads an exported ``.onnx`` image-classification model via ONNX Runtime (GPU
    when available, else CPU) together with its model-metadata.json contract, then
    maps a batch of CHW images to per-image class scores. Calling the instance
    preprocesses each image to the model's input contract (color layout, size,
    ``[0, 1]`` normalization), runs inference, and returns one ``(nClasses,)``
    float32 score array per image. Requires ``dataeval[onnx]``.

    Parameters
    ----------
    model_path : str or Path
        Path to the exported ``.onnx`` model file.
    metadata_path : str or Path
        Path to the model-metadata.json describing the input/output contract. Its
        declared task must be ``IMAGE_CLASSIFICATION``.
    image_size : tuple[int, int] or None, default None
        Optional ``(height, width)`` override for the model's input size. When set,
        images are resized to this size instead of the size declared in metadata;
        required when the metadata declares a variable (``-1``) dimension.
    scores_name : str, default "scores"
        Name of the model output tensor holding class scores.

    Raises
    ------
    ValueError
        If the metadata declares a task other than ``IMAGE_CLASSIFICATION``.
    ImportError
        If ``onnxruntime`` is not installed.
    FileNotFoundError
        If ``model_path`` does not exist.

    See Also
    --------
    ~dataeval.models.LiteRtImageClassifier : Same classifier backed by LiteRT.
    ~dataeval.models.OnnxObjectDetector : Opinionated object detector backed by ONNX Runtime.
    ~dataeval.models.read_model_metadata : Parse the metadata contract.

    Notes
    -----
    Implements the MAITE ``image_classification.Model`` protocol: instances expose
    a :class:`~maite.protocols.ModelMetadata` ``metadata`` attribute and are
    callable on a batch of images.

    Examples
    --------
    >>> from dataeval.models import OnnxImageClassifier
    >>> classifier = OnnxImageClassifier(onnx_classifier_path, classifier_metadata_path)
    >>> batch = [np.zeros((3, 16, 16), dtype=np.uint8), np.full((3, 16, 16), 255, dtype=np.uint8)]
    >>> scores = classifier(batch)
    >>> len(scores)
    2
    >>> scores[0].shape
    (4,)
    """

    _backend_label: str = "onnx"

    def _make_backend(self, model_path: str | Path) -> RuntimeBackend:
        return OnnxBackend(model_path)


class LiteRtImageClassifier(_BaseImageClassifier):
    """
    Opinionated image classifier backed by LiteRT (TensorFlow Lite).

    Loads an exported ``.tflite`` image-classification model via a LiteRT
    interpreter together with its model-metadata.json contract, then maps a batch
    of CHW images to per-image class scores. Calling the instance preprocesses each
    image to the model's input contract (color layout, size, ``[0, 1]``
    normalization), runs inference, and returns one ``(nClasses,)`` float32 score
    array per image. Requires ``dataeval[tflite]``.

    Parameters
    ----------
    model_path : str or Path
        Path to the exported ``.tflite`` model file.
    metadata_path : str or Path
        Path to the model-metadata.json describing the input/output contract. Its
        declared task must be ``IMAGE_CLASSIFICATION``.
    image_size : tuple[int, int] or None, default None
        Optional ``(height, width)`` override for the model's input size. When set,
        images are resized to this size instead of the size declared in metadata;
        required when the metadata declares a variable (``-1``) dimension.
    scores_name : str, default "scores"
        Name of the model output tensor holding class scores.

    Raises
    ------
    ValueError
        If the metadata declares a task other than ``IMAGE_CLASSIFICATION``.
    ImportError
        If neither ``tflite-runtime`` nor ``tensorflow`` is installed.
    FileNotFoundError
        If ``model_path`` does not exist.

    See Also
    --------
    ~dataeval.models.OnnxImageClassifier : Same classifier backed by ONNX Runtime.
    ~dataeval.models.LiteRtObjectDetector : Opinionated object detector backed by LiteRT.
    ~dataeval.models.read_model_metadata : Parse the metadata contract.

    Notes
    -----
    Implements the MAITE ``image_classification.Model`` protocol: instances expose
    a :class:`~maite.protocols.ModelMetadata` ``metadata`` attribute and are
    callable on a batch of images.
    """

    _backend_label: str = "litert"

    def _make_backend(self, model_path: str | Path) -> RuntimeBackend:
        return LiteRtBackend(model_path)


@dataclass
class _ObjectDetectionTarget:
    boxes: NDArray[Any]  # (N, 4), normalized (x0, y0, x1, y1)
    labels: NDArray[Any]  # (N,)
    scores: NDArray[Any]  # (N, nClasses)


class _BaseObjectDetector(ABC):
    """
    MAITE ``object_detection.Model`` over an opinionated ONNX/LiteRT model.

    Loads an exported object-detection model and its model-metadata.json contract,
    then maps a batch of CHW images to per-image detection targets. Calling the
    instance preprocesses each image to the model's input contract (color layout,
    size, ``[0, 1]`` normalization), runs inference, and returns one
    :obj:`ObjectDetectionTarget` per image.

    This abstract base holds the shared metadata/input/decoding pipeline; the
    runtime backend is the only thing that varies. It cannot be instantiated
    directly -- use a concrete subclass that selects a backend
    (:class:`OnnxObjectDetector`, :class:`LiteRtObjectDetector`), which document
    the constructor parameters.

    Notes
    -----
    Implements the MAITE ``object_detection.Model`` protocol: instances expose a
    :class:`~maite.protocols.ModelMetadata` ``metadata`` attribute and are callable
    on a batch of images.
    """

    #: Short backend identifier used in the :attr:`metadata` id; overridden per subclass.
    _backend_label: str = "model"

    def __init__(
        self,
        model_path: str | Path,
        metadata_path: str | Path,
        *,
        image_size: tuple[int, int] | None = None,
        boxes_name: str = "boxes",
        scores_name: str = "scores",
    ) -> None:
        self._spec: ModelIOSpec = read_model_metadata(metadata_path)
        if self._spec.task != "IMAGE_OBJECT_DETECTION":
            raise ValueError(f"metadata declares {self._spec.task}, not IMAGE_OBJECT_DETECTION")
        self._backend: RuntimeBackend = self._make_backend(model_path)
        self._image_size = image_size
        self._boxes_name = boxes_name
        self._scores_name = scores_name
        self.metadata: ModelMetadata = {"id": f"dataeval-{self._backend_label}-detector:{Path(model_path).name}"}

    @abstractmethod
    def _make_backend(self, model_path: str | Path) -> RuntimeBackend:
        """Load the model file with this subclass's runtime backend."""
        ...

    def __call__(self, input_batch: Sequence[ArrayLike]) -> list[ObjectDetectionTarget]:
        """
        Run inference on a batch of images and return per-image detections.

        Parameters
        ----------
        input_batch : Sequence[ArrayLike]
            Images in CHW layout. Each is preprocessed to the model's input
            contract before inference.

        Returns
        -------
        list[ObjectDetectionTarget]
            One :obj:`ObjectDetectionTarget` per input image, with labels taken
            as the argmax over each detection's class scores.

        Raises
        ------
        ValueError
            If the ``boxes_name`` or ``scores_name`` output is missing, or the
            output shapes are not ``(B, nBoxes, 4)`` and ``(B, nBoxes, nClasses)``.
        """
        h, w = self._image_size or (None, None)
        tensor = build_model_input(input_batch, self._spec, height=h, width=w)
        outputs = self._backend.run(tensor)
        for name in (self._boxes_name, self._scores_name):
            if name not in outputs:
                raise ValueError(f"model output {name!r} not found; got {list(outputs)}")
        boxes = np.asarray(outputs[self._boxes_name], dtype=np.float32)  # (B, nBoxes, 4)
        scores = np.asarray(outputs[self._scores_name], dtype=np.float32)  # (B, nBoxes, nClasses)
        self._validate_outputs(boxes, scores)
        return [
            _ObjectDetectionTarget(
                boxes=boxes[i],
                labels=scores[i].argmax(axis=1).astype(np.int64),
                scores=scores[i],
            )
            for i in range(boxes.shape[0])
        ]

    def _validate_outputs(self, boxes: NDArray[Any], scores: NDArray[Any]) -> None:
        """Validate detection output shapes."""
        if boxes.ndim != 3 or boxes.shape[-1] != 4:
            raise ValueError(f"detection boxes must be (B, nBoxes, 4); got {boxes.shape}")
        if scores.ndim != 3:
            raise ValueError(f"detection scores must be (B, nBoxes, nClasses); got {scores.shape}")


class OnnxObjectDetector(_BaseObjectDetector):
    """
    Opinionated object detector backed by ONNX Runtime.

    Loads an exported ``.onnx`` object-detection model via ONNX Runtime (GPU when
    available, else CPU) together with its model-metadata.json contract, then maps
    a batch of CHW images to per-image detection targets. Calling the instance
    preprocesses each image to the model's input contract (color layout, size,
    ``[0, 1]`` normalization), runs inference, and returns one detection target
    per image. Requires ``dataeval[onnx]``.

    Parameters
    ----------
    model_path : str or Path
        Path to the exported ``.onnx`` model file.
    metadata_path : str or Path
        Path to the model-metadata.json describing the input/output contract. Its
        declared task must be ``IMAGE_OBJECT_DETECTION``.
    image_size : tuple[int, int] or None, default None
        Optional ``(height, width)`` override for the model's input size. When set,
        images are resized to this size instead of the size declared in metadata;
        required when the metadata declares a variable (``-1``) dimension.
    boxes_name : str, default "boxes"
        Name of the model output tensor holding detection boxes.
    scores_name : str, default "scores"
        Name of the model output tensor holding per-class detection scores.

    Raises
    ------
    ValueError
        If the metadata declares a task other than ``IMAGE_OBJECT_DETECTION``.
    ImportError
        If ``onnxruntime`` is not installed.
    FileNotFoundError
        If ``model_path`` does not exist.

    See Also
    --------
    ~dataeval.models.LiteRtObjectDetector : Same detector backed by LiteRT.
    ~dataeval.models.OnnxImageClassifier : Opinionated image classifier backed by ONNX Runtime.
    ~dataeval.models.read_model_metadata : Parse the metadata contract.

    Notes
    -----
    Implements the MAITE ``object_detection.Model`` protocol: instances expose a
    :class:`~maite.protocols.ModelMetadata` ``metadata`` attribute and are callable
    on a batch of images, returning one MAITE
    :obj:`~dataeval.protocols.ObjectDetectionTarget` per image (with labels taken
    as the argmax over each detection's class scores).

    Examples
    --------
    >>> from dataeval.models import OnnxObjectDetector
    >>> detector = OnnxObjectDetector(onnx_detector_path, detector_metadata_path)
    >>> targets = detector([np.zeros((3, 16, 16), dtype=np.uint8)])
    >>> len(targets)
    1
    >>> targets[0].boxes.shape
    (5, 4)
    >>> targets[0].scores.shape
    (5, 4)
    """

    _backend_label: str = "onnx"

    def _make_backend(self, model_path: str | Path) -> RuntimeBackend:
        return OnnxBackend(model_path)


class LiteRtObjectDetector(_BaseObjectDetector):
    """
    Opinionated object detector backed by LiteRT (TensorFlow Lite).

    Loads an exported ``.tflite`` object-detection model via a LiteRT interpreter
    together with its model-metadata.json contract, then maps a batch of CHW images
    to per-image detection targets. Calling the instance preprocesses each image to
    the model's input contract (color layout, size, ``[0, 1]`` normalization), runs
    inference, and returns one detection target per image. Requires
    ``dataeval[tflite]``.

    Parameters
    ----------
    model_path : str or Path
        Path to the exported ``.tflite`` model file.
    metadata_path : str or Path
        Path to the model-metadata.json describing the input/output contract. Its
        declared task must be ``IMAGE_OBJECT_DETECTION``.
    image_size : tuple[int, int] or None, default None
        Optional ``(height, width)`` override for the model's input size. When set,
        images are resized to this size instead of the size declared in metadata;
        required when the metadata declares a variable (``-1``) dimension.
    boxes_name : str, default "boxes"
        Name of the model output tensor holding detection boxes.
    scores_name : str, default "scores"
        Name of the model output tensor holding per-class detection scores.

    Raises
    ------
    ValueError
        If the metadata declares a task other than ``IMAGE_OBJECT_DETECTION``.
    ImportError
        If neither ``tflite-runtime`` nor ``tensorflow`` is installed.
    FileNotFoundError
        If ``model_path`` does not exist.

    See Also
    --------
    ~dataeval.models.OnnxObjectDetector : Same detector backed by ONNX Runtime.
    ~dataeval.models.LiteRtImageClassifier : Opinionated image classifier backed by LiteRT.
    ~dataeval.models.read_model_metadata : Parse the metadata contract.

    Notes
    -----
    Implements the MAITE ``object_detection.Model`` protocol: instances expose a
    :class:`~maite.protocols.ModelMetadata` ``metadata`` attribute and are callable
    on a batch of images, returning one MAITE
    :obj:`~dataeval.protocols.ObjectDetectionTarget` per image (with labels taken
    as the argmax over each detection's class scores).
    """

    _backend_label: str = "litert"

    def _make_backend(self, model_path: str | Path) -> RuntimeBackend:
        return LiteRtBackend(model_path)
