"""
Common type protocols used for interoperability with DataEval.
"""

from __future__ import annotations

__all__ = [
    "AnnotatedDataset",
    "Array",
    "ArrayLike",
    "Dataset",
    "DatasetMetadata",
    "DeviceLike",
    "ImageClassificationDatum",
    "ImageClassificationDataset",
    "ObjectDetectionTarget",
    "ObjectDetectionDatum",
    "ObjectDetectionDataset",
    "SegmentationTarget",
    "SegmentationDatum",
    "SegmentationDataset",
    "Transform",
]


from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import (
    Any,
    Protocol,
    TypeAlias,
    TypedDict,
    TypeVar,
    runtime_checkable,
)

import numpy as np
import torch
from numpy.typing import NDArray
from typing_extensions import NotRequired, ReadOnly, Required

ArrayLike: TypeAlias = np.typing.ArrayLike
"""
Type alias for a `Union` representing objects that can be coerced into an array.

See Also
--------
`NumPy ArrayLike <https://numpy.org/doc/stable/reference/typing.html#numpy.typing.ArrayLike>`_
"""

DeviceLike: TypeAlias = int | str | tuple[str, int] | torch.device
"""
Type alias for a `Union` representing types that specify a torch.device.

See Also
--------
`torch.device <https://pytorch.org/docs/stable/tensor_attributes.html#torch.device>`_
"""


@runtime_checkable
class Array(Protocol):
    """
    Protocol for array objects providing interoperability with DataEval.

    Supports common array representations with popular libraries like
    PyTorch, Tensorflow and JAX, as well as NumPy arrays.

    Example
    -------
    >>> import numpy as np
    >>> import torch
    >>> from dataeval.protocols import Array

    Create array objects

    >>> ndarray = np.random.random((10, 10))
    >>> tensor = torch.tensor([1, 2, 3])

    Check type at runtime

    >>> isinstance(ndarray, Array)
    True

    >>> isinstance(tensor, Array)
    True
    """

    @property
    def shape(self) -> tuple[int, ...]: ...
    def __array__(self) -> NDArray[Any]: ...
    def __getitem__(self, key: Any, /) -> Any: ...
    def __iter__(self) -> Iterator[Any]: ...
    def __len__(self) -> int: ...


_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)
_T_cn = TypeVar("_T_cn", contravariant=True)

# ========== METADATA ==========


class DatasetMetadata(TypedDict, total=False):
    """
    Dataset level metadata required for all `AnnotatedDataset` classes.

    Attributes
    ----------
    id : Required[int | str]
        A unique identifier for the dataset
    index2label : NotRequired[dict[int, str]]
        A lookup table converting label value to class name
    """

    id: Required[ReadOnly[int | str]]
    index2label: NotRequired[ReadOnly[dict[int, str]]]


class ModelMetadata(TypedDict, total=False):
    """
    Model metadata required for all `AnnotatedModel` classes.

    Attributes
    ----------
    id : Required[str]
        A unique identifier for the model
    index2label : NotRequired[dict[int, str]]
        A lookup table converting label value to class name
    """

    id: Required[ReadOnly[str]]
    index2label: NotRequired[ReadOnly[dict[int, str]]]


class DatumMetadata(TypedDict, total=False):
    """
    Datum level metadata required for all `AnnotatedDataset` classes.

    Attributes
    ----------
    id : Required[int | str]
        A unique identifier for the datum
    """

    id: Required[ReadOnly[int | str]]


# ========== DATASETS ==========


@runtime_checkable
class Dataset(Protocol[_T_co]):
    """
    Protocol for a generic `Dataset`.

    Methods
    -------
    __getitem__(index: int)
        Returns datum at specified index.
    __len__()
        Returns dataset length.
    """

    def __getitem__(self, index: int, /) -> _T_co: ...
    def __len__(self) -> int: ...


@runtime_checkable
class AnnotatedDataset(Dataset[_T_co], Protocol[_T_co]):
    """
    Protocol for a generic `AnnotatedDataset`.

    Attributes
    ----------
    metadata : :class:`.DatasetMetadata` or derivatives.

    Methods
    -------
    __getitem__(index: int)
        Returns datum at specified index.
    __len__()
        Returns dataset length.

    Notes
    -----
    Inherits from :class:`.Dataset`.
    """

    @property
    def metadata(self) -> DatasetMetadata: ...


# ========== IMAGE CLASSIFICATION DATASETS ==========


ImageClassificationDatum: TypeAlias = tuple[ArrayLike, ArrayLike, DatumMetadata]
"""
Type alias for an image classification datum tuple.

- :class:`ArrayLike` of shape (C, H, W) - Image data in channel, height, width format.
- :class:`ArrayLike` of shape (N,) - Class label as one-hot encoded ground-truth or prediction confidences.
- dict[str, Any] - Datum level metadata.
"""


ImageClassificationDataset: TypeAlias = AnnotatedDataset[ImageClassificationDatum]
"""
Type alias for an :class:`AnnotatedDataset` of :class:`ImageClassificationDatum` elements.
"""

# ========== OBJECT DETECTION DATASETS ==========


@runtime_checkable
class ObjectDetectionTarget(Protocol):
    """
    Protocol for targets in an Object Detection dataset.

    Attributes
    ----------
    boxes : :class:`ArrayLike` of shape (N, 4)
    labels : :class:`ArrayLike` of shape (N,)
    scores : :class:`ArrayLike` of shape (N, M)
    """

    @property
    def boxes(self) -> ArrayLike: ...

    @property
    def labels(self) -> ArrayLike: ...

    @property
    def scores(self) -> ArrayLike: ...


ObjectDetectionDatum: TypeAlias = tuple[ArrayLike, ObjectDetectionTarget, DatumMetadata]
"""
Type alias for an object detection datum tuple.

- :class:`ArrayLike` of shape (C, H, W) - Image data in channel, height, width format.
- :class:`ObjectDetectionTarget` - Object detection target information for the image.
- dict[str, Any] - Datum level metadata.
"""


ObjectDetectionDataset: TypeAlias = AnnotatedDataset[ObjectDetectionDatum]
"""
Type alias for an :class:`AnnotatedDataset` of :class:`ObjectDetectionDatum` elements.
"""


# ========== SEGMENTATION DATASETS ==========


@runtime_checkable
class SegmentationTarget(Protocol):
    """
    Protocol for targets in a Segmentation dataset.

    Attributes
    ----------
    mask : :class:`ArrayLike`
    labels : :class:`ArrayLike`
    scores : :class:`ArrayLike`
    """

    @property
    def mask(self) -> ArrayLike: ...

    @property
    def labels(self) -> ArrayLike: ...

    @property
    def scores(self) -> ArrayLike: ...


SegmentationDatum: TypeAlias = tuple[ArrayLike, SegmentationTarget, DatumMetadata]
"""
Type alias for an image classification datum tuple.

- :class:`ArrayLike` of shape (C, H, W) - Image data in channel, height, width format.
- :class:`SegmentationTarget` - Segmentation target information for the image.
- dict[str, Any] - Datum level metadata.
"""

SegmentationDataset: TypeAlias = AnnotatedDataset[SegmentationDatum]
"""
Type alias for an :class:`AnnotatedDataset` of :class:`SegmentationDatum` elements.
"""


# ========== TRANSFORM ==========


@runtime_checkable
class Transform(Protocol[_T]):
    """
    Protocol defining a transform function.

    Requires a `__call__` method that returns transformed data.

    Example
    -------
    >>> from typing import Any
    >>> from numpy.typing import NDArray

    >>> class MyTransform:
    ...     def __init__(self, divisor: float) -> None:
    ...         self.divisor = divisor
    ...
    ...     def __call__(self, data: NDArray[Any], /) -> NDArray[Any]:
    ...         return data / self.divisor

    >>> my_transform = MyTransform(divisor=255.0)
    >>> isinstance(my_transform, Transform)
    True
    >>> my_transform(np.array([1, 2, 3]))
    array([0.004, 0.008, 0.012])
    """

    def __call__(self, data: _T, /) -> _T: ...


# ========== MODEL ==========


@runtime_checkable
class AnnotatedModel(Protocol):
    """
    Protocol for an annotated model.
    """

    @property
    def metadata(self) -> ModelMetadata: ...


EmbeddingModel: TypeAlias = Callable[[Array], Array] | Callable[[Iterable[Array]], Iterable[Array]]
"""
Type alias for a callable embedding model.

Embedding models should take an input array or a batch of arrays and return an output
representing the input data in a lower dimensional space.
"""

# ========== SUFFICIENCY STRATEGIES ==========


@runtime_checkable
class TrainingStrategy(Protocol[_T_cn]):
    """
    Protocol defining the interface for training a model on a dataset subset.

    Implementations must provide a `train` method with this signature.
    Uses structural typing - no explicit inheritance required.

    The @runtime_checkable decorator allows isinstance() checks if needed,
    though structural typing works without it at type-check time.

    Examples
    --------
    Creating a custom training strategy:

    >>> class MyTraining:
    ...     def __init__(self, learning_rate: float, epochs: int):
    ...         self.learning_rate = learning_rate
    ...         self.epochs = epochs
    ...
    ...     def train(self, model: torch.nn.Module, dataset: Dataset, indices: Sequence[int]) -> None:
    ...         # Custom training implementation
    ...         optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
    ...         for epoch in range(self.epochs):
    ...             # Training loop using specified indices
    ...             ...
    """

    def train(self, model: torch.nn.Module, dataset: Dataset[_T_cn], indices: Sequence[int]) -> None:
        """
        Train the model using the specified indices from the dataset.

        Parameters
        ----------
        model : nn.Module
            The model to train. Training should modify the model in-place.
        dataset : Dataset[T]
            The full dataset. Only samples at the specified indices should
            be used for training.
        indices : Sequence[int]
            Indices indicating which samples from the dataset to use for
            training this step. These allow the same model to be trained
            incrementally on growing subsets.

        Returns
        -------
        None
            Training modifies the model in-place.

        Notes
        -----
        Implementations should:
        - Only use samples at the specified indices
        - Modify the model parameters in-place
        - Handle their own loss computation and optimization
        - Be deterministic or set seeds internally for reproducibility
        """
        ...


@runtime_checkable
class EvaluationStrategy(Protocol[_T_cn]):
    """
    Protocol defining the interface for evaluating a trained model.

    Implementations must provide an `evaluate` method with this signature.
    Uses structural typing - no explicit inheritance required.

    The @runtime_checkable decorator allows isinstance() checks if needed,
    though structural typing works without it at type-check time.

    Examples
    --------
    Creating a custom evaluation strategy:

    >>> class MyEvaluation:
    ...     def __init__(self, batch_size: int, metrics: list[str]):
    ...         self.batch_size = batch_size
    ...         self.metrics = metrics
    ...
    ...     def evaluate(self, model: torch.nn.Module, dataset: Dataset) -> Mapping[str, float | np.ndarray]:
    ...         # Custom evaluation implementation
    ...         model.eval()
    ...         with torch.no_grad():
    ...             # Compute metrics
    ...             ...
    ...         return {"accuracy": 0.95, "f1": 0.93}
    """

    def evaluate(self, model: torch.nn.Module, dataset: Dataset[_T_cn]) -> Mapping[str, float | ArrayLike]:
        """
        Evaluate the model on the dataset and return performance metrics.

        Parameters
        ----------
        model : nn.Module
            The trained model to evaluate
        dataset : Dataset[T]
            The dataset to evaluate on (typically a test/validation set)

        Returns
        -------
        Mapping[str, float | ArrayLike]
            Mapping of metric names to values. Each value is either:
            - A scalar (float) for single-class metrics
            - An array (np.ndarray) for per-class or per-sample metrics

            Examples:
            - {"accuracy": 0.95}  # Single metric
            - {"accuracy": 0.95, "precision": 0.93, "recall": 0.94}  # Multiple metrics
            - {"accuracy": np.array([0.9, 0.85, 0.92])}  # Per-class metrics

        Notes
        -----
        Implementations should:
        - Set model to eval mode if needed
        - Return consistent metric names across calls
        - Handle both single-class and multi-class scenarios
        - Use the entire dataset (unlike training which uses subsets)
        """
        ...
