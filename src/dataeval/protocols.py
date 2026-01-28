"""
Common type protocols used for interoperability with DataEval.
"""

__all__ = [
    "AnnotatedDataset",
    "Array",
    "ArrayLike",
    "Dataset",
    "DatasetMetadata",
    "DeviceLike",
    "EmbeddingEncoder",
    "EvaluationSchedule",
    "EvaluationStrategy",
    "FeatureExtractor",
    "EvidenceLowerBoundLossFn",
    "ImageClassificationDatum",
    "ImageClassificationDataset",
    "LossFn",
    "Metadata",
    "ObjectDetectionTarget",
    "ObjectDetectionDatum",
    "ObjectDetectionDataset",
    "ReconstructionLossFn",
    "SegmentationTarget",
    "SegmentationDatum",
    "SegmentationDataset",
    "TrainingStrategy",
    "Transform",
    "UpdateStrategy",
]


from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import (
    Any,
    Literal,
    Protocol,
    TypeAlias,
    TypedDict,
    TypeVar,
    overload,
    runtime_checkable,
)

import numpy as np
import torch
from numpy.typing import NDArray
from typing_extensions import NotRequired, ReadOnly, Required, Self

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


@runtime_checkable
class SequenceLike(Protocol[_T_co]):
    """Protocol for sequence-like objects that can be indexed and iterated."""

    @overload
    def __getitem__(self, key: int, /) -> _T_co: ...
    @overload
    def __getitem__(self, key: Any, /) -> _T_co | Self: ...
    def __iter__(self) -> Iterator[_T_co]: ...
    def __len__(self) -> int: ...


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


@runtime_checkable
class Metadata(Protocol):
    """
    Minimal protocol for metadata objects used in bias and quality analysis.

    This protocol defines the minimum interface required for metadata objects
    to be used with DataEval's bias evaluators (Balance, Diversity, Parity)
    and quality evaluators (Outliers). Users can implement lightweight custom
    metadata containers that satisfy this protocol.

    Attributes
    ----------
    factor_names : Sequence[str]
        Names of the metadata factors.
    factor_data : NDArray[np.int64]
        Metadata factors in array of shape (n_samples, n_factors).
        Continuous factors or non-integer data should be preprocessed into
        discrete integer bins before being returned here.
    class_labels : NDArray[np.intp]
        Flat array of class labels with one entry per target/detection.
        For image classification, length equals number of images.
        For object detection, length equals total detections across all images.
    is_discrete : Sequence[bool]
        Whether each factor is discrete (True) or continuous (False).
        Must have the same length as factor_names.
    index2label : NotRequired[Mapping[int, str]]
        Optional mapping from class label indices to human-readable names.
    item_indices : NotRequired[NDArray[np.intp]]
        Optional array mapping each label back to its source item/image.
        If not provided, a 1:1 mapping is assumed (one label per image).

    Example
    -------
    Creating a simple metadata container:

    >>> import numpy as np
    >>> from dataeval.protocols import Metadata
    >>>
    >>> class SimpleMetadata(Metadata):
    ...     def __init__(self, factors, labels, names, discrete):
    ...         self._factors = factors
    ...         self._labels = labels
    ...         self._names = names
    ...         self._discrete = discrete
    ...
    ...     @property
    ...     def factor_names(self):
    ...         return self._names
    ...
    ...     @property
    ...     def factor_data(self):
    ...         return self._factors
    ...
    ...     @property
    ...     def class_labels(self):
    ...         return self._labels
    ...
    ...     @property
    ...     def is_discrete(self):
    ...         return self._discrete
    >>>
    >>> meta = SimpleMetadata(
    ...     factors=np.array([[0, 1], [1, 0], [0, 1]]),
    ...     labels=np.array([0, 1, 0]),
    ...     names=["age_bin", "gender"],
    ...     discrete=[True, True],
    ... )
    >>> isinstance(meta, Metadata)
    True
    """

    @property
    def factor_names(self) -> SequenceLike[str]: ...

    @property
    def factor_data(self) -> NDArray[np.int64]: ...

    @property
    def class_labels(self) -> NDArray[np.intp]: ...

    @property
    def is_discrete(self) -> Sequence[bool]: ...


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


@runtime_checkable
class FeatureExtractor(Protocol):
    """
    Protocol defining a feature extraction function for drift detection.

    Feature extractors transform arbitrary input data types into arrays
    suitable for drift detection. This enables drift detection on non-array
    inputs such as datasets, metadata, or raw model outputs.

    Common use cases include:
    - Extracting model prediction uncertainties from raw data
    - Computing embeddings from a neural network layer
    - Extracting statistical features from metadata
    - Converting structured data to numeric representations

    Example
    -------
    Creating a feature extractor for model uncertainties:

    >>> import torch
    >>> import torch.nn as nn
    >>> from dataeval.protocols import FeatureExtractor
    >>>
    >>> class UncertaintyExtractor:
    ...     def __init__(self, model: nn.Module) -> None:
    ...         self.model = model
    ...
    ...     def __call__(self, data: Any, /) -> Array:
    ...         # Get model predictions
    ...         with torch.no_grad():
    ...             preds = self.model(torch.tensor(data))
    ...             # Compute uncertainty as entropy
    ...             probs = torch.softmax(preds, dim=-1)
    ...             uncertainty = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
    ...         return uncertainty.numpy()
    >>>
    >>> model = nn.Linear(10, 3)
    >>> extractor = UncertaintyExtractor(model)
    >>> isinstance(extractor, FeatureExtractor)
    True

    Creating a feature extractor for metadata:

    >>> class MetadataExtractor:
    ...     def __call__(self, metadata_list: list, /) -> Array:
    ...         import numpy as np
    ...
    ...         # Extract statistics from metadata
    ...         features = [[m.brightness, m.contrast] for m in metadata_list]
    ...         return np.array(features)
    >>>
    >>> extractor = MetadataExtractor()
    >>> isinstance(extractor, FeatureExtractor)
    True
    """

    def __call__(self, data: Any, /) -> Array:
        """
        Extract features from input data.

        Parameters
        ----------
        data : Any
            Input data to extract features from. Can be any type that the
            specific extractor implementation supports.

        Returns
        -------
        Array
            Extracted features as an array suitable for drift detection.
            Should have shape (n_samples, n_features) or be convertible to
            such a shape.
        """
        ...


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


@runtime_checkable
class EmbeddingEncoder(Protocol):
    """
    Protocol for embedding encoders that extract features from datasets.

    Implementations handle all backend-specific logic including:
    - Model/function management
    - Device handling (if applicable)
    - Transforms (preprocessing)
    - Batching strategy
    - Layer extraction (if applicable)

    The :meth:`encode` method supports both streaming and non-streaming modes
    via the ``stream`` parameter.

    Example
    -------
    Creating a custom encoder:

    >>> import numpy as np
    >>> from numpy.typing import NDArray
    >>> from dataeval.protocols import EmbeddingEncoder, Dataset
    >>>
    >>> class MyEncoder:
    ...     def __init__(self, batch_size: int = 32):
    ...         self._batch_size = batch_size
    ...
    ...     @property
    ...     def batch_size(self) -> int:
    ...         return self._batch_size
    ...
    ...     def encode(self, dataset, indices, stream=False):
    ...         def _generate():
    ...             for batch_start in range(0, len(indices), self._batch_size):
    ...                 batch_idx = list(indices[batch_start : batch_start + self._batch_size])
    ...                 results = []
    ...                 for idx in batch_idx:
    ...                     item = dataset[idx]
    ...                     image = item[0] if isinstance(item, tuple) else item
    ...                     results.append(np.asarray(image).flatten())
    ...                 yield batch_idx, np.vstack(results)
    ...
    ...         if stream:
    ...             return _generate()
    ...         return np.vstack([emb for _, emb in _generate()])
    >>>
    >>> encoder = MyEncoder(batch_size=32)
    >>> isinstance(encoder, EmbeddingEncoder)
    True
    """

    @property
    def batch_size(self) -> int:
        """
        Return the batch size used for encoding.

        Returns
        -------
        int
            Number of samples processed per batch during encoding.
        """
        ...

    @overload
    def encode(
        self,
        dataset: Dataset[tuple[ArrayLike, Any, Any]] | Dataset[ArrayLike],
        indices: Sequence[int],
        stream: Literal[True],
    ) -> Iterator[tuple[Sequence[int], Array]]: ...

    @overload
    def encode(
        self,
        dataset: Dataset[tuple[ArrayLike, Any, Any]] | Dataset[ArrayLike],
        indices: Sequence[int],
        stream: Literal[False] = ...,
    ) -> Array: ...

    def encode(
        self,
        dataset: Dataset[tuple[ArrayLike, Any, Any]] | Dataset[ArrayLike],
        indices: Sequence[int],
        stream: bool = False,
    ) -> Iterator[tuple[Sequence[int], Array]] | Array:
        """
        Encode images at specified indices to embeddings.

        Parameters
        ----------
        dataset : Dataset
            Dataset providing images to encode. Can return either
            (image, label, metadata) tuples or just images.
        indices : Sequence[int]
            Indices of images to encode from the dataset.
        stream : bool, default False
            If True, yields (batch_indices, batch_embeddings) tuples for
            memory-efficient streaming. If False (default), returns all
            embeddings as a single array.

        Returns
        -------
        Array or Iterator[tuple[Sequence[int], Array]]
            When stream=False: Embeddings array of shape (len(indices), embedding_dim).
            When stream=True: Iterator yielding (batch_indices, batch_embeddings) tuples.
        """
        ...


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


# ========== DRIFT UPDATE STRATEGIES ==========


@runtime_checkable
class UpdateStrategy(Protocol):
    """
    Protocol defining the interface for updating reference data in drift detectors.

    Update strategies control how drift detectors maintain their reference dataset
    as new data arrives. Implementations must provide a `__call__` method that
    updates the reference data based on new observations.

    Examples
    --------
    Creating a custom update strategy that keeps a moving average:

    >>> import numpy as np
    >>> from numpy.typing import NDArray
    >>> from dataeval.utils.arrays import flatten_samples
    >>>
    >>> class MovingAverageUpdate:
    ...     '''Update strategy that maintains a moving average of reference data.'''
    ...
    ...     def __init__(self, n: int, alpha: float = 0.9) -> None:
    ...         '''
    ...         Parameters
    ...         ----------
    ...         n : int
    ...             Maximum number of samples to maintain
    ...         alpha : float, default 0.9
    ...             Exponential moving average weight (0 < alpha < 1)
    ...         '''
    ...         self.n = n
    ...         self.alpha = alpha
    ...
    ...     def __call__(self, x_ref: NDArray[np.float32], x_new: NDArray[np.float32]) -> NDArray[np.float32]:
    ...         '''
    ...         Update reference data with exponential moving average.
    ...
    ...         Parameters
    ...         ----------
    ...         x_ref : NDArray[np.float32]
    ...             Current reference data of shape (n_ref, n_features)
    ...         x_new : NDArray[np.float32]
    ...             New observations of shape (n_new, n_features)
    ...
    ...         Returns
    ...         -------
    ...         NDArray[np.float32]
    ...             Updated reference data of shape (n_updated, n_features)
    ...         '''
    ...         x_new_flat = flatten_samples(x_new)
    ...         # Compute moving average for overlapping samples
    ...         n_overlap = min(len(x_ref), len(x_new_flat))
    ...         if n_overlap > 0:
    ...             x_ref[:n_overlap] = self.alpha * x_ref[:n_overlap] + (1 - self.alpha) * x_new_flat[:n_overlap]
    ...         # Append remaining new samples
    ...         result = np.concatenate([x_ref, x_new_flat[n_overlap:]], axis=0)
    ...         return result[-self.n :]

    Using a custom update strategy with a drift detector:

    >>> from dataeval.shift import DriftUnivariate
    >>> import numpy as np
    >>>
    >>> # Create reference data
    >>> ref_data = np.random.normal(0, 1, (100, 10))
    >>>
    >>> # Initialize drift detector with custom update strategy
    >>> update_strategy = MovingAverageUpdate(n=100, alpha=0.9)
    >>> detector = DriftUnivariate(ref_data, method="ks", update_strategy=update_strategy)
    >>>
    >>> # Detect drift on new data - reference will be updated automatically
    >>> new_data = np.random.normal(0.5, 1, (50, 10))
    >>> result = detector.predict(new_data)

    Notes
    -----
    Implementations should:
    - Accept current reference data and new observations
    - Return updated reference data with consistent shape
    - Handle edge cases (empty arrays, size mismatches)
    - Maintain internal state if needed (e.g., sample counts)
    - Ensure output size doesn't exceed configured limits

    See Also
    --------
    LastSeenUpdate : Built-in strategy keeping the last n samples
    ReservoirSamplingUpdate : Built-in strategy using reservoir sampling
    """

    def __call__(self, x_ref: NDArray[np.float32], x_new: NDArray[np.float32]) -> NDArray[np.float32]: ...


@runtime_checkable
class EvaluationSchedule(Protocol):
    """
    Protocol for determining evaluation points in sufficiency analysis.

    Implementations determine at which dataset sizes to train and
    evaluate the model during sufficiency analysis.

    Examples
    --------
    Custom scheduler evaluating at 0%, 50%, 100% of the dataset

    >>> class MidpointSchedule:
    ...     def get_step(self, dataset_length: int) -> np.typing.NDArray[np.intp]:
    ...         return np.array([0, dataset_length // 2, dataset_length - 1], dtype=np.intp)
    """

    def get_steps(self, dataset_length: int) -> np.typing.NDArray[np.intp]:
        """
        Calculate evaluation points for given dataset length.

        Parameters
        ----------
        dataset_length : int
            Total length of training dataset

        Returns
        -------
        NDArray[np.intp]
            Array of dataset sizes at which to evaluate, must be
            monotonically increasing and within [1, dataset_length]
        """
        ...


# ========== LOSS FUNCTIONS ==========


@runtime_checkable
class LossFn(Protocol):
    """
    Protocol for generic loss functions that can be used with PyTorch models.

    This is the base protocol for all loss functions. It supports both
    class-based (torch.nn.Module-like) and functional loss implementations.

    Examples
    --------
    Using built-in PyTorch loss:

    >>> import torch.nn as nn
    >>> loss_fn = nn.MSELoss()
    >>> isinstance(loss_fn, LossFn)
    True

    Creating a custom functional loss:

    >>> def custom_loss(y_true, y_pred):
    ...     return torch.mean((y_true - y_pred) ** 2)
    >>> isinstance(custom_loss, LossFn)
    True

    Creating a custom class-based loss:

    >>> class CustomLoss:
    ...     def __call__(self, y_true, y_pred):
    ...         return torch.mean((y_true - y_pred) ** 2)
    >>> loss_fn = CustomLoss()
    >>> isinstance(loss_fn, LossFn)
    True
    """

    def __call__(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor: ...


@runtime_checkable
class ReconstructionLossFn(Protocol):
    """
    Protocol for reconstruction-based loss functions (Autoencoder).

    Used for standard autoencoders that only return reconstruction.
    The loss function takes the original input and reconstruction.

    Examples
    --------
    Using MSE for reconstruction:

    >>> import torch
    >>> import torch.nn as nn
    >>> loss_fn = nn.MSELoss()
    >>> x = torch.randn(32, 1, 28, 28)
    >>> x_recon = torch.randn(32, 1, 28, 28)
    >>> loss = loss_fn(x, x_recon)

    Creating a custom reconstruction loss:

    >>> class CustomReconstructionLoss:
    ...     def __call__(self, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
    ...         return torch.mean(torch.abs(x - x_recon))
    >>> loss_fn = CustomReconstructionLoss()
    """

    def __call__(self, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss.

        Parameters
        ----------
        x : torch.Tensor
            Original input
        x_recon : torch.Tensor
            Reconstructed output

        Returns
        -------
        torch.Tensor
            Scalar loss value
        """
        ...


@runtime_checkable
class EvidenceLowerBoundLossFn(Protocol):
    """
    Protocol for Evidence Lower Bound (ELBO) loss functions.

    ELBO loss functions take the original input, reconstruction, mean (mu),
    and log-variance (logvar) to compute the ELBO loss.

    Examples
    --------
    Using the ELBO class:

    >>> from dataeval.utils.losses import ELBOLoss
    >>> loss_fn = ELBOLoss(beta=1.0)
    >>> x = torch.randn(32, 1, 28, 28)
    >>> x_recon = torch.randn(32, 1, 28, 28)
    >>> mu = torch.randn(32, 128)
    >>> logvar = torch.randn(32, 128)
    >>> loss = loss_fn(x, x_recon, mu, logvar)

    Creating a custom ELBO loss:

    >>> class CustomELBOLoss:
    ...     def __init__(self, beta: float = 1.0):
    ...         self.beta = beta
    ...
    ...     def __call__(
    ...         self, x: torch.Tensor, x_recon: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
    ...     ) -> torch.Tensor:
    ...         recon_loss = torch.mean((x - x_recon) ** 2)
    ...         kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    ...         return recon_loss + self.beta * kld_loss
    """

    def __call__(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute VAE loss (ELBO).

        Parameters
        ----------
        x : torch.Tensor
            Original input
        x_recon : torch.Tensor
            Reconstructed output
        mu : torch.Tensor
            Mean of latent distribution
        logvar : torch.Tensor
            Log-variance of latent distribution

        Returns
        -------
        torch.Tensor
            Scalar loss value (reconstruction + KL divergence)
        """
        ...


# ========== CALLBACKS ==========


class ProgressCallback(Protocol):
    """
    Protocol for a callable progress callback function.

    Parameters
    ----------
    step : int
        The current step or iteration number.
    total : int or None
        The total number of steps or iterations, if known.
    desc : str or None
        Optional description of the progress.
    extra_info : Mapping[str, Any] or None
        Optional dictionary of additional information.
    """

    def __call__(
        self,
        step: int,
        *,
        total: int | None = None,
        desc: str | None = None,
        extra_info: dict[str, Any] | None = None,
    ) -> None: ...
