"""Common type protocols used for interoperability with DataEval."""

__all__ = [
    "AnnotatedDataset",
    "Model",
    "Array",
    "ArrayLike",
    "Chunker",
    "Dataset",
    "DatasetMetadata",
    "DatumMetadata",
    "DeviceLike",
    "EvaluationSchedule",
    "EvaluationStrategy",
    "EvidenceLowerBoundLossFn",
    "FeatureExtractor",
    "ImageClassificationDatum",
    "ImageClassificationDataset",
    "LossFn",
    "Matcher",
    "MetadataLike",
    "ModelMetadata",
    "ModelResetStrategy",
    "MultiobjectTrackingDatum",
    "MultiobjectTrackingDataset",
    "MultiobjectTrackingTarget",
    "ObjectDetectionDatum",
    "ObjectDetectionDataset",
    "ObjectDetectionTarget",
    "ProgressCallback",
    "ReconstructionLossFn",
    "SegmentationDatum",
    "SegmentationDataset",
    "SegmentationTarget",
    "SequenceLike",
    "SingleFrameObjectTrackingTarget",
    "Threshold",
    "ThresholdBounds",
    "ThresholdLike",
    "ThresholdLimits",
    "TrainingStrategy",
    "Transform",
    "UpdateStrategy",
    "VideoFrame",
    "VideoStream",
]


from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    TypeAlias,
    TypeVar,
    overload,
    runtime_checkable,
)

import maite.protocols
import maite.protocols.multiobject_tracking
import maite.protocols.object_detection
import numpy as np
import torch
from numpy.typing import NDArray
from typing_extensions import Self

if TYPE_CHECKING:
    from dataeval.types import Correspondence, OntologyConcept

# ========== MAITE RE-EXPORTS ==========
#
# DataEval re-exports these MAITE protocols/types under ``dataeval.protocols`` so
# downstream code imports a single, stable namespace. They are aliased (not
# subclassed) to preserve object identity: ``isinstance`` checks, ``TypeVar``
# constraints, and TypedDict construction behave exactly as the MAITE originals.
# The docstrings below give a short DataEval-facing summary and link to the
# canonical MAITE reference via intersphinx.

ObjectDetectionTarget: TypeAlias = maite.protocols.object_detection.ObjectDetectionTarget
"""
Object-detection target for a single image.

- ``boxes`` : :obj:`ArrayLike` of shape (N, 4) - Bounding boxes in (x0, y0, x1, y1) format
- ``labels`` : :obj:`ArrayLike` of shape (N,) - Class labels for each bounding box
- ``scores`` : :obj:`ArrayLike` of shape (N,) - Confidence scores for each bounding box
"""

MultiobjectTrackingTarget: TypeAlias = maite.protocols.multiobject_tracking.MultiobjectTrackingTarget
"""
Set of tracked objects over a sequence of video frames.
"""

SingleFrameObjectTrackingTarget: TypeAlias = maite.protocols.multiobject_tracking.SingleFrameObjectTrackingTarget
"""
Single-frame object-tracking target (tracked objects within one frame).
"""

VideoFrame: TypeAlias = maite.protocols.multiobject_tracking.VideoFrame
"""
Contents of a single decoded video frame.
"""

VideoStream: TypeAlias = maite.protocols.multiobject_tracking.VideoStream
"""
Iterable of :obj:`VideoFrame` representing a single video.
"""

DatumMetadata: TypeAlias = maite.protocols.DatumMetadata
"""
Metadata associated with a single datum (item-level).

A :class:`~typing.TypedDict` with the following keys:

- ``id`` : ``int | str`` (required, read-only) - Unique identifier for the datum.

Implementations may add further string-keyed entries; only ``id`` is required by
the protocol. Extra keys are passed through unchanged.
"""

DatasetMetadata: TypeAlias = maite.protocols.DatasetMetadata
"""
Metadata associated with a dataset (collection-level).

A :class:`~typing.TypedDict` with the following keys:

- ``id`` : ``str`` (required, read-only) - Unique identifier for the dataset.
- ``index2label`` : ``dict[int, str]`` (optional, read-only) - Mapping from integer
  class index to the corresponding human-readable label name.

Implementations may add further string-keyed entries; only ``id`` is required by
the protocol. Extra keys are passed through unchanged.
"""

ModelMetadata: TypeAlias = maite.protocols.ModelMetadata
"""
Metadata associated with a model.

A :class:`~typing.TypedDict` with the following keys:

- ``id`` : ``str`` (required, read-only) - Unique identifier for the model.
- ``index2label`` : ``dict[int, str]`` (optional, read-only) - Mapping from integer
  class index to the corresponding human-readable label name the model predicts.

Implementations may add further string-keyed entries; only ``id`` is required by
the protocol. Extra keys are passed through unchanged.
"""

_InputType = TypeVar("_InputType", contravariant=True)
_TargetType = TypeVar("_TargetType", covariant=True)

Model: TypeAlias = maite.protocols.Model[_InputType, _TargetType]
"""
Model protocol specifying batch inference behavior over data.

Re-export of the generic MAITE ``Model`` protocol. Use bare for any model, or
specialize the input/target types for a concrete task — e.g.
``Model[ArrayLike, ObjectDetectionTarget]``.
"""

ArrayLike: TypeAlias = np.typing.ArrayLike
"""
Type alias for a `Union` representing objects that can be coerced into an array.

See Also
--------
:obj:`numpy.typing.ArrayLike`
"""

DeviceLike: TypeAlias = int | str | tuple[str, int] | torch.device
"""
Type alias for a `Union` representing types that specify a torch.device.

See Also
--------
:class:`torch.device`
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
    def ndim(self) -> int:
        """Number of dimensions of the array."""
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the array."""
        ...

    def __array__(self) -> NDArray[Any]:
        """Convert to numpy array."""
        ...

    def __getitem__(self, key: Any, /) -> Any:
        """Return item at key."""
        ...

    def __iter__(self) -> Iterator[Any]:
        """Return iterator."""
        ...

    def __len__(self) -> int:
        """Return length."""
        ...


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

    def __iter__(self) -> Iterator[_T_co]:
        """Return an iterator over the sequence."""
        ...

    def __len__(self) -> int:
        """Return the length of the sequence."""
        ...


@runtime_checkable
class SegmentationTarget(Protocol):
    """Protocol for targets in a Segmentation dataset."""

    @property
    def mask(self) -> ArrayLike:
        """:obj:`ArrayLike` segmentation mask."""
        ...

    @property
    def labels(self) -> ArrayLike:
        """:obj:`ArrayLike` class labels."""
        ...

    @property
    def scores(self) -> ArrayLike:
        """:obj:`ArrayLike` prediction scores."""
        ...


# ========== METADATA ==========


@runtime_checkable
class MetadataLike(Protocol):
    """
    Minimal protocol for metadata objects used in bias and quality analysis.

    This protocol defines the minimum interface required for metadata objects
    to be used with DataEval's bias evaluators (Balance, Diversity, Parity)
    and quality evaluators (Outliers). Users can implement lightweight custom
    metadata containers that satisfy this protocol.

    Attributes
    ----------
    index2label : NotRequired[Mapping[int, str]]
        Optional mapping from class label indices to human-readable names.
    item_indices : NotRequired[NDArray[np.intp]]
        Optional array mapping each label back to its source item/image.
        If not provided, a 1:1 mapping is assumed (one label per image).

    Example
    -------
    Creating a simple metadata container:

    >>> import numpy as np
    >>> from dataeval.protocols import MetadataLike
    >>>
    >>> class SimpleMetadata(MetadataLike):
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
    >>> isinstance(meta, MetadataLike)
    True
    """

    @property
    def factor_names(self) -> SequenceLike[str]:
        """Names of the metadata factors."""
        ...

    @property
    def factor_data(self) -> NDArray[np.int64]:
        """
        Metadata factors in array of shape (n_samples, n_factors).

        Continuous factors or non-integer data should be preprocessed into
        discrete integer bins before being returned here.
        """
        ...

    @property
    def class_labels(self) -> NDArray[np.intp]:
        """
        Flat array of class labels with one entry per target/detection.

        For image classification, length equals number of images.
        For object detection, length equals total detections across all images.
        """
        ...

    @property
    def is_discrete(self) -> Sequence[bool]:
        """
        Whether each factor is discrete (True) or continuous (False).

        Must have the same length as factor_names.
        """
        ...


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

    def __getitem__(self, index: int, /) -> _T_co:
        """Return item at index."""
        ...

    def __len__(self) -> int:
        """Return length."""
        ...


@runtime_checkable
class AnnotatedDataset(Dataset[_T_co], Protocol[_T_co]):
    """
    Protocol for a generic `AnnotatedDataset`.

    Methods
    -------
    __getitem__(index: int)
        Returns datum at specified index.
    __len__()
        Returns dataset length.

    Notes
    -----
    Inherits from :class:`.Dataset`. Matches :class:`maite.protocols.Dataset` structurally.
    """

    @property
    def metadata(self) -> DatasetMetadata:
        """:obj:`.DatasetMetadata` or derivatives."""
        ...


# ========== IMAGE CLASSIFICATION DATASETS ==========


ImageClassificationDatum: TypeAlias = tuple[ArrayLike, ArrayLike, DatumMetadata]
"""
Type alias for an image classification datum tuple.

- :obj:`ArrayLike` of shape (C, H, W) - Image data in channel, height, width format.
- :obj:`ArrayLike` of shape (N,) - Class label as one-hot encoded ground-truth or prediction confidences.
- :obj:`DatumMetadata` - Datum level metadata.
"""


ImageClassificationDataset: TypeAlias = AnnotatedDataset[ImageClassificationDatum]
"""
Type alias for an :class:`AnnotatedDataset` of :obj:`ImageClassificationDatum` elements.
"""

# ========== OBJECT DETECTION DATASETS ==========


ObjectDetectionDatum: TypeAlias = tuple[ArrayLike, ObjectDetectionTarget, DatumMetadata]
"""
Type alias for an object detection datum tuple.

- :obj:`ArrayLike` of shape (C, H, W) - Image data in channel, height, width format.
- :obj:`ObjectDetectionTarget` - Object detection target information for the image.
- :obj:`DatumMetadata` - Datum level metadata.
"""


ObjectDetectionDataset: TypeAlias = AnnotatedDataset[ObjectDetectionDatum]
"""
Type alias for an :class:`AnnotatedDataset` of :obj:`ObjectDetectionDatum` elements.
"""


# ========== SEGMENTATION DATASETS ==========


SegmentationDatum: TypeAlias = tuple[ArrayLike, SegmentationTarget, DatumMetadata]
"""
Type alias for a segmentation datum tuple.

- :obj:`ArrayLike` of shape (C, H, W) - Image data in channel, height, width format.
- :class:`SegmentationTarget` - Segmentation target information for the image.
- :obj:`DatumMetadata` - Datum level metadata.
"""

SegmentationDataset: TypeAlias = AnnotatedDataset[SegmentationDatum]
"""
Type alias for an :class:`AnnotatedDataset` of :obj:`SegmentationDatum` elements.
"""


# ========== MULTI-OBJECT TRACKING DATASETS ==========


MultiobjectTrackingDatum: TypeAlias = tuple[VideoStream, MultiobjectTrackingTarget, DatumMetadata]
"""
Type alias for a multi-object tracking datum tuple.

- :obj:`VideoStream` - An iterable of :obj:`VideoFrame` for a single video.
- :obj:`MultiobjectTrackingTarget` - Tracked objects across the sequence of frames.
- :obj:`DatumMetadata` - Datum level metadata.
"""


MultiobjectTrackingDataset: TypeAlias = AnnotatedDataset[MultiobjectTrackingDatum]
"""
Type alias for an :class:`AnnotatedDataset` of :obj:`MultiobjectTrackingDatum` elements.
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

    def __call__(self, data: _T, /) -> _T:
        """Apply transform to data."""
        ...


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


# ========== SUFFICIENCY STRATEGIES ==========


_M = TypeVar("_M")


@runtime_checkable
class ModelResetStrategy(Protocol[_M]):
    """
    Protocol for resetting model parameters between training runs.

    This protocol enables backend-agnostic model reset functionality.
    Implementations can provide custom reset logic for any ML framework
    (PyTorch, TensorFlow, JAX, etc.).

    For PyTorch models (nn.Module), a default implementation is provided
    that calls reset_parameters() on each layer. For other backends,
    users must provide their own reset strategy.

    See Also
    --------
    :class:`~dataeval.performance.Sufficiency` : Uses this protocol for model reset between runs

    Examples
    --------
    Custom reset for PyTorch with specific initialization:

    >>> import torch.nn as nn
    >>> class XavierReset:
    ...     def __call__(self, model: nn.Module) -> nn.Module:
    ...         for m in model.modules():
    ...             if hasattr(m, "weight") and m.weight is not None:
    ...                 nn.init.xavier_uniform_(m.weight)
    ...             if hasattr(m, "bias") and m.bias is not None:
    ...                 nn.init.zeros_(m.bias)
    ...         return model

    Reset strategy for JAX models using parameter reinitialization:

    >>> class JAXReset:
    ...     def __init__(self, init_fn, rng_key):
    ...         self.init_fn = init_fn
    ...         self.rng_key = rng_key
    ...
    ...     def __call__(self, params):
    ...         import jax.random as random
    ...
    ...         # Reinitialize parameters with new random key
    ...         self.rng_key, subkey = random.split(self.rng_key)
    ...         return self.init_fn(subkey)

    Reset by reloading model weights from checkpoint:

    >>> class CheckpointReset:
    ...     def __init__(self, checkpoint_path: str):
    ...         self.checkpoint_path = checkpoint_path
    ...
    ...     def __call__(self, model: nn.Module) -> nn.Module:
    ...         import torch
    ...
    ...         model.load_state_dict(torch.load(self.checkpoint_path))
    ...         return model
    """

    def __call__(self, model: _M) -> _M:
        """
        Reset model parameters to initial state.

        Parameters
        ----------
        model : M
            The model to reset. Can be any model type (PyTorch Module,
            TensorFlow model, JAX parameters, etc.).

        Returns
        -------
        M
            The reset model. May be the same instance with modified
            parameters or a new instance entirely.

        Notes
        -----
        Implementations should:
        - Return the model in a state equivalent to freshly initialized
        - Handle the specific backend's requirements for parameter reset
        - Be deterministic or set seeds internally for reproducibility
        """
        ...


@runtime_checkable
class TrainingStrategy(Protocol[_T_cn]):
    """
    Protocol defining the interface for training a model on a dataset subset.

    Implementations must provide a `train` method with this signature.
    Uses structural typing - no explicit inheritance required.

    The @runtime_checkable decorator allows isinstance() checks if needed,
    though structural typing works without it at type-check time.

    The model parameter accepts any type to support different ML backends
    (PyTorch, TensorFlow, JAX, etc.).

    Examples
    --------
    Creating a custom training strategy for PyTorch:

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

    Creating a training strategy for JAX:

    >>> class JAXTraining:
    ...     def __init__(self, learning_rate: float, epochs: int):
    ...         self.learning_rate = learning_rate
    ...         self.epochs = epochs
    ...
    ...     def train(self, params, dataset: Dataset, indices: Sequence[int]) -> None:
    ...         import jax
    ...         import jax.numpy as jnp
    ...
    ...         # JAX training with functional updates
    ...         for epoch in range(self.epochs):
    ...             for idx in indices:
    ...                 x, y, _ = dataset[idx]
    ...                 grads = jax.grad(loss_fn)(params, x, y)
    ...                 params = jax.tree.map(lambda p, g: p - self.learning_rate * g, params, grads)
    """

    def train(self, model: Any, dataset: Dataset[_T_cn], indices: Sequence[int]) -> None:
        """
        Train the model using the specified indices from the dataset.

        Parameters
        ----------
        model : Any
            The model to train. Can be any model type (PyTorch Module,
            TensorFlow model, etc.). Training should modify the model in-place
            when the backend supports it.
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

    The model parameter accepts any type to support different ML backends
    (PyTorch, TensorFlow, JAX, etc.).

    Examples
    --------
    Creating a custom evaluation strategy for PyTorch:

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

    Creating an evaluation strategy for JAX:

    >>> class JAXEvaluation:
    ...     def __init__(self, apply_fn):
    ...         self.apply_fn = apply_fn  # JAX model's forward function
    ...
    ...     def evaluate(self, params, dataset: Dataset) -> Mapping[str, float]:
    ...         import jax.numpy as jnp
    ...
    ...         correct = 0
    ...         total = len(dataset)
    ...         for i in range(total):
    ...             x, y, _ = dataset[i]
    ...             pred = self.apply_fn(params, x)
    ...             if jnp.argmax(pred) == jnp.argmax(y):
    ...                 correct += 1
    ...         return {"accuracy": correct / total}
    """

    def evaluate(self, model: Any, dataset: Dataset[_T_cn]) -> Mapping[str, float | ArrayLike]:
        """
        Evaluate the model on the dataset and return performance metrics.

        Parameters
        ----------
        model : Any
            The trained model to evaluate. Can be any model type (PyTorch
            Module, TensorFlow model, JAX parameters, etc.).
        dataset : Dataset[T]
            The dataset to evaluate on (typically a test/validation set)

        Returns
        -------
        Mapping[str, float | ArrayLike]
            Mapping of metric names to values. Each value is either:
            - A scalar (float) for single-class metrics
            - An array (np.ndarray) for per-class or per-sample metrics
            - Examples:

                `{"accuracy": 0.95}`  # Single metric
                `{"accuracy": 0.95, "precision": 0.93, "recall": 0.94}`  # Multiple metrics
                `{"accuracy": np.array([0.9, 0.85, 0.92])}`  # Per-class metrics

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

    See Also
    --------
    :class:`~dataeval.shift.update_strategies.LastSeenUpdateStrategy` : Built-in strategy keeping the last n samples
    :class:`~dataeval.shift.update_strategies.ReservoirSamplingUpdateStrategy` : Built-in strategy using reservoir sampling

    Notes
    -----
    Implementations should:
    - Accept current reference data and new observations
    - Return updated reference data with consistent shape
    - Handle edge cases (empty arrays, size mismatches)
    - Maintain internal state if needed (e.g., sample counts)
    - Ensure output size doesn't exceed configured limits

    Examples
    --------
    Creating a custom update strategy that keeps a moving average:

    >>> import numpy as np
    >>> from numpy.typing import NDArray
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
    ...     def __call__(self, reference_data: NDArray[np.float32], data: NDArray[np.float32]) -> NDArray[np.float32]:
    ...         '''
    ...         Update reference data with exponential moving average.
    ...
    ...         Parameters
    ...         ----------
    ...         reference_data : NDArray[np.float32]
    ...             Current reference data of shape (n_ref, n_features)
    ...         data : NDArray[np.float32]
    ...             New observations of shape (n_new, n_features)
    ...
    ...         Returns
    ...         -------
    ...         NDArray[np.float32]
    ...             Updated reference data of shape (n_updated, n_features)
    ...         '''
    ...         data_flat = np.atleast_2d(np.asarray(data, dtype=np.float32))
    ...         # Compute moving average for overlapping samples
    ...         n_overlap = min(len(reference_data), len(data_flat))
    ...         if n_overlap > 0:
    ...             reference_data[:n_overlap] = (
    ...                 self.alpha * reference_data[:n_overlap] + (1 - self.alpha) * data_flat[:n_overlap]
    ...             )
    ...         # Append remaining new samples
    ...         result = np.concatenate([reference_data, data_flat[n_overlap:]], axis=0)
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
    >>> detector = DriftUnivariate(method="ks", update_strategy=update_strategy).fit(ref_data)
    >>>
    >>> # Detect drift on new data - reference will be updated automatically
    >>> new_data = np.random.normal(0.5, 1, (50, 10))
    >>> result = detector.predict(new_data)
    """  # noqa: E501

    def __call__(self, reference_data: NDArray[np.float32], data: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return updated reference data."""
        ...


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
    ...     def get_steps(self, dataset_length: int) -> np.typing.NDArray[np.intp]:
    ...         return np.array([0, dataset_length // 2, dataset_length - 1], dtype=np.intp)
    """

    def get_steps(self, dataset_length: int) -> np.typing.NDArray[np.intp]:
        """
        Compute evaluation points for given dataset length.

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

    def __call__(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        """Compute loss."""
        ...


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


# ========== CHUNKERS =========


@runtime_checkable
class Chunker(Protocol):
    """
    Protocol for chunking datasets into subsets by returning index arrays.

    Implementations must provide a `__call__` method that takes the number
    of samples and returns a list of index arrays representing the chunks.

    Examples
    --------
    Creating a simple chunker that splits the dataset into equal parts:

    >>> import numpy as np
    >>> from dataeval.protocols import Chunker
    >>>
    >>> class EqualChunker:
    ...     def __init__(self, n_chunks: int):
    ...         self.n_chunks = n_chunks
    ...
    ...     def __call__(self, n: int) -> list[NDArray[np.intp]]:
    ...         return [idx.astype(np.intp) for idx in np.array_split(np.arange(n), self.n_chunks)]
    >>>
    >>> chunker = EqualChunker(n_chunks=5)
    >>> isinstance(chunker, Chunker)
    True
    """

    def __call__(self, n: int) -> list[NDArray[np.intp]]:
        """
        Split n samples into chunks, returning index arrays.

        Parameters
        ----------
        n : int
            Number of samples to chunk.

        Returns
        -------
        list[NDArray[np.intp]]
            List of index arrays, each containing integer indices for one chunk.
            The union of all index arrays should cover ``range(n)`` without overlap.
        """
        ...


# ========== CALLBACKS ==========


@runtime_checkable
class ProgressCallback(Protocol):
    """
    Protocol for a callable progress callback function.

    Examples
    --------
    Creating a simple progress callback:

    >>> from dataeval.protocols import ProgressCallback
    >>>
    >>> class PrintProgress:
    ...     def __call__(
    ...         self,
    ...         step: int,
    ...         *,
    ...         total: int | None = None,
    ...         desc: str | None = None,
    ...         extra_info: dict[str, Any] | None = None,
    ...     ) -> None:
    ...         pct = f" ({step}/{total})" if total else ""
    ...         prefix = f"{desc}: " if desc else ""
    ...         print(f"{prefix}Step {step}{pct}")
    >>>
    >>> callback = PrintProgress()
    >>> isinstance(callback, ProgressCallback)
    True
    """

    def __call__(
        self,
        step: int,
        *,
        total: int | None = None,
        desc: str | None = None,
        extra_info: dict[str, Any] | None = None,
    ) -> None:
        """
        Call the progress callback with the current step information.

        Parameters
        ----------
        step : int
            The current step or iteration number.
        total : int or None, optional
            The total number of steps or iterations, if known.
        desc : str or None, optional
            Optional description of the progress.
        extra_info : Mapping[str, Any] or None, optional
            Optional dictionary of additional information.

        Returns
        -------
        None
            This method is intended for side effects (e.g., updating a progress bar).
        """
        ...


# ========== THRESHOLD ==========


@runtime_checkable
class Threshold(Protocol):
    """
    Protocol for threshold objects used in bias and quality evaluators.

    This protocol defines the interface for threshold objects that determine
    whether a given score or metric exceeds a predefined threshold, indicating
    potential bias or quality issues.

    Methods
    -------
    __call__(self, data: NDArray[Any]) -> tuple[float | None, float | None]
        Return the lower and upper threshold values based on the input data.
    """

    def __call__(self, data: NDArray[Any]) -> tuple[float | None, float | None]:
        """
        Compute threshold values based on input data.

        Parameters
        ----------
        data : NDArray[Any]
            Input data used to compute the threshold. The specific requirements
            for this data depend on the implementation of the threshold.

        Returns
        -------
        tuple[float | None, float | None]
            A tuple containing the lower and upper threshold values. If a particular
            threshold is not applicable, it can be set to None.
        """
        ...


# ========== ONTOLOGY ALIGNMENT ==========


@runtime_checkable
class Matcher(Protocol):
    """
    Protocol for an element-level matcher used in ontology alignment.

    A matcher proposes candidate :class:`~dataeval.types.Correspondence` objects
    between a *source* and a *target* vocabulary, each supplied as an iterable of
    :class:`~dataeval.types.OntologyConcept` (an :class:`~dataeval.Ontology`
    satisfies this directly). It is the extension seam of
    :func:`dataeval.core.label_alignment`: exact terminological anchoring is built
    into ``label_alignment`` itself, while additional matchers (string-similarity,
    and later embedding / instance-based matchers) are supplied via its
    ``matchers`` argument and consulted for source concepts the exact pass left
    unanchored.

    A matcher needs only each concept's ``id``, ``label``, and ``synonyms``
    (``parents`` is available for light structural hints) so it can be written and
    tested against a plain list of concepts.

    Implementations should be permissive — propose any plausible correspondence
    with a calibrated ``confidence`` — and let ``label_alignment`` apply the
    acceptance threshold and pick the best proposal per source concept. A matcher
    need not deduplicate or resolve conflicts itself.

    Example
    -------
    A trivial matcher proposing an equivalence for an exact id match:

    >>> from dataeval.types import Correspondence
    >>> from dataeval.protocols import Matcher
    >>>
    >>> class IdMatcher:
    ...     def __call__(self, source, target):
    ...         target_ids = {c.id for c in target}
    ...         return [
    ...             Correspondence(source=c.id, target=c.id, relation="equivalent", matcher="id")
    ...             for c in source
    ...             if c.id in target_ids
    ...         ]
    >>>
    >>> isinstance(IdMatcher(), Matcher)
    True
    """

    def __call__(
        self,
        source: "Iterable[OntologyConcept]",
        target: "Iterable[OntologyConcept]",
    ) -> "Sequence[Correspondence]":
        """
        Propose candidate correspondences from ``source`` concepts to ``target``.

        Parameters
        ----------
        source : Iterable[OntologyConcept]
            The vocabulary being mapped *from*. May be traversed more than once,
            so pass a re-iterable collection (an :class:`~dataeval.Ontology` or a
            list), not a one-shot iterator.
        target : Iterable[OntologyConcept]
            The reference vocabulary being mapped *to*.

        Returns
        -------
        Sequence[Correspondence]
            Candidate correspondences, each with a ``confidence`` in ``[0, 1]``.
            ``label_alignment`` filters by threshold and keeps the best per source
            concept.
        """
        ...


ThresholdBounds: TypeAlias = float | tuple[float | None, float | None] | None
ThresholdLimits: TypeAlias = tuple[float | None, float | None]
ThresholdLike: TypeAlias = (
    str
    | ThresholdBounds
    | tuple[str, ThresholdBounds]
    | tuple[str, ThresholdBounds | None, ThresholdLimits]
    | tuple[ThresholdBounds | None, ThresholdLimits]
    | Threshold
)
"""Type alias for threshold specifications.

Values default to modified z-score thresholds if not provided.

- ``float``: symmetric multiplier (same for lower and upper)
- ``str``: named threshold (e.g., "modzscore") with default bounds
- ``tuple[float | None, float | None]``: ``(lower, upper)`` for asymmetric bounds
- ``tuple[str, float | tuple[float | None, float | None]]``: named threshold with optional lower and upper bounds
- ``tuple[str, bounds, (lower_limit, upper_limit)]``: named threshold with bounds and limit clamping,
  e.g. ``("zscore", (1.0, 3.5), (0.0, 1.0))``. Pass ``None`` for bounds to use defaults:
  ``("zscore", None, (0.0, 1.0))``
- ``tuple[bounds | None, (lower_limit, upper_limit)]``: default threshold with bounds and limit clamping,
  e.g. ``(2.5, (0.0, 1.0))`` or ``(None, (0.0, 1.0))`` for default multiplier
- ``Threshold``: a fully configured Threshold instance
"""
