"""Uncertainty-based feature extractor for drift detection."""

__all__ = []

from collections.abc import Sequence
from typing import Any, Literal, cast

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.special import expit, softmax
from scipy.stats import entropy

from dataeval._experimental import deprecated
from dataeval.config import get_device
from dataeval.protocols import Array, DeviceLike, FeatureExtractor, Transform
from dataeval.types import ReprMixin
from dataeval.utils._internal import as_numpy, iter_images
from dataeval.utils.training import predict


def _prediction_uncertainty(
    preds: Array, preds_type: Literal["probs", "logits"] = "probs", normalize: bool = True
) -> NDArray[np.float32]:
    """Convert model predictions to uncertainty scores using entropy.

    Computes prediction uncertainty as the entropy of the predicted class
    probability distribution. Higher entropy indicates greater model uncertainty,
    with maximum uncertainty at uniform distributions and minimum at confident
    single-class predictions.

    Parameters
    ----------
    preds : Array
        Model predictions for a batch of instances. For "probs" type, should
        contain class probabilities that sum to 1 across the last dimension.
        For "logits" type, contains raw model outputs before softmax.
    preds_type : "probs" or "logits", default "probs"
        Type of prediction values. "probs" expects probabilities in [0,1] that
        sum to 1. "logits" expects raw outputs in [-inf,inf] and applies softmax.
        Default "probs" assumes model outputs normalized probabilities.
    normalize : bool
        Whether or not to normalize the shannon entropy by the maximum possible
        entropy for the number of classes present in the logits array.

    Returns
    -------
    NDArray[np.float32]
        Uncertainty scores for each instance with shape (n_samples, 1).
        Values are always >= 0, with higher values indicating greater uncertainty.

    Raises
    ------
    ValueError
        If preds_type is "probs" but probabilities don't sum to 1 within tolerance.
    NotImplementedError
        If preds_type is not "probs" or "logits".

    Notes
    -----
    Uncertainty is computed as Shannon entropy: -sum(p * log(p)) where p are
    the predicted class probabilities. This provides a principled measure of
    model confidence that is widely used in uncertainty quantification.
    """
    preds_np = as_numpy(preds)
    if preds_type == "probs":
        if np.abs(1 - np.nan_to_num(np.nansum(preds_np, axis=-1))).mean() > 1e-6:
            raise ValueError("Probabilities across labels should sum to 1")
        probs = preds_np
    elif preds_type == "logits":
        probs = softmax(preds_np, axis=-1)
    else:
        raise NotImplementedError("Only prediction types 'probs' and 'logits' supported.")
    uncertainties = np.asarray(entropy(probs, axis=-1), dtype=np.float64)

    divisor = np.log(len(preds_np[-1])) if normalize else 1.0
    return (uncertainties[:, None] / divisor).astype(np.float32)


def _classwise_prediction_uncertainty(
    preds: Array, preds_type: Literal["probs", "logits"] = "probs", normalize: bool = True, threshold: float = 0.99
) -> dict[int, NDArray[np.float32]]:
    """Compute per-class prediction uncertainty scores using entropy.

    Groups detections by their predicted class and computes prediction
    uncertainty for each class independently. A detection is assigned to
    every class whose confidence meets or exceeds a ratio threshold relative
    to the maximum confidence for that detection. Lower threshold values
    allow more classes per detection; a threshold of 1.0 enforces
    single-class (winner-take-all) assignment.

    Parameters
    ----------
    preds : Array
        Model predictions for a batch of detections. Each element should
        contain per-class scores (logits or probabilities) for a single
        detection.
    preds_type : "probs" or "logits", default "probs"
        Type of prediction values. "probs" expects probabilities in [0, 1]
        that sum to 1. "logits" expects raw outputs in [-inf, inf] and
        applies softmax.
    normalize : bool, default True
        Whether to normalize Shannon entropy by the maximum possible entropy
        for the number of classes present.
    threshold : float, default 0.99
        Confidence ratio cutoff for class assignment. A detection is assigned
        to every class whose confidence is at least ``threshold``
        times the maximum confidence for that detection.

    Returns
    -------
    dict[int, NDArray[np.float32]]
        Mapping from class index to uncertainty scores (np.ndarray) for detections
        assigned to that class. Each value has shape (n_detections, 1). Classes with
        no detections above the assignment threshold are absent from the dict.
    """
    preds_array = as_numpy(preds)
    if preds_array.size == 0:
        return {}

    sigmoid = expit(preds_array)
    rescaled = sigmoid / sigmoid.max(axis=1, keepdims=True)
    mask = rescaled >= threshold

    classwise_uncertainties = {}
    for cl in np.where(mask.any(axis=0))[0]:
        rows = preds_array[mask[:, cl]]
        classwise_uncertainties[int(cl)] = _prediction_uncertainty(rows, preds_type, normalize)
    return classwise_uncertainties


class _UncertaintyBase(ReprMixin):
    """Shared scoring + config for uncertainty extractors.

    Wraps a :class:`~dataeval.protocols.FeatureExtractor` (``scores``) that turns
    raw data into per-instance class scores of shape ``(n, n_classes)``. The
    score producer owns all model/backend concerns (inference, device, batching,
    detection decoding); this layer only applies entropy. Any callable satisfying
    the ``FeatureExtractor`` protocol works -- :class:`TorchExtractor`,
    :class:`OnnxExtractor`, an :class:`~dataeval.Embeddings`, or a custom one.

    Running both per-instance and per-class uncertainty on the same data calls
    ``scores`` once each. The only expensive step is inference, so to avoid
    paying it twice, wrap ``scores`` in a caching :class:`~dataeval.Embeddings`
    and share that one instance between the two extractors -- the second call
    hits the cache.
    """

    def __init__(
        self,
        scores: FeatureExtractor,
        preds_type: Literal["probs", "logits"] = "logits",
        normalize: bool = True,
    ) -> None:
        self._scores = scores
        self.preds_type: Literal["probs", "logits"] = preds_type
        self.normalize = normalize

    def _repr_overrides(self) -> dict[str, str]:
        """Render ``scores`` as its class name instead of the full repr."""
        return {"scores": self._scores.__class__.__name__}

    def _score(self, data: Any) -> NDArray[Any] | None:
        """Run the score producer; return ``(n, n_classes)`` or ``None`` if empty."""
        preds = self._scores(data)
        return None if len(preds) == 0 else as_numpy(preds)


class UncertaintyExtractor(_UncertaintyBase):
    """Per-instance prediction entropy as a drift feature.

    Implements the :class:`~dataeval.protocols.FeatureExtractor` protocol:
    ``__call__`` returns a ``(n_samples, 1)`` array of Shannon-entropy
    uncertainty scores, suitable for :class:`~dataeval.shift.DriftUnivariate`.

    Parameters
    ----------
    scores : FeatureExtractor
        Producer of per-instance class scores ``(n, n_classes)``. Owns the model,
        backend, batching and any detection decoding.
    preds_type : "probs" or "logits", default "logits"
        Format of the scores. "logits" applies softmax before entropy; "probs"
        expects values that already sum to 1.
    normalize : bool, default True
        Normalize Shannon entropy by the maximum possible entropy for the number
        of classes present.

    Example
    -------
    >>> import numpy as np
    >>> from dataeval.extractors._uncertainty import UncertaintyExtractor
    >>>
    >>> class FixedScores:
    ...     def __call__(self, data):
    ...         return np.array([[2.0, 1.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    >>>
    >>> ex = UncertaintyExtractor(FixedScores(), preds_type="logits")
    >>> out = ex(None)
    >>> out.shape
    (2, 1)
    """

    def __call__(self, data: Any) -> NDArray[np.float32]:
        """Extract per-instance uncertainty scores of shape ``(n_samples, 1)``."""
        preds = self._score(data)
        if preds is None:
            return np.empty((0, 1), dtype=np.float32)
        return _prediction_uncertainty(preds, self.preds_type, self.normalize)


class ClasswiseUncertaintyExtractor(_UncertaintyBase):
    """Per-class prediction entropy distributions for detection models.

    Groups detections by predicted class and returns one uncertainty array per
    class. A detection is assigned to every class whose rescaled (sigmoid)
    confidence is at least ``threshold`` times its maximum, so a detection may
    contribute to multiple classes.

    ``__call__`` returns a ``dict``, so this is **not** a drift feature extractor:
    pick a single class's array out of the dict and feed that to a detector. (It
    will still pass ``isinstance(x, FeatureExtractor)`` at runtime, which only
    checks for ``__call__``; do not pass it to a drift detector directly.)

    To run both per-instance (:class:`UncertaintyExtractor`) and per-class
    uncertainty on the same data without paying for inference twice, wrap the
    ``scores`` extractor in a caching :class:`~dataeval.Embeddings` and share that
    one instance between both extractors.

    Parameters
    ----------
    scores : FeatureExtractor
        Producer of per-detection class scores ``(n_detections, n_classes)``.
    preds_type : "probs" or "logits", default "logits"
        Format of the scores.
    normalize : bool, default True
        Normalize Shannon entropy by the maximum possible entropy.
    threshold : float, default 0.99
        Confidence ratio cutoff for class assignment. ``1.0`` enforces
        single-class (winner-take-all) assignment; lower values allow more
        classes per detection.

    Example
    -------
    >>> import numpy as np
    >>> import torch.nn as nn
    >>> from dataeval.extractors import TorchExtractor, ClasswiseUncertaintyExtractor
    >>>
    >>> model = nn.Linear(16, 10)
    >>> scores = TorchExtractor(model, device="cpu", batch_size=8)
    >>> extractor = ClasswiseUncertaintyExtractor(scores, preds_type="logits")
    >>> per_class = extractor(np.random.randn(8, 16).astype(np.float32))
    >>> isinstance(per_class, dict)
    True
    """

    def __init__(
        self,
        scores: FeatureExtractor,
        preds_type: Literal["probs", "logits"] = "logits",
        normalize: bool = True,
        threshold: float = 0.99,
    ) -> None:
        super().__init__(scores, preds_type, normalize)
        self.threshold = threshold

    def __call__(self, data: Any) -> dict[int, NDArray[np.float32]]:
        """Compute per-class uncertainty distributions; ``{}`` when data is empty."""
        preds = self._score(data)
        if preds is None:
            return {}
        return _classwise_prediction_uncertainty(preds, self.preds_type, self.normalize, self.threshold)


def _classifier_uncertainty(
    preds: Array,
    preds_type: Literal["probs", "logits"] = "probs",
) -> torch.Tensor:
    """Convert model predictions to uncertainty scores using entropy.

    Computes prediction uncertainty as the entropy of the predicted class
    probability distribution. Higher entropy indicates greater model uncertainty,
    with maximum uncertainty at uniform distributions and minimum at confident
    single-class predictions.

    Parameters
    ----------
    preds : Array
        Model predictions for a batch of instances. For "probs" type, should
        contain class probabilities that sum to 1 across the last dimension.
        For "logits" type, contains raw model outputs before softmax.
    preds_type : "probs" or "logits", default "probs"
        Type of prediction values. "probs" expects probabilities in [0,1] that
        sum to 1. "logits" expects raw outputs in [-inf,inf] and applies softmax.
        Default "probs" assumes model outputs normalized probabilities.

    Returns
    -------
    torch.Tensor
        Uncertainty scores for each instance with shape (n_samples, 1).
        Values are always >= 0, with higher values indicating greater uncertainty.

    Raises
    ------
    ValueError
        If preds_type is "probs" but probabilities don't sum to 1 within tolerance.
    NotImplementedError
        If preds_type is not "probs" or "logits".

    Notes
    -----
    Uncertainty is computed as Shannon entropy: -sum(p * log(p)) where p are
    the predicted class probabilities. This provides a principled measure of
    model confidence that is widely used in uncertainty quantification.
    """
    preds_np = as_numpy(preds)
    if preds_type == "probs":
        if np.abs(1 - np.nan_to_num(np.nansum(preds_np, axis=-1))).mean() > 1e-6:
            raise ValueError("Probabilities across labels should sum to 1")
        probs = preds_np
    elif preds_type == "logits":
        probs = softmax(preds_np, axis=-1)
    else:
        raise NotImplementedError("Only prediction types 'probs' and 'logits' supported.")

    uncertainties = cast(np.ndarray, entropy(probs, axis=-1))
    return torch.as_tensor(uncertainties[:, None])


@deprecated(
    since="1.1",
    removal="2.0",
    alternative="UncertaintyExtractor or ClasswiseUncertaintyExtractor",
    details=(
        "Wrap a TorchExtractor (or any FeatureExtractor) in UncertaintyExtractor for per-instance "
        "uncertainty, or ClasswiseUncertaintyExtractor for per-class uncertainty."
    ),
)
class ClassifierUncertaintyExtractor:
    """
    Computes prediction entropy from a classification model for drift detection.

    .. deprecated:: 1.1
        Wrap a ``TorchExtractor`` (or any ``FeatureExtractor``) in
        :class:`UncertaintyExtractor` for per-instance uncertainty, or
        :class:`ClasswiseUncertaintyExtractor` for per-class uncertainty.
        ``ClassifierUncertaintyExtractor`` will be removed in version 2.0.

    This class implements the :class:`~dataeval.protocols.FeatureExtractor` protocol
    for use with drift detectors (e.g., :class:`~dataeval.shift.DriftUnivariate`).
    It computes prediction uncertainty (entropy) from a classification model.

    Uncertainty-based drift detection monitors changes in model confidence rather
    than raw input features. This approach is particularly effective for detecting
    drift that affects model performance even when input statistics remain similar,
    such as out-of-domain samples or adversarial examples.

    Parameters
    ----------
    model : torch.nn.Module
        Classification model to compute predictions and uncertainties.
        Should output class probabilities or logits.
    preds_type : "probs" or "logits", default "probs"
        Format of model outputs. "probs" expects normalized probabilities
        summing to 1. "logits" expects raw model outputs and applies softmax.
    batch_size : int, default 32
        Batch size for model inference. Larger batches improve GPU
        utilization but require more memory.
    transforms : Transform or Sequence[Transform] or None, default None
        Preprocessing transforms to apply before model inference. Should match
        preprocessing used during model training for consistent predictions.
    device : DeviceLike or None, default None
        Hardware device for computation. When None, uses DataEval's
        configured device or PyTorch's default.

    Attributes
    ----------
    model : torch.nn.Module
        The classification model used for predictions.
    preds_type : {"probs", "logits"}
        Format of model outputs.
    batch_size : int
        Batch size for inference.
    device : torch.device
        Hardware device for computation.

    See Also
    --------
    UncertaintyExtractor : Per-instance prediction entropy as a drift feature.
    ClasswiseUncertaintyExtractor : Per-class prediction entropy distributions.
    dataeval.shift.DriftUnivariate : Univariate drift detection with multiple statistical tests

    Notes
    -----
    The uncertainty extractor computes Shannon entropy: -sum(p * log(p)) where p
    are the predicted class probabilities. Higher entropy indicates greater model
    uncertainty.

    This approach works best with well-calibrated models trained on representative
    data. Poorly calibrated models may produce misleading uncertainty estimates
    that don't reliably indicate data quality issues.

    Uncertainty-based drift detection is complementary to feature-based methods
    and can detect semantic drift (changes in data meaning) that may not be
    apparent in raw feature statistics.

    Example
    -------
    Basic usage with DriftUnivariate

    >>> import numpy as np
    >>> import torch.nn as nn
    >>> from dataeval.shift import DriftUnivariate
    >>> from dataeval.extractors import ClassifierUncertaintyExtractor
    >>>
    >>> # Create dummy datasets
    >>> train_dataset = np.random.randn(100, 16).astype(np.float32)
    >>> test_dataset = np.random.randn(20, 16).astype(np.float32)
    >>>
    >>> # Create a simple model
    >>> model = nn.Sequential(nn.Linear(16, 10), nn.Softmax(dim=-1))
    >>>
    >>> # Create uncertainty feature extractor
    >>> uncertainty_extractor = ClassifierUncertaintyExtractor(model=model, preds_type="probs", batch_size=32)
    >>>
    >>> # Use with DriftUnivariate for uncertainty-based drift detection
    >>> drift_detector = DriftUnivariate(method="ks", extractor=uncertainty_extractor).fit(train_dataset)
    >>>
    >>> # Detect drift on new data
    >>> result = drift_detector.predict(test_dataset)
    >>> print(f"Drift detected: {result.drifted}")
    Drift detected: False

    With data preprocessing transforms

    >>> import torch
    >>>
    >>> # Create new datasets and model for this example
    >>> train_dataset = np.random.randn(100, 16).astype(np.float32)
    >>> test_dataset = np.random.randn(20, 16).astype(np.float32)
    >>> model = nn.Sequential(nn.Linear(16, 10), nn.Softmax(dim=-1))
    >>>
    >>> # Simple transform (no normalization needed for this dummy data)
    >>> transforms = lambda x: x.float() if not x.is_floating_point() else x
    >>>
    >>> uncertainty_extractor = ClassifierUncertaintyExtractor(model=model, transforms=transforms, device="cpu")
    >>>
    >>> drift_detector = DriftUnivariate(method="ks", extractor=uncertainty_extractor).fit(train_dataset)

    Using different statistical methods

    >>> # Create datasets and model for this example
    >>> train_dataset = np.random.randn(100, 16).astype(np.float32)
    >>> test_dataset = np.random.randn(20, 16).astype(np.float32)
    >>> model = nn.Sequential(nn.Linear(16, 10), nn.Softmax(dim=-1))
    >>> uncertainty_extractor = ClassifierUncertaintyExtractor(model=model, preds_type="probs", batch_size=32)
    >>>
    >>> # Use Cramér-von Mises test instead of Kolmogorov-Smirnov
    >>> drift_detector = DriftUnivariate(
    ...     method="cvm",  # More sensitive to overall distributional changes
    ...     extractor=uncertainty_extractor,
    ... ).fit(train_dataset)
    >>>
    >>> # Or use Mann-Whitney U test for robust median shift detection
    >>> drift_detector = DriftUnivariate(
    ...     method="mwu",  # Robust to outliers
    ...     extractor=uncertainty_extractor,
    ... ).fit(train_dataset)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        preds_type: Literal["probs", "logits"] = "probs",
        batch_size: int = 32,
        transforms: Transform[torch.Tensor] | Sequence[Transform[torch.Tensor]] | None = None,
        device: DeviceLike | None = None,
    ) -> None:
        self.model = model
        self.preds_type: Literal["probs", "logits"] = preds_type
        self.batch_size: int = batch_size
        self.device: DeviceLike = get_device(device)
        self._transforms: list[Transform[torch.Tensor]] = (
            [] if transforms is None else [transforms] if isinstance(transforms, Transform) else list(transforms)
        )

    def _apply_transforms(self, x: torch.Tensor) -> torch.Tensor:
        """Apply preprocessing transforms to input data."""
        for transform in self._transforms:
            x = transform(x)
        return x

    def __call__(self, data: Any) -> Array:
        """Extract uncertainty features from raw data.

        Parameters
        ----------
        data : Any
            Iterable of images, or a MAITE-style dataset whose items are
            ``(image, target, metadata)`` tuples. The image is extracted from
            position 0 of the tuple when present.

        Returns
        -------
        Array
            Uncertainty scores as numpy array of shape (n_samples, 1).
        """
        batch_images: list[np.ndarray] = [as_numpy(image) for image in iter_images(data)]
        if not batch_images:
            return np.empty((0, 1), dtype=np.float32)
        batch_array = np.stack(batch_images)
        preds = predict(batch_array, self.model, self.device, self.batch_size, self._apply_transforms)
        uncertainties = _classifier_uncertainty(preds[0] if isinstance(preds, tuple) else preds, self.preds_type)
        return uncertainties.cpu().numpy()

    def __repr__(self) -> str:
        """Return string representation of the extractor."""
        return (
            f"{self.__class__.__name__}("
            f"model={self.model.__class__.__name__}, "
            f"preds_type={self.preds_type!r}, "
            f"batch_size={self.batch_size}, "
            f"device={self.device})"
        )
