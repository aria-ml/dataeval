"""Uncertainty-based feature extractor for drift detection."""

__all__ = []

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, softmax
from scipy.stats import entropy

from dataeval.protocols import Array, FeatureExtractor
from dataeval.types import ReprMixin
from dataeval.utils._array import as_numpy


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
        """Extract per-instance uncertainty scores of shape ``(n_samples, 1)``.

        Parameters
        ----------
        data : Any
            Passed straight through to the wrapped ``scores`` extractor, so the
            accepted input contract is whatever that extractor accepts -- for
            example an iterable of images, a full MAITE dataset (whose items are
            ``(image, target, metadata)`` tuples), a raw ``(n, n_classes)`` score
            array, or an :class:`~dataeval.Embeddings`.

        Returns
        -------
        NDArray[np.float32]
            Uncertainty scores of shape ``(n_samples, 1)``; an empty ``(0, 1)``
            array when the score producer yields no rows.
        """
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
        """Compute per-class uncertainty distributions; ``{}`` when data is empty.

        Parameters
        ----------
        data : Any
            Passed straight through to the wrapped ``scores`` extractor, so the
            accepted input contract is whatever that extractor accepts -- for
            example an iterable of images, a full MAITE dataset (whose items are
            ``(image, target, metadata)`` tuples), a raw
            ``(n_detections, n_classes)`` score array, or an
            :class:`~dataeval.Embeddings`.

        Returns
        -------
        dict[int, NDArray[np.float32]]
            Mapping from class index to a ``(n_detections, 1)`` array of
            uncertainty scores for detections assigned to that class; ``{}`` when
            the score producer yields no rows.
        """
        preds = self._score(data)
        if preds is None:
            return {}
        return _classwise_prediction_uncertainty(preds, self.preds_type, self.normalize, self.threshold)
