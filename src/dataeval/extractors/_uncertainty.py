"""
Uncertainty-based feature extractor for drift detection.
"""

__all__ = []

from collections.abc import Sequence
from typing import Any, Literal, cast

import numpy as np
import torch
from scipy.special import softmax
from scipy.stats import entropy

from dataeval.config import get_device
from dataeval.protocols import Array, DeviceLike, Transform
from dataeval.utils.arrays import as_numpy
from dataeval.utils.training import predict


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


class UncertaintyFeatureExtractor:
    """Feature extractor that converts data to model uncertainty scores.

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

    Example
    -------
    Basic usage with DriftUnivariate

    >>> import numpy as np
    >>> import torch.nn as nn
    >>> from dataeval.shift import DriftUnivariate
    >>> from dataeval.extractors import UncertaintyFeatureExtractor
    >>>
    >>> # Create dummy datasets
    >>> train_dataset = np.random.randn(100, 16).astype(np.float32)
    >>> test_dataset = np.random.randn(20, 16).astype(np.float32)
    >>>
    >>> # Create a simple model
    >>> model = nn.Sequential(nn.Linear(16, 10), nn.Softmax(dim=-1))
    >>>
    >>> # Create uncertainty feature extractor
    >>> uncertainty_extractor = UncertaintyFeatureExtractor(model=model, preds_type="probs", batch_size=32)
    >>>
    >>> # Use with DriftUnivariate for uncertainty-based drift detection
    >>> drift_detector = DriftUnivariate(train_dataset, method="ks", feature_extractor=uncertainty_extractor)
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
    >>> uncertainty_extractor = UncertaintyFeatureExtractor(model=model, transforms=transforms, device="cpu")
    >>>
    >>> drift_detector = DriftUnivariate(train_dataset, method="ks", feature_extractor=uncertainty_extractor)

    Using different statistical methods

    >>> # Create datasets and model for this example
    >>> train_dataset = np.random.randn(100, 16).astype(np.float32)
    >>> test_dataset = np.random.randn(20, 16).astype(np.float32)
    >>> model = nn.Sequential(nn.Linear(16, 10), nn.Softmax(dim=-1))
    >>> uncertainty_extractor = UncertaintyFeatureExtractor(model=model, preds_type="probs", batch_size=32)
    >>>
    >>> # Use Cramér-von Mises test instead of Kolmogorov-Smirnov
    >>> drift_detector = DriftUnivariate(
    ...     train_dataset,
    ...     method="cvm",  # More sensitive to overall distributional changes
    ...     feature_extractor=uncertainty_extractor,
    ... )
    >>>
    >>> # Or use Mann-Whitney U test for robust median shift detection
    >>> drift_detector = DriftUnivariate(
    ...     train_dataset,
    ...     method="mwu",  # Robust to outliers
    ...     feature_extractor=uncertainty_extractor,
    ... )

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

    See Also
    --------
    dataeval.shift.DriftUnivariate : Univariate drift detection with multiple statistical tests
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
            Raw input data to compute uncertainties for.

        Returns
        -------
        Array
            Uncertainty scores as numpy array of shape (n_samples, 1).
        """
        preds = predict(data, self.model, self.device, self.batch_size, self._apply_transforms)
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
