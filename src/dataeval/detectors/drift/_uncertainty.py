"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

__all__ = []

from collections.abc import Sequence
from typing import Literal, cast

import numpy as np
import torch
from scipy.special import softmax
from scipy.stats import entropy

from dataeval.config import DeviceLike, get_device
from dataeval.detectors.drift._base import BaseDrift, UpdateStrategy
from dataeval.detectors.drift._ks import DriftKS
from dataeval.outputs import DriftOutput
from dataeval.typing import Array, Transform
from dataeval.utils._array import as_numpy
from dataeval.utils.torch._internal import predict_batch


def classifier_uncertainty(
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
        if np.abs(1 - np.sum(preds_np, axis=-1)).mean() > 1e-6:
            raise ValueError("Probabilities across labels should sum to 1")
        probs = preds_np
    elif preds_type == "logits":
        probs = softmax(preds_np, axis=-1)
    else:
        raise NotImplementedError("Only prediction types 'probs' and 'logits' supported.")

    uncertainties = cast(np.ndarray, entropy(probs, axis=-1))
    return torch.as_tensor(uncertainties[:, None])


class DriftUncertainty(BaseDrift):
    """Drift detector using model prediction uncertainty.

    Detects drift by monitoring changes in the distribution of model prediction
    uncertainties (entropy) rather than input features directly. Uses
    :term:`Kolmogorov-Smirnov (K-S) Test` to compare uncertainty distributions
    between reference and test data.

    This approach is particularly effective for detecting drift that affects model
    confidence even when input features remain statistically similar, such as
    out-of-domain samples or adversarial examples.

    Parameters
    ----------
    data : Embeddings or Array
        Reference dataset used as baseline distribution for drift detection.
        Should represent the expected "normal" data distribution.
    p_val : float, default 0.05
        Significance threshold for statistical tests, between 0 and 1.
        For FDR correction, this represents the acceptable false discovery rate.
        Default 0.05 provides 95% confidence level for drift detection.
    update_strategy : UpdateStrategy or None, default None
        Strategy for updating reference data when new data arrives.
        When None, reference data remains fixed throughout detection.
    correction : "bonferroni" or "fdr", default "bonferroni"
        Multiple testing correction method for multivariate drift detection.
        "bonferroni" provides conservative family-wise error control by
        dividing significance threshold by number of features.
        "fdr" uses Benjamini-Hochberg procedure for less conservative control.
        Default "bonferroni" minimizes false positive drift detections.
    preds_type : "probs" or "logits", default "probs"
        Format of model prediction outputs. "probs" expects normalized
        probabilities summing to 1. "logits" expects raw model outputs
        and applies softmax normalization internally.
        Default "probs" assumes standard classification model outputs.
    batch_size : int, default 32
        Batch size for model inference during uncertainty computation.
        Larger batches improve GPU utilization but require more memory.
        Default 32 balances efficiency and memory usage.
    transforms : Transform, Sequence[Transform] or None, default None
        Data transformations applied before model inference. Should match
        preprocessing used during model training for consistent predictions.
        When None, uses raw input data without preprocessing.
    device : DeviceLike or None, default None
        Hardware device for computation. When None, automatically selects
        DataEval's configured device, falling back to PyTorch's default.

    Attributes
    ----------
    model : torch.nn.Module
        Classification model used for uncertainty computation.
    device : torch.device
        Hardware device used for model inference.
    batch_size : int
        Batch size for model predictions.
    preds_type : {"probs", "logits"}
        Format of model prediction outputs.

    Example
    -------
    >>> model = ClassificationModel()
    >>> drift_detector = DriftUncertainty(x_ref, model=model, batch_size=16)

    Verify reference images have not drifted

    >>> result = drift_detector.predict(x_test)
    >>> print(f"Drift detected: {result.drifted}")
    Drift detected: True

    >>> print(f"Mean uncertainty change: {result.distance:.4f}")
    Mean uncertainty change: 0.8160

    With data preprocessing

    >>> import torchvision.transforms.v2 as T
    >>> transforms = T.Compose([T.ToDtype(torch.float32)])
    >>> drift_detector = DriftUncertainty(x_ref, model=model, batch_size=16, transforms=transforms)

    Notes
    -----
    Uncertainty-based drift detection is complementary to feature-based methods.
    It can detect semantic drift (changes in data meaning) that may not be
    apparent in raw feature statistics, making it valuable for monitoring
    model performance in production environments.

    The method assumes that model uncertainty is a reliable indicator of
    data quality. This works best with well-calibrated models trained on
    representative data. Poorly calibrated models may produce misleading
    uncertainty estimates.

    For optimal performance, ensure the model and transforms match those used
    during training, and that the reference data represents the expected
    operational distribution where the model performs reliably.
    """

    def __init__(
        self,
        data: Array,
        model: torch.nn.Module,
        p_val: float = 0.05,
        update_strategy: UpdateStrategy | None = None,
        correction: Literal["bonferroni", "fdr"] = "bonferroni",
        preds_type: Literal["probs", "logits"] = "probs",
        batch_size: int = 32,
        transforms: Transform[torch.Tensor] | Sequence[Transform[torch.Tensor]] | None = None,
        device: DeviceLike | None = None,
    ) -> None:
        self.model: torch.nn.Module = model
        self.device: torch.device = get_device(device)
        self.batch_size: int = batch_size
        self.preds_type: Literal["probs", "logits"] = preds_type

        self._transforms = (
            [] if transforms is None else [transforms] if isinstance(transforms, Transform) else transforms
        )
        self._detector = DriftKS(
            data=self._preprocess(data).cpu().numpy(),
            p_val=p_val,
            update_strategy=update_strategy,
            correction=correction,
        )

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply preprocessing transforms to input data."""
        for transform in self._transforms:
            x = transform(x)
        return x

    def _preprocess(self, x: Array) -> torch.Tensor:
        """Convert input data to uncertainty scores via model predictions."""
        preds = predict_batch(x, self.model, self.device, self.batch_size, self._transform)
        return classifier_uncertainty(preds, self.preds_type)

    def predict(self, x: Array) -> DriftOutput:
        """Predict whether model uncertainty distribution has drifted.

        Computes prediction uncertainties for the input data and tests
        whether their distribution significantly differs from the reference
        uncertainty distribution using Kolmogorov-Smirnov test.

        Parameters
        ----------
        x : Array
            Batch of instances to test for uncertainty drift.

        Returns
        -------
        DriftOutput
            Drift detection results including overall prediction, p-values,
            test statistics, and feature-level analysis of uncertainty values.

        Notes
        -----
        The returned DriftOutput treats uncertainty values as "features" for
        consistency with the underlying KS test implementation, even though
        uncertainty-based drift typically involves univariate analysis.
        """
        return self._detector.predict(self._preprocess(x).cpu().numpy())
