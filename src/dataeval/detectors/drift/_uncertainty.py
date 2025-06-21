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
    """
    Evaluate model_fn on x and transform predictions to prediction uncertainties.

    Parameters
    ----------
    x : Array
        Batch of instances.
    model_fn : Callable
        Function that evaluates a :term:`classification<Classification>` model on x in a single call (contains
        batching logic if necessary).
    preds_type : "probs" | "logits", default "probs"
        Type of prediction output by the model. Options are 'probs' (in [0,1]) or
        'logits' (in [-inf,inf]).

    Returns
    -------
    NDArray
        A scalar indication of uncertainty of the model on each instance in x.
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
    """
    Test for a change in the number of instances falling into regions on which \
        the model is uncertain.

    Performs a K-S test on prediction entropies.

    Parameters
    ----------
    data : Array
        Data used as reference distribution.
    model : Callable
        :term:`Classification` model outputting class probabilities (or logits)
    p_val : float, default 0.05
        :term:`P-Value` used for the significance of the test.
    update_strategy : UpdateStrategy or None, default None
        Reference data can optionally be updated using an UpdateStrategy class. Update
        using the last n instances seen by the detector with LastSeenUpdateStrategy
        or via reservoir sampling with ReservoirSamplingUpdateStrategy.
    correction : "bonferroni" or "fdr", default "bonferroni"
        Correction type for multivariate data. Either 'bonferroni' or 'fdr' (False
        Discovery Rate).
    preds_type : "probs" or "logits", default "probs"
        Type of prediction output by the model. Options are 'probs' (in [0,1]) or
        'logits' (in [-inf,inf]).
    batch_size : int, default 32
        Batch size used to evaluate model. Only relevant when backend has been
        specified for batch prediction.
    transforms : Transform, Sequence[Transform] or None, default None
        Transform(s) to apply to the data.
    device : DeviceLike or None, default None
        Device type used. The default None tries to use the GPU and falls back on
        CPU if needed. Can be specified by passing either 'cuda' or 'cpu'.

    Example
    -------
    >>> model = ClassificationModel()
    >>> drift = DriftUncertainty(x_ref, model=model, batch_size=20)

    Verify reference images have not drifted

    >>> drift.predict(x_ref.copy()).drifted
    False

    Test incoming images for drift

    >>> drift.predict(x_test).drifted
    True
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
        for transform in self._transforms:
            x = transform(x)
        return x

    def _preprocess(self, x: Array) -> torch.Tensor:
        preds = predict_batch(x, self.model, self.device, self.batch_size, self._transform)
        return classifier_uncertainty(preds, self.preds_type)

    def predict(self, x: Array) -> DriftOutput:
        """
        Predict whether a batch of data has drifted from the reference data.

        Parameters
        ----------
        x : Array
            Batch of instances.

        Returns
        -------
        DriftUnvariateOutput
            Dictionary containing the drift prediction, :term:`p-value<P-Value>`, and threshold
            statistics.
        """
        return self._detector.predict(self._preprocess(x).cpu().numpy())
