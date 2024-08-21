"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from functools import partial
from typing import Callable, Dict, Literal, Optional, Union

import numpy as np
from scipy.special import softmax
from scipy.stats import entropy

from dataeval._internal.interop import ArrayLike

from .base import UpdateStrategy
from .ks import DriftKS
from .torch import get_device, preprocess_drift


def classifier_uncertainty(
    x: np.ndarray,
    model_fn: Callable,
    preds_type: Literal["probs", "logits"] = "probs",
) -> np.ndarray:
    """
    Evaluate model_fn on x and transform predictions to prediction uncertainties.

    Parameters
    ----------
    x
        Batch of instances.
    model_fn
        Function that evaluates a classification model on x in a single call (contains
        batching logic if necessary).
    preds_type
        Type of prediction output by the model. Options are 'probs' (in [0,1]) or
        'logits' (in [-inf,inf]).

    Returns
    -------
    A scalar indication of uncertainty of the model on each instance in x.
    """

    preds = model_fn(x)

    if preds_type == "probs":
        if np.abs(1 - np.sum(preds, axis=-1)).mean() > 1e-6:
            raise ValueError("Probabilities across labels should sum to 1")
        probs = preds
    elif preds_type == "logits":
        probs = softmax(preds, axis=-1)
    else:
        raise NotImplementedError("Only prediction types 'probs' and 'logits' supported.")

    uncertainties = entropy(probs, axis=-1)
    return uncertainties[:, None]  # Detectors expect N x d  # type: ignore


class DriftUncertainty:
    """
    Test for a change in the number of instances falling into regions on which the
    model is uncertain. Performs a K-S test on prediction entropies.

    Parameters
    ----------
    x_ref : ArrayLike
        Data used as reference distribution. Should be disjoint from the data the
        model was trained on for accurate p-values.
    model : Callable
        Classification model outputting class probabilities (or logits)
    p_val : float, default 0.05
        p-value used for the significance of the test.
    x_ref_preprocessed : bool, default False
        Whether the given reference data `x_ref` has been preprocessed yet. If
        `x_ref_preprocessed=True`, only the test data `x` will be preprocessed at
        prediction time. If `x_ref_preprocessed=False`, the reference data will
        also be preprocessed.
    update_x_ref : Optional[UpdateStrategy], default None
        Reference data can optionally be updated using an UpdateStrategy class. Update
        using the last n instances seen by the detector with
        :py:class:`dataeval.detectors.LastSeenUpdateStrategy`
        or via reservoir sampling with
        :py:class:`dataeval.detectors.ReservoirSamplingUpdateStrategy`.
    preds_type : Literal["probs", "logits"], default "logits"
        Type of prediction output by the model. Options are 'probs' (in [0,1]) or
        'logits' (in [-inf,inf]).
    batch_size : int, default 32
        Batch size used to evaluate model. Only relevant when backend has been
        specified for batch prediction.
    preprocess_batch_fn : Optional[Callable], default None
        Optional batch preprocessing function. For example to convert a list of
        objects to a batch which can be processed by the model.
    device : Optional[str], default None
        Device type used. The default None tries to use the GPU and falls back on
        CPU if needed. Can be specified by passing either 'cuda', 'gpu' or 'cpu'.
    input_shape : Optional[tuple], default None
        Shape of input data.
    """

    def __init__(
        self,
        x_ref: ArrayLike,
        model: Callable,
        p_val: float = 0.05,
        x_ref_preprocessed: bool = False,
        update_x_ref: Optional[UpdateStrategy] = None,
        preds_type: Literal["probs", "logits"] = "probs",
        batch_size: int = 32,
        preprocess_batch_fn: Optional[Callable] = None,
        device: Optional[str] = None,
    ) -> None:
        def model_fn(x: np.ndarray) -> np.ndarray:
            return preprocess_drift(
                x,
                model,  # type: ignore
                batch_size=batch_size,
                preprocess_batch_fn=preprocess_batch_fn,
                device=get_device(device),
            )

        preprocess_fn = partial(
            classifier_uncertainty,
            model_fn=model_fn,
            preds_type=preds_type,
        )

        self._detector = DriftKS(
            x_ref=x_ref,
            p_val=p_val,
            x_ref_preprocessed=x_ref_preprocessed,
            update_x_ref=update_x_ref,
            preprocess_fn=preprocess_fn,  # type: ignore
        )

    def predict(self, x: ArrayLike) -> Dict[str, Union[int, float, np.ndarray]]:
        """
        Predict whether a batch of data has drifted from the reference data.

        Parameters
        ----------
        x : ArrayLike
            Batch of instances.

        Returns
        -------
        Dictionary containing the drift prediction, p-value, and threshold statistics.
        """
        return self._detector.predict(x)
