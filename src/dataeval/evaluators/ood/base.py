"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

__all__ = []

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar, cast

import numpy as np
import torch
from numpy.typing import NDArray

from dataeval.config import get_batch_size, get_device
from dataeval.protocols import Array, ArrayLike, DeviceLike, ProgressCallback
from dataeval.types import DictOutput, set_metadata
from dataeval.utils._array import as_numpy, to_numpy
from dataeval.utils._gmm import GaussianMixtureModelParams, gmm_params
from dataeval.utils._train import train


@dataclass(frozen=True)
class OODScoreOutput(DictOutput):
    """
    Output class for instance and feature scores from out-of-distribution detectors.

    Attributes
    ----------
    instance_score : NDArray
        Instance score of the evaluated dataset.
    feature_score : NDArray | None, default None
        Feature score, if available, of the evaluated dataset.
    """

    instance_score: NDArray[np.float32]
    feature_score: NDArray[np.float32] | None = None

    def get(self, ood_type: Literal["instance", "feature"]) -> NDArray[np.float32]:
        """
        Returns either the instance or feature score.

        Parameters
        ----------
        ood_type : "instance" | "feature"

        Returns
        -------
        NDArray
            Either the instance or feature score based on input selection
        """
        return self.instance_score if ood_type == "instance" or self.feature_score is None else self.feature_score


@dataclass(frozen=True)
class OODOutput(DictOutput):
    """
    Output class for predictions from out-of-distribution detectors.

    Attributes
    ----------
    is_ood : NDArray
        Array of images that are detected as :term:Out-of-Distribution (OOD)`
    instance_score : NDArray
        Instance score of the evaluated dataset
    feature_score : NDArray | None
        Feature score, if available, of the evaluated dataset
    """

    is_ood: NDArray[np.bool_]
    instance_score: NDArray[np.float32]
    feature_score: NDArray[np.float32] | None


TGMMParams = TypeVar("TGMMParams")


class OODGMMMixin(Generic[TGMMParams]):
    _gmm_params: TGMMParams


TModel = TypeVar("TModel", bound=Callable)
TLossFn = TypeVar("TLossFn", bound=Callable)
TOptimizer = TypeVar("TOptimizer")


class OODFitMixin(Generic[TLossFn, TOptimizer], ABC):
    @abstractmethod
    def fit(
        self,
        x_ref: ArrayLike,
        threshold_perc: float,
        loss_fn: TLossFn | None,
        optimizer: TOptimizer | None,
        epochs: int,
        batch_size: int | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """
        Train the model and infer the threshold value.

        Parameters
        ----------
        x_ref : ArrayLike
            Training data.
        threshold_perc : float, default 100.0
            Percentage of reference data that is normal.
        loss_fn : TLossFn
            Loss function used for training.
        optimizer : TOptimizer
            Optimizer used for training.
        epochs : int, default 20
            Number of training epochs.
        batch_size : int, default 64
            Batch size used for training.
        progress_callback : ProgressCallback, default None
            Callback to update training progress.
        """


class OODBaseMixin(Generic[TModel], ABC):
    _ref_score: OODScoreOutput
    _threshold_perc: float
    _data_info: tuple[tuple, type] | None = None

    def __init__(
        self,
        model: TModel,
    ) -> None:
        self.model = model

    def _get_data_info(self, X: NDArray) -> tuple[tuple, type]:
        if not isinstance(X, np.ndarray):
            raise TypeError("Dataset should of type: `NDArray`.")
        if np.min(X) < 0 or np.max(X) > 1:
            raise ValueError("Embeddings must be on the unit interval [0-1].")

        return X.shape[1:], X.dtype.type

    def _validate(self, X: NDArray) -> None:
        check_data_info = self._get_data_info(X)
        if self._data_info is not None and check_data_info != self._data_info:
            raise RuntimeError(
                f"Expect data of type: {self._data_info[1]} and shape: {self._data_info[0]}. \
                               Provided data is type: {check_data_info[1]} and shape: {check_data_info[0]}."
            )

    def _validate_state(self, X: NDArray) -> None:
        attrs = [k for c in self.__class__.mro()[:-1][::-1] if hasattr(c, "__annotations__") for k in c.__annotations__]
        if not all(hasattr(self, attr) for attr in attrs) or any(getattr(self, attr) for attr in attrs) is None:
            raise RuntimeError("Metric needs to be `fit` before method call.")
        self._validate(X)

    @abstractmethod
    def _score(self, X: NDArray[np.float32], batch_size: int | None = None) -> OODScoreOutput: ...

    @set_metadata
    def score(self, X: ArrayLike, batch_size: int | None = None) -> OODScoreOutput:
        """
        Compute the :term:`out of distribution<Out-of-distribution (OOD)>` scores for a given dataset.

        Parameters
        ----------
        X : ArrayLike
            Input data to score.
        batch_size : int | None, default None
            Number of instances to process in each batch.
            Use a smaller batch size if your dataset is large or if you encounter memory issues.

        Raises
        ------
        ValueError
            X input data must be unit interval [0-1].

        Returns
        -------
        OODScoreOutput
            An object containing the instance-level and feature-level OOD scores.
        """
        self._validate(X := as_numpy(X).astype(np.float32))
        return self._score(X, get_batch_size(batch_size))

    def _threshold_score(self, ood_type: Literal["feature", "instance"] = "instance") -> np.floating:
        return np.percentile(self._ref_score.get(ood_type), self._threshold_perc)

    @set_metadata
    def predict(
        self,
        X: ArrayLike,
        batch_size: int | None = None,
        ood_type: Literal["feature", "instance"] = "instance",
    ) -> OODOutput:
        """
        Predict whether instances are :term:`out of distribution<Out-of-distribution (OOD)>` or not.

        Parameters
        ----------
        X : ArrayLike
            Input data for out-of-distribution prediction.
        batch_size : int, default 1e10
            Number of instances to process in each batch.
        ood_type : "feature" | "instance", default "instance"
            Predict out-of-distribution at the 'feature' or 'instance' level.

        Raises
        ------
        ValueError
            X input data must be unit interval [0-1].

        Returns
        -------
        Dictionary containing the outlier predictions for the selected level,
        and the OOD scores for the data including both 'instance' and 'feature' (if present) level scores.
        """
        self._validate_state(X := to_numpy(X).astype(np.float32))
        # compute outlier scores
        score = self.score(X, batch_size=batch_size)
        ood_pred = score.get(ood_type) > self._threshold_score(ood_type)
        return OODOutput(is_ood=ood_pred, **score.data())


class OODBase(OODBaseMixin[torch.nn.Module], OODFitMixin[Callable[..., torch.Tensor], torch.optim.Optimizer]):
    def __init__(self, model: torch.nn.Module, device: DeviceLike | None = None) -> None:
        self.device: torch.device = get_device(device)
        super().__init__(model)

    def fit(
        self,
        x_ref: ArrayLike,
        threshold_perc: float,
        loss_fn: Callable[..., torch.Tensor] | None,
        optimizer: torch.optim.Optimizer | None,
        epochs: int,
        batch_size: int | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """
        Train the model and infer the threshold value.

        Parameters
        ----------
        x_ref : ArrayLike
            Training data.
        threshold_perc : float, default 100.0
            Percentage of reference data that is normal.
        loss_fn : Callable | None, default None
            Loss function used for training.
        optimizer : Optimizer, default keras.optimizers.Adam
            Optimizer used for training.
        epochs : int, default 20
            Number of training epochs.
        batch_size : int, default 64
            Batch size used for training.
        progress_callback : ProgressCallback, default None
            Callback to update training progress.
        """

        # Train the model
        train(
            model=self.model,
            x_train=to_numpy(x_ref),
            y_train=None,
            loss_fn=loss_fn,
            optimizer=optimizer,
            preprocess_fn=None,
            epochs=epochs,
            batch_size=batch_size,
            device=self.device,
            progress_callback=progress_callback,
        )

        # Infer the threshold values
        self._ref_score = self.score(x_ref, batch_size)
        self._threshold_perc = threshold_perc


class OODBaseGMM(OODBase, OODGMMMixin[GaussianMixtureModelParams]):
    def fit(
        self,
        x_ref: ArrayLike,
        threshold_perc: float,
        loss_fn: Callable[..., torch.Tensor] | None,
        optimizer: torch.optim.Optimizer | None,
        epochs: int,
        batch_size: int | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        super().fit(x_ref, threshold_perc, loss_fn, optimizer, epochs, get_batch_size(batch_size), progress_callback)

        # Calculate the GMM parameters
        _, z, gamma = cast(tuple[torch.Tensor, torch.Tensor, torch.Tensor], self.model(x_ref))
        self._gmm_params = gmm_params(z, gamma)


class EmbeddingBasedOODBase(OODBaseMixin[Callable[[Any], Any]], ABC):
    """
    Base class for embedding-based OOD detection methods.

    These methods work directly on embedding representations,
    using distance metrics or density estimation in embedding space.
    Inherits from OODBaseMixin to get automatic thresholding.
    """

    def __init__(self) -> None:
        """Initialize embedding-based OOD detector."""
        # Pass a dummy callable as model since we don't use it
        super().__init__(lambda x: x)

    def _get_data_info(self, X: NDArray) -> tuple[tuple, type]:
        """Override to skip [0-1] validation for embeddings."""
        if not isinstance(X, np.ndarray):
            raise TypeError("Dataset should of type: `NDArray`.")
        # Skip the [0-1] range check for embeddings
        return X.shape[1:], X.dtype.type

    @abstractmethod
    def fit_embeddings(self, embeddings: Array, threshold_perc: float = 95.0) -> None:
        """
        Fit using reference embeddings.

        Parameters
        ----------
        embeddings : Array
            Reference (in-distribution) embeddings
        threshold_perc : float, default 95.0
            Percentage of reference data considered normal
        """
