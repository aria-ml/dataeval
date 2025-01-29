from __future__ import annotations

from dataeval.detectors.ood.output import OODOutput, OODScoreOutput

__all__ = []

from abc import ABC, abstractmethod
from typing import Callable, Generic, Literal, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dataeval._interop import to_numpy
from dataeval._output import set_metadata

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
        batch_size: int,
        verbose: bool,
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
        verbose : bool, default True
            Whether to print training progress.
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
    def _score(self, X: ArrayLike, batch_size: int = int(1e10)) -> OODScoreOutput: ...

    @set_metadata
    def score(self, X: ArrayLike, batch_size: int = int(1e10)) -> OODScoreOutput:
        """
        Compute the :term:`out of distribution<Out-of-distribution (OOD)>` scores for a given dataset.

        Parameters
        ----------
        X : ArrayLike
            Input data to score.
        batch_size : int, default 1e10
            Number of instances to process in each batch.
            Use a smaller batch size if your dataset is large or if you encounter memory issues.

        Returns
        -------
        OODScoreOutput
            An object containing the instance-level and feature-level OOD scores.
        """
        return self._score(X, batch_size)

    def _threshold_score(self, ood_type: Literal["feature", "instance"] = "instance") -> np.floating:
        return np.percentile(self._ref_score.get(ood_type), self._threshold_perc)

    @set_metadata
    def predict(
        self,
        X: ArrayLike,
        batch_size: int = int(1e10),
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

        Returns
        -------
        Dictionary containing the outlier predictions for the selected level,
        and the OOD scores for the data including both 'instance' and 'feature' (if present) level scores.
        """
        self._validate_state(X := to_numpy(X))
        # compute outlier scores
        score = self.score(X, batch_size=batch_size)
        ood_pred = score.get(ood_type) > self._threshold_score(ood_type)
        return OODOutput(is_ood=ood_pred, **score.dict())
