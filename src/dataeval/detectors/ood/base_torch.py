"""
Adapted for Pytorch from

Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

from typing import Callable, Literal, cast

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

import dataeval.models.torch.trainer as torch_trainer
from dataeval.detectors.ood.base import OODBase, OODOutput, OODScoreOutput
from dataeval.interop import to_numpy
from dataeval.output import set_metadata
from dataeval.torch.models.gmm import GaussianMixtureModelParams, gmm_params


class OODBaseTorch(OODBase):
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

        self._ref_score: OODScoreOutput
        self._threshold_perc: float
        self._data_info: tuple[tuple, type] | None = None

        if not isinstance(model, torch.nn.Module):
            raise TypeError("Model should be of type 'torch.nn.Module'.")

    def _get_data_info(self, X: NDArray) -> tuple[tuple, type]:
        if not isinstance(X, np.ndarray):
            raise TypeError("Dataset should of type: `NDArray`.")
        return X.shape[1:], X.dtype.type

    def _validate(self, X: NDArray) -> None:
        check_data_info = self._get_data_info(X)
        if self._data_info is not None and check_data_info != self._data_info:
            raise RuntimeError(f"Expect data of type: {self._data_info[1]} and shape: {self._data_info[0]}. \
                               Provided data is type: {check_data_info[1]} and shape: {check_data_info[0]}.")

    def _validate_state(self, X: NDArray, additional_attrs: list[str] | None = None) -> None:
        attrs = ["_data_info", "_threshold_perc", "_ref_score"]
        attrs = attrs if additional_attrs is None else attrs + additional_attrs
        if not all(hasattr(self, attr) for attr in attrs) or any(getattr(self, attr) for attr in attrs) is None:
            raise RuntimeError("Metric needs to be `fit` before method call.")
        self._validate(X)

    def _threshold_score(self, ood_type: Literal["feature", "instance"] = "instance") -> np.floating:
        return np.percentile(self._ref_score.get(ood_type), self._threshold_perc)

    def fit(
        self,
        x_ref: ArrayLike,
        threshold_perc: float,
        loss_fn: Callable[..., torch.nn.Module],
        optimizer: torch.optim.Optimizer,
        epochs: int = 20,
        batch_size: int = 64,
        verbose: bool = True,
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
        verbose : bool, default True
            Whether to print training progress.
        """
        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # Train the model
        torch_trainer.trainer(
            model=self.model,
            loss_fn=loss_fn,
            x_train=to_numpy(x_ref),
            optimizer=optimizer,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )

        # Infer the threshold values
        self._ref_score = self.score(x_ref, batch_size)
        self._threshold_perc = threshold_perc

    @set_metadata()  # just bash through for now
    def predict(
        self,
        X: ArrayLike,
        batch_size: int = int(1e10),
        ood_type: Literal["feature", "instance"] = "instance",
    ) -> OODOutput:
        """
        Predict whether instances are out-of-distribution or not.

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


class OODGMMBase(OODBase):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model)
        self.gmm_params: GaussianMixtureModelParams

    def _validate_state(self, X: NDArray, additional_attrs: list[str] | None = None) -> None:
        if additional_attrs is None:
            additional_attrs = ["gmm_params"]
        super()._validate_state(X, additional_attrs)

    def fit(
        self,
        x_ref: ArrayLike,
        threshold_perc: float = 100.0,
        loss_fn: Callable[..., torch.Tensor] | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        epochs: int = 20,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> None:
        # Train the model
        torch_trainer.trainer(
            model=self.model,
            loss_fn=loss_fn,
            x_train=to_numpy(x_ref),
            optimizer=optimizer,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )

        # Calculate the GMM parameters
        _, z, gamma = cast(tuple[torch.Tensor, torch.Tensor, torch.Tensor], self.model(x_ref))
        self.gmm_params = gmm_params(z, gamma)

        # Infer the threshold values
        self._ref_score = self.score(x_ref, batch_size)
        self._threshold_perc = threshold_perc
