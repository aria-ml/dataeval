"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Literal, NamedTuple, Optional, Tuple, cast

import keras
import numpy as np
import tensorflow as tf

from dataeval._internal.interop import ArrayLike, to_numpy
from dataeval._internal.models.tensorflow.gmm import GaussianMixtureModelParams, gmm_params
from dataeval._internal.models.tensorflow.trainer import trainer


class OODScore(NamedTuple):
    """
    NamedTuple containing the instance and (optionally) feature score.

    Parameters
    ----------
    instance_score : np.ndarray
        Instance score of the evaluated dataset.
    feature_score : Optional[np.ndarray], default None
        Feature score, if available, of the evaluated dataset.
    """

    instance_score: np.ndarray
    feature_score: Optional[np.ndarray] = None

    def get(self, ood_type: Literal["instance", "feature"]) -> np.ndarray:
        return self.instance_score if ood_type == "instance" or self.feature_score is None else self.feature_score


class OODBase(ABC):
    def __init__(self, model: keras.Model) -> None:
        self.model = model

        self._ref_score: OODScore
        self._threshold_perc: float
        self._data_info: Optional[Tuple[tuple, type]] = None

        if not isinstance(model, keras.Model):
            raise TypeError("Model should be of type 'keras.Model'.")

    def _get_data_info(self, X: np.ndarray) -> Tuple[tuple, type]:
        if not isinstance(X, np.ndarray):
            raise TypeError("Dataset should of type: `np.ndarray`.")
        return X.shape[1:], X.dtype.type

    def _validate(self, X: np.ndarray) -> None:
        check_data_info = self._get_data_info(X)
        if self._data_info is not None and check_data_info != self._data_info:
            raise RuntimeError(f"Expect data of type: {self._data_info[1]} and shape: {self._data_info[0]}. \
                               Provided data is type: {check_data_info[1]} and shape: {check_data_info[0]}.")

    def _validate_state(self, X: np.ndarray, additional_attrs: Optional[List[str]] = None) -> None:
        attrs = ["_data_info", "_threshold_perc", "_ref_score"]
        attrs = attrs if additional_attrs is None else attrs + additional_attrs
        if not all(hasattr(self, attr) for attr in attrs) or any(getattr(self, attr) for attr in attrs) is None:
            raise RuntimeError("Metric needs to be `fit` before method call.")
        self._validate(X)

    @abstractmethod
    def score(self, X: ArrayLike, batch_size: int = int(1e10)) -> OODScore:
        """
        Compute instance and (optionally) feature level outlier scores.

        Parameters
        ----------
        X : ArrayLike
            Batch of instances.
        batch_size : int, default int(1e10)
            Batch size used when making predictions with the autoencoder.

        Returns
        -------
        Instance and feature level outlier scores.
        """

    def _threshold_score(self, ood_type: Literal["feature", "instance"] = "instance") -> np.floating:
        return np.percentile(self._ref_score.get(ood_type), self._threshold_perc)

    def fit(
        self,
        x_ref: ArrayLike,
        threshold_perc: float,
        loss_fn: Callable,
        optimizer: keras.optimizers.Optimizer,
        epochs: int,
        batch_size: int,
        verbose: bool,
    ) -> None:
        """
        Train the model and infer the threshold value.

        Parameters
        ----------
        x_ref: : ArrayLike
            Training batch.
        threshold_perc : float
            Percentage of reference data that is normal.
        loss_fn : Callable
            Loss function used for training.
        optimizer : keras.optimizers.Optimizer
            Optimizer used for training.
        epochs : int
            Number of training epochs.
        batch_size : int
            Batch size used for training.
        verbose : bool
            Whether to print training progress.
        """
        # Train the model
        trainer(
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

    def predict(
        self,
        X: ArrayLike,
        batch_size: int = int(1e10),
        ood_type: Literal["feature", "instance"] = "instance",
    ) -> Dict[str, np.ndarray]:
        """
        Predict whether instances are out-of-distribution or not.

        Parameters
        ----------
        X : ArrayLike
            Batch of instances.
        batch_size : int, default int(1e10)
            Batch size used when making predictions with the autoencoder.
        ood_type : Literal["feature", "instance"], default "instance"
            Predict out-of-distribution at the 'feature' or 'instance' level.

        Returns
        -------
        Dictionary containing the outlier predictions and both feature and instance level outlier scores.
        """
        self._validate_state(X := to_numpy(X))
        # compute outlier scores
        score = self.score(X, batch_size=batch_size)
        ood_pred = (score.get(ood_type) > self._threshold_score(ood_type)).astype(int)
        return {**{"is_ood": ood_pred}, **score._asdict()}


class OODGMMBase(OODBase):
    def __init__(self, model: keras.Model) -> None:
        super().__init__(model)
        self.gmm_params: GaussianMixtureModelParams

    def _validate_state(self, X: np.ndarray, additional_attrs: Optional[List[str]] = None) -> None:
        if additional_attrs is None:
            additional_attrs = ["gmm_params"]
        super()._validate_state(X, additional_attrs)

    def fit(
        self,
        x_ref: ArrayLike,
        threshold_perc: float,
        loss_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        optimizer: keras.optimizers.Optimizer,
        epochs: int,
        batch_size: int,
        verbose: bool,
    ) -> None:
        # Train the model
        trainer(
            model=self.model,
            loss_fn=loss_fn,
            x_train=to_numpy(x_ref),
            optimizer=optimizer,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )

        # Calculate the GMM parameters
        _, z, gamma = cast(Tuple[tf.Tensor, tf.Tensor, tf.Tensor], self.model(x_ref))
        self.gmm_params = gmm_params(z, gamma)

        # Infer the threshold values
        self._ref_score = self.score(x_ref, batch_size)
        self._threshold_perc = threshold_perc
