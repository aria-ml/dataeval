"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Literal, cast

import keras
import numpy as np
import tensorflow as tf
from numpy.typing import ArrayLike, NDArray

from dataeval._internal.interop import to_numpy
from dataeval._internal.models.tensorflow.gmm import GaussianMixtureModelParams, gmm_params
from dataeval._internal.models.tensorflow.trainer import trainer
from dataeval._internal.output import OutputMetadata, set_metadata


@dataclass(frozen=True)
class OODOutput(OutputMetadata):
    """
    Output class for predictions from :class:`OOD_AE`, :class:`OOD_AEGMM`, :class:`OOD_LLR`,
    :class:`OOD_VAE`, and :class:`OOD_VAEGMM` out-of-distribution detectors

    Attributes
    ----------
    is_ood : NDArray
        Array of images that are detected as out of distribution
    instance_score : NDArray
        Instance score of the evaluated dataset
    feature_score : NDArray | None
        Feature score, if available, of the evaluated dataset
    """

    is_ood: NDArray[np.bool_]
    instance_score: NDArray[np.float32]
    feature_score: NDArray[np.float32] | None


@dataclass(frozen=True)
class OODScoreOutput(OutputMetadata):
    """
    Output class for instance and feature scores from :class:`OOD_AE`, :class:`OOD_AEGMM`,
    :class:`OOD_LLR`, :class:`OOD_VAE`, and :class:`OOD_VAEGMM` out-of-distribution detectors

    Parameters
    ----------
    instance_score : NDArray
        Instance score of the evaluated dataset.
    feature_score : NDArray | None, default None
        Feature score, if available, of the evaluated dataset.
    """

    instance_score: NDArray[np.float32]
    feature_score: NDArray[np.float32] | None = None

    def get(self, ood_type: Literal["instance", "feature"]) -> NDArray:
        """
        Returns either the instance or feature score

        Parameters
        ----------
        ood_type : "instance" | "feature"

        Returns
        -------
        NDArray
            Either the instance or feature score based on input selection
        """
        return self.instance_score if ood_type == "instance" or self.feature_score is None else self.feature_score


class OODBase(ABC):
    def __init__(self, model: keras.Model) -> None:
        self.model = model

        self._ref_score: OODScoreOutput
        self._threshold_perc: float
        self._data_info: tuple[tuple, type] | None = None

        if not isinstance(model, keras.Model):
            raise TypeError("Model should be of type 'keras.Model'.")

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

    @abstractmethod
    def score(self, X: ArrayLike, batch_size: int = int(1e10)) -> OODScoreOutput:
        """
        Compute the out-of-distribution (OOD) scores for a given dataset.

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

    def _threshold_score(self, ood_type: Literal["feature", "instance"] = "instance") -> np.floating:
        return np.percentile(self._ref_score.get(ood_type), self._threshold_perc)

    def fit(
        self,
        x_ref: ArrayLike,
        threshold_perc: float = 100.0,
        loss_fn: Callable[..., tf.Tensor] | None = None,
        optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam,
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

    @set_metadata("dataeval.detectors")
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
    def __init__(self, model: keras.Model) -> None:
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
        loss_fn: Callable[..., tf.Tensor] | None = None,
        optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam,
        epochs: int = 20,
        batch_size: int = 64,
        verbose: bool = True,
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
        _, z, gamma = cast(tuple[tf.Tensor, tf.Tensor, tf.Tensor], self.model(x_ref))
        self.gmm_params = gmm_params(z, gamma)

        # Infer the threshold values
        self._ref_score = self.score(x_ref, batch_size)
        self._threshold_perc = threshold_perc
