"""
Source code derived from Alibi-Detect 0.11.4.

https://github.com/SeldonIO/alibi-detect/tree/v0.11.4.

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

__all__ = []

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from dataeval.protocols import ArrayLike
from dataeval.types import DictOutput, set_metadata
from dataeval.utils.arrays import as_numpy


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
        Return either the instance or feature score.

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
        Array of images that are detected as :term:`Out-of-Distribution (OOD)`
    instance_score : NDArray
        Instance score of the evaluated dataset
    feature_score : NDArray | None
        Feature score, if available, of the evaluated dataset
    """

    is_ood: NDArray[np.bool_]
    instance_score: NDArray[np.float32]
    feature_score: NDArray[np.float32] | None


class BaseOOD:
    """Lightweight base for all OOD detectors.

    Provides the shared fit/score/predict lifecycle:

    - Data validation against fitted shape/dtype
    - Percentile-based thresholding from reference scores
    - Unified :meth:`score` and :meth:`predict` API

    Subclasses must implement:

    - :meth:`fit` — store ``_ref_score`` and ``_threshold_perc``
    - :meth:`_score` — compute OOD scores for preprocessed input

    Subclasses may override:

    - :meth:`_preprocess` — custom input conversion (default: ``as_numpy`` + float32)
    - :meth:`_get_data_info` — custom input validation (default: shape + dtype check)
    """

    _ref_score: OODScoreOutput
    _threshold_perc: float

    def __init__(self) -> None:
        self._data_info: tuple[tuple, type] | None = None

    def _get_data_info(self, x: NDArray) -> tuple[tuple, type]:
        """Extract shape and dtype info from *x*.

        Override to add extra validation (e.g. value-range checks).

        Raises
        ------
        TypeError
            If *x* is not an :class:`~numpy.ndarray`.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("Dataset should be of type: `NDArray`.")
        return (x.shape[1:], x.dtype.type)

    def _validate(self, x: NDArray) -> None:
        """Validate that *x* matches the shape/dtype seen at fit time."""
        check_data_info = self._get_data_info(x)
        if self._data_info is not None and check_data_info != self._data_info:
            raise RuntimeError(
                f"Expect data of type: {self._data_info[1]} and shape: {self._data_info[0]}. "
                f"Provided data is type: {check_data_info[1]} and shape: {check_data_info[0]}.",
            )

    def _validate_state(self) -> None:
        """Ensure the detector has been fitted."""
        if not hasattr(self, "_ref_score") or not hasattr(self, "_threshold_perc"):
            raise RuntimeError("Detector needs to be `fit` before calling predict or score.")

    def _threshold_score(self, ood_type: Literal["feature", "instance"] = "instance") -> np.floating:
        """Get the threshold score for a given OOD type."""
        return np.percentile(self._ref_score.get(ood_type), self._threshold_perc)

    def _preprocess(self, x: ArrayLike) -> NDArray[np.float32]:
        """Convert *x* to a float32 :class:`~numpy.ndarray`.

        Override to add extra preprocessing (e.g. flattening).
        """
        return as_numpy(x).astype(np.float32)

    def _score(self, x: NDArray[np.float32], batch_size: int = int(1e10)) -> OODScoreOutput:
        """Compute OOD scores for preprocessed input. Must be implemented by subclasses."""
        raise NotImplementedError

    @set_metadata
    def score(self, x: ArrayLike, batch_size: int = int(1e10)) -> OODScoreOutput:
        """
        Compute :term:`out of distribution<Out-of-distribution (OOD)>` scores for a given dataset.

        Parameters
        ----------
        x : ArrayLike
            Input data to score.
        batch_size : int, default 1e10
            Number of instances to process per batch (only used by some detectors).

        Returns
        -------
        OODScoreOutput
            Instance-level (and optionally feature-level) OOD scores.
            Higher scores indicate samples more likely to be OOD.
        """
        x_np = self._preprocess(x)
        self._validate(x_np)
        return self._score(x_np, batch_size)

    @set_metadata
    def predict(
        self,
        x: ArrayLike,
        batch_size: int = int(1e10),
        ood_type: Literal["feature", "instance"] = "instance",
    ) -> OODOutput:
        """
        Predict whether instances are :term:`out of distribution<Out-of-distribution (OOD)>`.

        Parameters
        ----------
        x : ArrayLike
            Input data for OOD prediction.
        batch_size : int, default 1e10
            Number of instances to process per batch (only used by some detectors).
        ood_type : "feature" | "instance", default "instance"
            Predict OOD at the ``"feature"`` or ``"instance"`` level.

        Returns
        -------
        OODOutput
            Predictions including ``is_ood`` boolean array and OOD scores.
        """
        self._validate_state()
        scores = self.score(x, batch_size=batch_size)
        ood_pred = scores.get(ood_type) > self._threshold_score(ood_type)
        return OODOutput(is_ood=ood_pred, **scores.data())
