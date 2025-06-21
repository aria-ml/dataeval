"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

__all__ = []

import math
from abc import abstractmethod
from collections.abc import Callable
from functools import wraps
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from dataeval.data import Embeddings
from dataeval.outputs import DriftOutput
from dataeval.outputs._base import set_metadata
from dataeval.typing import Array
from dataeval.utils._array import as_numpy, flatten

R = TypeVar("R")


@runtime_checkable
class UpdateStrategy(Protocol):
    """
    Protocol for reference dataset update strategy for drift detectors
    """

    def __call__(self, x_ref: NDArray[np.float32], x_new: NDArray[np.float32], count: int) -> NDArray[np.float32]: ...


def update_strategy(fn: Callable[..., R]) -> Callable[..., R]:
    """Decorator to update x_ref with x using selected update methodology"""

    @wraps(fn)
    def _(self: BaseDrift, data: Embeddings | Array, *args: tuple[Any, ...], **kwargs: dict[str, Any]) -> R:
        output = fn(self, data, *args, **kwargs)

        # update reference dataset
        if self.update_strategy is not None:
            self._x_ref = self.update_strategy(self.x_ref, self._encode(data), self.n)
            self.n += len(data)

        return output

    return _


class BaseDrift:
    p_val: float
    update_strategy: UpdateStrategy | None
    correction: Literal["bonferroni", "fdr"]
    n: int

    def __init__(
        self,
        data: Embeddings | Array,
        p_val: float = 0.05,
        update_strategy: UpdateStrategy | None = None,
        correction: Literal["bonferroni", "fdr"] = "bonferroni",
    ) -> None:
        # Type checking
        if update_strategy is not None and not isinstance(update_strategy, UpdateStrategy):
            raise ValueError("`update_strategy` is not a valid UpdateStrategy class.")
        if correction not in ["bonferroni", "fdr"]:
            raise ValueError("`correction` must be `bonferroni` or `fdr`.")

        self._data = data
        self.p_val = p_val
        self.update_strategy = update_strategy
        self.correction = correction
        self.n = len(data)

        self._x_ref: NDArray[np.float32] | None = None

    @property
    def x_ref(self) -> NDArray[np.float32]:
        """
        Retrieve the reference data of the drift detector.

        Returns
        -------
        NDArray[np.float32]
            The reference data as a 32-bit floating point numpy array.
        """
        if self._x_ref is None:
            self._x_ref = self._encode(self._data)
        return self._x_ref

    def _encode(self, data: Embeddings | Array) -> NDArray[np.float32]:
        array = (
            data.to_numpy().astype(np.float32)
            if isinstance(data, Embeddings)
            else self._data.new(data).to_numpy().astype(np.float32)
            if isinstance(self._data, Embeddings)
            else as_numpy(data).astype(np.float32)
        )
        return flatten(array)


class BaseDriftUnivariate(BaseDrift):
    def __init__(
        self,
        data: Embeddings | Array,
        p_val: float = 0.05,
        update_strategy: UpdateStrategy | None = None,
        correction: Literal["bonferroni", "fdr"] = "bonferroni",
        n_features: int | None = None,
    ) -> None:
        super().__init__(data, p_val, update_strategy, correction)

        self._n_features = n_features

    @property
    def n_features(self) -> int:
        """
        Get the number of features in the reference data.

        If the number of features is not provided during initialization, it will be inferred
        from the reference data (``x_ref``).

        Returns
        -------
        int
            Number of features in the reference data.
        """
        # lazy process n_features as needed
        if self._n_features is None:
            self._n_features = int(math.prod(self._data[0].shape))

        return self._n_features

    def score(self, data: Embeddings | Array) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Calculates p-values and test statistics per feature.

        Parameters
        ----------
        data : Embeddings or Array
            Batch of instances to score.

        Returns
        -------
        tuple[NDArray, NDArray]
            Feature level p-values and test statistics
        """
        x_np = self._encode(data)
        p_val = np.zeros(self.n_features, dtype=np.float32)
        dist = np.zeros_like(p_val)
        for f in range(self.n_features):
            dist[f], p_val[f] = self._score_fn(self.x_ref[:, f], x_np[:, f])
        return p_val, dist

    @abstractmethod
    def _score_fn(self, x: NDArray[np.float32], y: NDArray[np.float32]) -> tuple[np.float32, np.float32]: ...

    def _apply_correction(self, p_vals: NDArray[np.float32]) -> tuple[bool, float]:
        """
        Apply the specified correction method (Bonferroni or FDR) to the p-values.

        If the correction method is Bonferroni, the threshold for detecting :term:`drift<Drift>`
        is divided by the number of features. For FDR, the correction is applied
        using the Benjamini-Hochberg procedure.

        Parameters
        ----------
        p_vals : NDArray
            Array of p-values from the univariate tests for each feature.

        Returns
        -------
        tuple[bool, float]
            A tuple containing a boolean indicating if drift was detected and the
            threshold after correction.
        """
        if self.correction == "bonferroni":
            threshold = self.p_val / self.n_features
            drift_pred = bool((p_vals < threshold).any())
            return drift_pred, threshold
        if self.correction == "fdr":
            n = p_vals.shape[0]
            i = np.arange(n) + np.int_(1)
            p_sorted = np.sort(p_vals)
            q_threshold = self.p_val * i / n
            below_threshold = p_sorted < q_threshold
            try:
                idx_threshold = int(np.where(below_threshold)[0].max())
            except ValueError:  # sorted p-values not below thresholds
                return bool(below_threshold.any()), q_threshold.min()
            return bool(below_threshold.any()), q_threshold[idx_threshold]
        raise ValueError("`correction` needs to be either `bonferroni` or `fdr`.")

    @set_metadata
    @update_strategy
    def predict(self, data: Embeddings | Array) -> DriftOutput:
        """
        Predict whether a batch of data has drifted from the reference data and update
        reference data using specified update strategy.

        Parameters
        ----------
        data : Embeddings or Array
            Batch of instances to predict drift on.

        Returns
        -------
        DriftOutput
            Dictionary containing the :term:`drift<Drift>` prediction and optionally the feature level
            p-values, threshold after multivariate correction if needed and test :term:`statistics<Statistics>`.
        """
        # compute drift scores
        p_vals, dist = self.score(data)

        feature_drift = (p_vals < self.p_val).astype(np.bool_)
        drift_pred, threshold = self._apply_correction(p_vals)
        return DriftOutput(
            drift_pred, threshold, float(np.mean(p_vals)), float(np.mean(dist)), feature_drift, self.p_val, p_vals, dist
        )
