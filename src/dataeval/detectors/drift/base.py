"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

__all__ = ["DriftOutput"]

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Literal, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dataeval.interop import as_numpy, to_numpy
from dataeval.output import OutputMetadata, set_metadata

R = TypeVar("R")


class UpdateStrategy(ABC):
    """
    Updates reference dataset for drift detector

    Parameters
    ----------
    n : int
        Update with last n instances seen by the detector.
    """

    def __init__(self, n: int) -> None:
        self.n = n

    @abstractmethod
    def __call__(self, x_ref: NDArray[Any], x: NDArray[Any], count: int) -> NDArray[Any]:
        """Abstract implementation of update strategy"""


@dataclass(frozen=True)
class DriftBaseOutput(OutputMetadata):
    """
    Base output class for Drift detector classes

    Attributes
    ----------
    is_drift : bool
        Drift prediction for the images
    threshold : float
        Threshold after multivariate correction if needed
    """

    is_drift: bool
    threshold: float
    p_val: float
    distance: float


@dataclass(frozen=True)
class DriftOutput(DriftBaseOutput):
    """
    Output class for :class:`DriftCVM`, :class:`DriftKS`, and :class:`DriftUncertainty` drift detectors

    Attributes
    ----------
    is_drift : bool
        :term:`Drift` prediction for the images
    threshold : float
        Threshold after multivariate correction if needed
    feature_drift : NDArray
        Feature-level array of images detected to have drifted
    feature_threshold : float
        Feature-level threshold to determine drift
    p_vals : NDArray
        Feature-level p-values
    distances : NDArray
        Feature-level distances
    """

    # is_drift: bool
    # threshold: float
    # p_val: float
    # distance: float
    feature_drift: NDArray[np.bool_]
    feature_threshold: float
    p_vals: NDArray[np.float32]
    distances: NDArray[np.float32]


def update_x_ref(fn: Callable[..., R]) -> Callable[..., R]:
    """Decorator to update x_ref with x using selected update methodology"""

    @wraps(fn)
    def _(self, x, *args, **kwargs) -> R:
        output = fn(self, x, *args, **kwargs)

        # update reference dataset
        if self.update_x_ref is not None:
            self._x_ref = self.update_x_ref(self.x_ref, x, self.n)

        # used for reservoir sampling
        self.n += len(x)
        return output

    return _


def preprocess_x(fn: Callable[..., R]) -> Callable[..., R]:
    """Decorator to run preprocess_fn on x before calling wrapped function"""

    @wraps(fn)
    def _(self, x, *args, **kwargs) -> R:
        if self._x_refcount == 0:
            self._x = self._preprocess(x)
        self._x_refcount += 1
        output = fn(self, self._x, *args, **kwargs)
        self._x_refcount -= 1
        if self._x_refcount == 0:
            del self._x
        return output

    return _


class BaseDrift:
    """
    A generic :term:`drift<Drift>` detection component for preprocessing data and applying statistical correction.

    This class handles common tasks related to drift detection, such as preprocessing
    the reference data (`x_ref`), performing statistical correction (e.g., Bonferroni, FDR),
    and updating the reference data if needed.

    Parameters
    ----------
    x_ref : ArrayLike
        The reference dataset used for drift detection. This is the baseline data against
        which new data points will be compared.
    p_val : float, optional
        The significance level for detecting drift, by default 0.05.
    x_ref_preprocessed : bool, optional
        Flag indicating whether the reference data has already been preprocessed, by default False.
    update_x_ref : UpdateStrategy, optional
        A strategy object specifying how the reference data should be updated when drift is detected,
        by default None.
    preprocess_fn : Callable[[ArrayLike], ArrayLike], optional
        A function to preprocess the data before drift detection, by default None.
    correction : {'bonferroni', 'fdr'}, optional
        Statistical correction method applied to p-values, by default "bonferroni".

    Attributes
    ----------
    _x_ref : ArrayLike
        The reference dataset that is either raw or preprocessed.
    p_val : float
        The significance level for drift detection.
    update_x_ref : UpdateStrategy or None
        The strategy for updating the reference data if applicable.
    preprocess_fn : Callable or None
        Function used for preprocessing input data before drift detection.
    correction : str
        Statistical correction method applied to p-values.
    n : int
        The number of samples in the reference dataset (`x_ref`).
    x_ref_preprocessed : bool
        A flag that indicates whether the reference dataset has been preprocessed.
    _x_refcount : int
        Counter for how many times the reference data has been accessed after preprocessing.

    Methods
    -------
    x_ref:
        Property that returns the reference dataset, and applies preprocessing if not already done.
    _preprocess(x):
        Preprocesses the given data using the specified `preprocess_fn` if provided.
    """

    def __init__(
        self,
        x_ref: ArrayLike,
        p_val: float = 0.05,
        x_ref_preprocessed: bool = False,
        update_x_ref: UpdateStrategy | None = None,
        preprocess_fn: Callable[..., ArrayLike] | None = None,
        correction: Literal["bonferroni", "fdr"] = "bonferroni",
    ) -> None:
        # Type checking
        if preprocess_fn is not None and not isinstance(preprocess_fn, Callable):
            raise ValueError("`preprocess_fn` is not a valid Callable.")
        if update_x_ref is not None and not isinstance(update_x_ref, UpdateStrategy):
            raise ValueError("`update_x_ref` is not a valid ReferenceUpdate class.")
        if correction not in ["bonferroni", "fdr"]:
            raise ValueError("`correction` must be `bonferroni` or `fdr`.")

        self._x_ref = to_numpy(x_ref)
        self.x_ref_preprocessed: bool = x_ref_preprocessed

        # Other attributes
        self.p_val = p_val
        self.update_x_ref = update_x_ref
        self.preprocess_fn = preprocess_fn
        self.correction = correction
        self.n: int = len(self._x_ref)

        # Ref counter for preprocessed x
        self._x_refcount = 0

    @property
    def x_ref(self) -> NDArray[Any]:
        """
        Retrieve the reference data, applying preprocessing if not already done.

        Returns
        -------
        NDArray
            The reference dataset (`x_ref`), preprocessed if needed.
        """
        if not self.x_ref_preprocessed:
            self.x_ref_preprocessed = True
            if self.preprocess_fn is not None:
                self._x_ref = as_numpy(self.preprocess_fn(self._x_ref))

        return self._x_ref

    def _preprocess(self, x: ArrayLike) -> ArrayLike:
        """
        Preprocess the given data before computing the :term:`drift<Drift>` scores.

        Parameters
        ----------
        x : ArrayLike
            The input data to preprocess.

        Returns
        -------
        ArrayLike
            The preprocessed input data.
        """
        if self.preprocess_fn is not None:
            x = self.preprocess_fn(x)
        return x


class BaseDriftUnivariate(BaseDrift):
    """
    Base class for :term:`drift<Drift>` detection methods using univariate statistical tests.

    This class inherits from `BaseDrift` and serves as a generic component for detecting
    distribution drift in univariate features. If the number of features `n_features` is greater
    than 1, a multivariate correction method (e.g., Bonferroni or FDR) is applied to control
    the :term:`false positive rate<False Positive Rate (FP)>`, ensuring it does not exceed the specified
    :term:`p-value<P-Value>`.

    Parameters
    ----------
    x_ref : ArrayLike
        Reference data used as the baseline to compare against when detecting drift.
    p_val : float, default 0.05
        Significance level used for detecting drift.
    x_ref_preprocessed : bool, default False
        Indicates whether the reference data has been preprocessed.
    update_x_ref : UpdateStrategy | None, default None
        Strategy for updating the reference data when drift is detected.
    preprocess_fn : Callable[ArrayLike] | None, default None
        Function used to preprocess input data before detecting drift.
    correction : 'bonferroni' | 'fdr', default 'bonferroni'
        Multivariate correction method applied to p-values.
    n_features : int | None, default None
        Number of features used in the univariate drift tests. If not provided, it will
        be inferred from the data.

    Attributes
    ----------
    p_val : float
        The significance level for drift detection.
    correction : str
        The method for controlling the :term:`False Discovery Rate (FDR)` or applying a Bonferroni correction.
    update_x_ref : UpdateStrategy | None
        Strategy for updating the reference data if applicable.
    preprocess_fn : Callable | None
        Function used for preprocessing input data before drift detection.
    """

    def __init__(
        self,
        x_ref: ArrayLike,
        p_val: float = 0.05,
        x_ref_preprocessed: bool = False,
        update_x_ref: UpdateStrategy | None = None,
        preprocess_fn: Callable[[ArrayLike], ArrayLike] | None = None,
        correction: Literal["bonferroni", "fdr"] = "bonferroni",
        n_features: int | None = None,
    ) -> None:
        super().__init__(
            x_ref,
            p_val,
            x_ref_preprocessed,
            update_x_ref,
            preprocess_fn,
            correction,
        )

        self._n_features = n_features

    @property
    def n_features(self) -> int:
        """
        Get the number of features in the reference data.

        If the number of features is not provided during initialization, it will be inferred
        from the reference data (``x_ref``). If a preprocessing function is provided, the number
        of features will be inferred after applying the preprocessing function.

        Returns
        -------
        int
            Number of features in the reference data.
        """
        # lazy process n_features as needed
        if not isinstance(self._n_features, int):
            # compute number of features for the univariate tests
            if not isinstance(self.preprocess_fn, Callable) or self.x_ref_preprocessed:
                # infer features from preprocessed reference data
                self._n_features = self.x_ref.reshape(self.x_ref.shape[0], -1).shape[-1]
            else:
                # infer number of features after applying preprocessing step
                x = as_numpy(self.preprocess_fn(self._x_ref[0:1]))  # type: ignore
                self._n_features = x.reshape(x.shape[0], -1).shape[-1]

        return self._n_features

    @preprocess_x
    @abstractmethod
    def score(self, x: ArrayLike) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Abstract method to calculate feature scores after preprocessing.

        Parameters
        ----------
        x : ArrayLike
            The batch of data to calculate univariate :term:`drift<Drift>` scores for each feature.

        Returns
        -------
        tuple[NDArray, NDArray]
            A tuple containing p-values and distance :term:`statistics<Statistics>` for each feature.
        """

    def _apply_correction(self, p_vals: NDArray) -> tuple[bool, float]:
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
        elif self.correction == "fdr":
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
        else:
            raise ValueError("`correction` needs to be either `bonferroni` or `fdr`.")

    @set_metadata()
    @preprocess_x
    @update_x_ref
    def predict(
        self,
        x: ArrayLike,
    ) -> DriftOutput:
        """
        Predict whether a batch of data has drifted from the reference data and update
        reference data using specified update strategy.

        Parameters
        ----------
        x : ArrayLike
            Batch of instances.

        Returns
        -------
        DriftOutput
            Dictionary containing the :term:`drift<Drift>` prediction and optionally the feature level
            p-values, threshold after multivariate correction if needed and test :term:`statistics<Statistics>`.
        """
        # compute drift scores
        p_vals, dist = self.score(x)

        feature_drift = (p_vals < self.p_val).astype(np.bool_)
        drift_pred, threshold = self._apply_correction(p_vals)
        return DriftOutput(
            drift_pred, threshold, float(np.mean(p_vals)), float(np.mean(dist)), feature_drift, self.p_val, p_vals, dist
        )
