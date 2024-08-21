"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from abc import ABC, abstractmethod
from functools import wraps
from typing import Callable, Dict, Literal, Optional, Tuple, Union

import numpy as np

from dataeval._internal.interop import ArrayLike, to_numpy


def update_x_ref(fn):
    @wraps(fn)
    def _(self, x, *args, **kwargs):
        output = fn(self, x, *args, **kwargs)

        # update reference dataset
        if self.update_x_ref is not None:
            self._x_ref = self.update_x_ref(self.x_ref, x, self.n)

        # used for reservoir sampling
        self.n += len(x)
        return output

    return _


def preprocess_x(fn):
    @wraps(fn)
    def _(self, x, *args, **kwargs):
        if self._x_refcount == 0:
            self._x = self._preprocess(x)
        self._x_refcount += 1
        output = fn(self, self._x, *args, **kwargs)
        self._x_refcount -= 1
        if self._x_refcount == 0:
            del self._x
        return output

    return _


class UpdateStrategy(ABC):
    def __init__(self, n: int):
        self.n = n

    @abstractmethod
    def __call__(self, x_ref: np.ndarray, x: np.ndarray, count: int) -> np.ndarray:
        """Abstract implementation of update strategy"""


class LastSeenUpdate(UpdateStrategy):
    """
    Updates reference dataset for drift detector using last seen method.

    Parameters
    ----------
    n : int
        Update with last n instances seen by the detector.
    """

    def __call__(self, x_ref: np.ndarray, x: np.ndarray, count: int) -> np.ndarray:
        x_updated = np.concatenate([x_ref, x], axis=0)
        return x_updated[-self.n :]


class ReservoirSamplingUpdate(UpdateStrategy):
    """
    Updates reference dataset for drift detector using reservoir sampling method.

    Parameters
    ----------
    n : int
        Update with reservoir sampling of size n.
    """

    def __call__(self, x_ref: np.ndarray, x: np.ndarray, count: int) -> np.ndarray:
        if x.shape[0] + count <= self.n:
            return np.concatenate([x_ref, x], axis=0)

        n_ref = x_ref.shape[0]
        output_size = min(self.n, n_ref + x.shape[0])
        shape = (output_size,) + x.shape[1:]
        x_reservoir = np.zeros(shape, dtype=x_ref.dtype)
        x_reservoir[:n_ref] = x_ref
        for item in x:
            count += 1
            if n_ref < self.n:
                x_reservoir[n_ref, :] = item
                n_ref += 1
            else:
                r = np.random.randint(0, count)
                if r < self.n:
                    x_reservoir[r, :] = item
        return x_reservoir


class BaseDrift:
    """Generic drift detector component handling preprocessing of data and correction"""

    def __init__(
        self,
        x_ref: ArrayLike,
        p_val: float = 0.05,
        x_ref_preprocessed: bool = False,
        update_x_ref: Optional[UpdateStrategy] = None,
        preprocess_fn: Optional[Callable[[ArrayLike], ArrayLike]] = None,
        correction: Literal["bonferroni", "fdr"] = "bonferroni",
    ) -> None:
        # Type checking
        if preprocess_fn is not None and not isinstance(preprocess_fn, Callable):
            raise ValueError("`preprocess_fn` is not a valid Callable.")
        if update_x_ref is not None and not isinstance(update_x_ref, UpdateStrategy):
            raise ValueError("`update_x_ref` is not a valid ReferenceUpdate class.")
        if correction not in ["bonferroni", "fdr"]:
            raise ValueError("`correction` must be `bonferroni` or `fdr`.")

        self._x_ref = x_ref
        self.x_ref_preprocessed = x_ref_preprocessed

        # Other attributes
        self.p_val = p_val
        self.update_x_ref = update_x_ref
        self.preprocess_fn = preprocess_fn
        self.correction = correction
        self.n = len(self._x_ref)  # type: ignore

        # Ref counter for preprocessed x
        self._x_refcount = 0

    @property
    def x_ref(self) -> np.ndarray:
        if not self.x_ref_preprocessed:
            self.x_ref_preprocessed = True
            if self.preprocess_fn is not None:
                self._x_ref = self.preprocess_fn(self._x_ref)

        self._x_ref = to_numpy(self._x_ref)
        return self._x_ref

    def _preprocess(self, x: ArrayLike) -> ArrayLike:
        """Data preprocessing before computing the drift scores."""
        if self.preprocess_fn is not None:
            x = self.preprocess_fn(x)
        return x


class BaseUnivariateDrift(BaseDrift):
    """
    Generic drift detector component which serves as a base class for methods using
    univariate tests. If n_features > 1, a multivariate correction is applied such
    that the false positive rate is upper bounded by the specified p-value, with
    equality in the case of independent features.
    """

    def __init__(
        self,
        x_ref: ArrayLike,
        p_val: float = 0.05,
        x_ref_preprocessed: bool = False,
        update_x_ref: Optional[UpdateStrategy] = None,
        preprocess_fn: Optional[Callable[[ArrayLike], ArrayLike]] = None,
        correction: Literal["bonferroni", "fdr"] = "bonferroni",
        n_features: Optional[int] = None,
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
        # lazy process n_features as needed
        if not isinstance(self._n_features, int):
            # compute number of features for the univariate tests
            if not isinstance(self.preprocess_fn, Callable) or self.x_ref_preprocessed:
                # infer features from preprocessed reference data
                self._n_features = self.x_ref.reshape(self.x_ref.shape[0], -1).shape[-1]
            else:
                # infer number of features after applying preprocessing step
                x = to_numpy(self.preprocess_fn(self._x_ref[0:1]))  # type: ignore
                self._n_features = x.reshape(x.shape[0], -1).shape[-1]

        return self._n_features

    @preprocess_x
    @abstractmethod
    def score(self, x: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """Abstract method to calculate feature score after preprocessing"""

    def _apply_correction(self, p_vals: np.ndarray) -> Tuple[int, float]:
        if self.correction == "bonferroni":
            threshold = self.p_val / self.n_features
            drift_pred = int((p_vals < threshold).any())
            return drift_pred, threshold
        elif self.correction == "fdr":
            n = p_vals.shape[0]
            i = np.arange(n) + 1
            p_sorted = np.sort(p_vals)
            q_threshold = self.p_val * i / n
            below_threshold = p_sorted < q_threshold
            try:
                idx_threshold = int(np.where(below_threshold)[0].max())
            except ValueError:  # sorted p-values not below thresholds
                return int(below_threshold.any()), q_threshold.min()
            return int(below_threshold.any()), q_threshold[idx_threshold]
        else:
            raise ValueError("`correction` needs to be either `bonferroni` or `fdr`.")

    @preprocess_x
    @update_x_ref
    def predict(
        self,
        x: ArrayLike,
        drift_type: Literal["batch", "feature"] = "batch",
    ) -> Dict[str, Union[int, float, np.ndarray]]:
        """
        Predict whether a batch of data has drifted from the reference data and update
        reference data using specified update strategy.

        Parameters
        ----------
        x : ArrayLike
            Batch of instances.
        drift_type : Literal["batch", "feature"], default "batch"
            Predict drift at the 'feature' or 'batch' level. For 'batch', the test
            statistics for each feature are aggregated using the Bonferroni or False
            Discovery Rate correction (if n_features>1).

        Returns
        -------
        Dictionary containing the drift prediction and optionally the feature level
                p-values, threshold after multivariate correction if needed and test
                statistics.
        """
        # compute drift scores
        p_vals, dist = self.score(x)

        # TODO: return both feature-level and batch-level drift predictions by default
        # values below p-value threshold are drift
        if drift_type == "feature":
            drift_pred = (p_vals < self.p_val).astype(int)
            threshold = self.p_val
        elif drift_type == "batch":
            drift_pred, threshold = self._apply_correction(p_vals)
        else:
            raise ValueError("`drift_type` needs to be either `feature` or `batch`.")

        # populate drift dict
        return {
            "is_drift": drift_pred,
            "p_val": p_vals,
            "threshold": threshold,
            "distance": dist,
        }
