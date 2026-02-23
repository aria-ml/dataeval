"""Domain Classifier based Out-of-Distribution detector."""

__all__ = []

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval.protocols import ArrayLike
from dataeval.shift._ood._base import BaseOOD, OODScoreOutput
from dataeval.shift._shared._domain_classifier import compute_class1_rates
from dataeval.utils.arrays import flatten_samples


class OODDomainClassifier(BaseOOD):
    """Domain Classifier based Out-of-Distribution detector.

    Uses a LightGBM classifier's ability to distinguish test samples from
    reference samples as an OOD signal. Samples that a classifier can easily
    identify as "not reference" are likely OOD.

    During :meth:`fit`, establishes a null distribution of per-point class-1
    prediction rates by running repeated k-fold CV on internal splits of the
    reference data. The threshold is set as ``mean + n_std * std`` of this
    null distribution.

    During :meth:`predict`/:meth:`score`, treats test data as class 1 and
    reference as class 0, runs repeated k-fold CV, and returns per-point
    class-1 rates. Points with rates exceeding the threshold are flagged OOD.

    Parameters
    ----------
    n_folds : int, default 5
        Number of cross-validation folds per repeat.
    n_repeats : int, default 5
        Number of times to repeat the k-fold split.
    n_std : float, default 2.0
        Number of standard deviations above the null mean for threshold.
    hyperparameters : dict or None, default None
        LightGBM hyperparameters.
    config : OODDomainClassifier.Config or None, default None
        Optional configuration object.

    Examples
    --------
    >>> ref = np.random.randn(200, 8).astype(np.float32)
    >>> test = np.random.randn(50, 8).astype(np.float32) + 3
    >>> detector = OODDomainClassifier(n_folds=3, n_repeats=3)
    >>> detector.fit(ref)
    >>> predictions = detector.predict(test)
    """

    @dataclass
    class Config:
        """
        Configuration for OODDomainClassifier.

        Attributes
        ----------
        n_folds : int, default 5
            Number of cross-validation folds.
        n_repeats : int, default 5
            Number of k-fold repeats.
        n_std : float, default 2.0
            Threshold multiplier for standard deviations above null mean.
        threshold_perc : float, default 95.0
            Percentile-based threshold (alternative to n_std).
        hyperparameters : dict or None, default None
            LightGBM hyperparameters.
        """

        n_folds: int = 5
        n_repeats: int = 5
        n_std: float = 2.0
        threshold_perc: float = 95.0
        hyperparameters: dict[str, Any] | None = None

    def __init__(
        self,
        n_folds: int | None = None,
        n_repeats: int | None = None,
        n_std: float | None = None,
        hyperparameters: dict[str, Any] | None = None,
        config: Config | None = None,
    ) -> None:
        super().__init__()
        self.config: OODDomainClassifier.Config = config or OODDomainClassifier.Config()

        self._n_folds = n_folds if n_folds is not None else self.config.n_folds
        self._n_repeats = n_repeats if n_repeats is not None else self.config.n_repeats
        self._n_std = n_std if n_std is not None else self.config.n_std
        self._hyperparameters = hyperparameters if hyperparameters is not None else self.config.hyperparameters

        self._x_ref: NDArray[np.float32] | None = None
        self._null_mean: float = 0.0
        self._null_std: float = 1.0
        self._threshold: float = 0.0

    def _preprocess(self, x: ArrayLike) -> NDArray[np.float32]:
        """Convert and flatten input to 2-D float32 array."""
        x_np = super()._preprocess(x)
        return flatten_samples(np.atleast_2d(x_np))

    def fit(self, x_ref: ArrayLike, threshold_perc: float | None = None) -> None:
        """Fit the detector using reference (in-distribution) data.

        Computes a null distribution of class-1 prediction rates by splitting
        the reference data internally (half as pseudo-class-0, half as
        pseudo-class-1) and running repeated k-fold CV. The OOD threshold
        is derived from this null distribution.

        Parameters
        ----------
        x_ref : ArrayLike
            Reference (in-distribution) data.
        threshold_perc : float or None, default None
            Percentage of reference data considered normal (0-100).
            If None, uses config.threshold_perc.
        """
        threshold_perc = threshold_perc if threshold_perc is not None else self.config.threshold_perc

        self._x_ref = flatten_samples(np.atleast_2d(np.asarray(x_ref, dtype=np.float32)))
        self._data_info = (self._x_ref.shape[1:], self._x_ref.dtype.type)

        # Build null distribution: split reference into two halves,
        # label them 0/1, and compute class-1 rates.
        n = len(self._x_ref)
        half = n // 2
        indices = np.random.permutation(n)
        x_null = self._x_ref[indices]
        y_null = np.concatenate([np.zeros(half, dtype=np.intp), np.ones(n - half, dtype=np.intp)])

        null_rates = compute_class1_rates(
            x_null,
            y_null,
            n_folds=self._n_folds,
            n_repeats=self._n_repeats,
            hyperparameters=self._hyperparameters,
        )

        self._null_mean = float(np.mean(null_rates))
        self._null_std = float(np.std(null_rates))
        self._threshold = self._null_mean + self._n_std * self._null_std

        # Compute reference scores for percentile-based thresholding
        self._ref_score = self.score(x_ref)
        self._threshold_perc = threshold_perc

    def _score(self, x: NDArray[np.float32], batch_size: int = int(1e10)) -> OODScoreOutput:  # noqa: ARG002
        """Compute per-point class-1 rates for test data vs reference."""
        x_ref = self._x_ref
        if x_ref is None:
            raise RuntimeError("Detector needs to be `fit` before calling score.")

        x_combined = np.concatenate([x_ref, x], axis=0)
        y = np.concatenate([np.zeros(len(x_ref), dtype=np.intp), np.ones(len(x), dtype=np.intp)])

        rates = compute_class1_rates(
            x_combined,
            y,
            n_folds=self._n_folds,
            n_repeats=self._n_repeats,
            hyperparameters=self._hyperparameters,
        )

        # Return only the test-point rates (second half)
        test_rates = rates[len(x_ref) :]
        return OODScoreOutput(instance_score=test_rates)
