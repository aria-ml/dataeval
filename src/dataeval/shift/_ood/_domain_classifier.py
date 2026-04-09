"""Domain Classifier based Out-of-Distribution detector."""

__all__ = []

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from dataeval.exceptions import NotFittedError
from dataeval.protocols import ArrayLike, FeatureExtractor
from dataeval.shift._ood._base import BaseOOD, ExtractorMixin, OODScoreOutput
from dataeval.shift._shared._domain_classifier import compute_class1_rates
from dataeval.utils._internal import flatten_samples


class OODDomainClassifier(ExtractorMixin, BaseOOD):
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

    Note: By default, this detector uses the ``n_std`` based threshold for
    predictions. If a value for ``threshold_perc`` is provided (either directly
    or via config), it will use percentile-based thresholding from reference
    scores instead.

    Parameters
    ----------
    n_folds : int, default 5
        Number of cross-validation folds per repeat.
    n_repeats : int, default 5
        Number of times to repeat the k-fold split.
    n_std : float, default 2.0
        Number of standard deviations above the null mean for threshold.
        Used when threshold_perc is not explicitly set.
    hyperparameters : dict or None, default None
        LightGBM hyperparameters.
    threshold_perc : float or None, default None
        Percentage of reference data considered normal (0-100).
        If provided, overrides ``n_std`` for percentile-based thresholding.
    extractor : FeatureExtractor or None, default None
        Feature extractor for transforming input data before scoring.
        When provided, raw data is passed through the extractor in both
        :meth:`fit` and :meth:`score`/:meth:`predict`. When None, data
        is used as-is (must be array-like embeddings).
    config : OODDomainClassifier.Config or None, default None
        Optional configuration object.

    Examples
    --------
    >>> ref = np.random.randn(200, 8).astype(np.float32)
    >>> test = np.random.randn(50, 8).astype(np.float32) + 3
    >>> detector = OODDomainClassifier(n_folds=3, n_repeats=3)
    >>> detector.fit(ref)
    OODDomainClassifier(n_folds=3, n_repeats=3, n_std=2.0, threshold_perc=None, hyperparameters=None, extractor=None, fitted=True)
    >>> predictions = detector.predict(test)
    """  # noqa: E501

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
            Used when threshold_perc is None.
        threshold_perc : float or None, default None
            Percentile-based threshold. If provided, overrides n_std.
        hyperparameters : dict or None, default None
            LightGBM hyperparameters.
        extractor : FeatureExtractor or None, default None
            Feature extractor for transforming input data before scoring.
        """

        n_folds: int = 5
        n_repeats: int = 5
        n_std: float = 2.0
        threshold_perc: float | None = None
        hyperparameters: dict[str, Any] | None = None
        extractor: FeatureExtractor | None = None

    def __init__(
        self,
        n_folds: int | None = None,
        n_repeats: int | None = None,
        n_std: float | None = None,
        hyperparameters: dict[str, Any] | None = None,
        threshold_perc: float | None = None,
        extractor: FeatureExtractor | None = None,
        config: Config | None = None,
    ) -> None:
        base_config = config or OODDomainClassifier.Config()

        self._threshold_perc_set = threshold_perc is not None or (config is not None and config.threshold_perc is not None)
        perc = threshold_perc if threshold_perc is not None else (base_config.threshold_perc or 95.0)
        super().__init__(perc)

        self._n_folds = n_folds if n_folds is not None else base_config.n_folds
        self._n_repeats = n_repeats if n_repeats is not None else base_config.n_repeats
        self._n_std = n_std if n_std is not None else base_config.n_std
        self._hyperparameters = hyperparameters if hyperparameters is not None else base_config.hyperparameters
        self._extractor = extractor if extractor is not None else base_config.extractor
        self.config: OODDomainClassifier.Config = OODDomainClassifier.Config(
            n_folds=self._n_folds,
            n_repeats=self._n_repeats,
            n_std=self._n_std,
            threshold_perc=threshold_perc if threshold_perc is not None else base_config.threshold_perc,
            hyperparameters=self._hyperparameters,
            extractor=self._extractor,
        )

        self._reference_data: NDArray[np.float32] | None = None
        self._null_mean: float = 0.0
        self._null_std: float = 1.0
        self._threshold: float = 0.0

    def _preprocess(self, x: ArrayLike) -> NDArray[np.float32]:
        """Convert and flatten input to 2-D float32 array."""
        x_np = super()._preprocess(x)
        return flatten_samples(np.atleast_2d(x_np))

    def fit(self, reference_data: Any) -> Self:
        """Fit the detector using reference (in-distribution) data.

        Computes a null distribution of class-1 prediction rates by splitting
        the reference data internally (half as pseudo-class-0, half as
        pseudo-class-1) and running repeated k-fold CV. The OOD threshold
        is derived from this null distribution.

        Parameters
        ----------
        reference_data : Any
            Reference (in-distribution) data. When an extractor is
            configured, this can be any data type accepted by the
            extractor. Otherwise, must be array-like embeddings.

        Returns
        -------
        Self
            The fitted detector (for method chaining).
        """
        self._reference_data = self._preprocess(reference_data)
        self._data_info = (self._reference_data.shape[1:], self._reference_data.dtype.type)

        # Build null distribution: split reference into two halves,
        # label them 0/1, and compute class-1 rates.
        n = len(self._reference_data)
        half = n // 2
        indices = np.random.permutation(n)
        x_null = self._reference_data[indices]
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
        self._ref_score = self.score(reference_data)
        return self

    def _threshold_score(self, ood_type: Literal["feature", "instance"] = "instance") -> np.floating:
        """Get the threshold score. Prefers n_std threshold unless threshold_perc was explicitly set."""
        if not self._threshold_perc_set and ood_type == "instance":
            return np.float64(self._threshold)
        return super()._threshold_score(ood_type)

    def _score(self, x: NDArray[np.float32], batch_size: int | None = None) -> OODScoreOutput:  # noqa: ARG002
        """Compute per-point class-1 rates for test data vs reference."""
        x_ref = self._reference_data
        if x_ref is None:
            raise NotFittedError("Detector needs to be `fit` before calling score.")

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
