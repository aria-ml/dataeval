"""
Multivariate Domain Classifier for drift detection.

Source code derived from NannyML 0.13.0
https://github.com/NannyML/nannyml/blob/main/nannyml/drift/multivariate/domain_classifier/calculator.py
https://github.com/NannyML/nannyml/blob/main/nannyml/base.py

Licensed under Apache Software License (Apache 2.0)
"""

__all__ = []

import copy
import logging
import warnings
from dataclasses import dataclass
from logging import Logger
from typing import Any, Literal

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from typing_extensions import Self

from dataeval.config import get_max_processes, get_seed
from dataeval.shift._drift._chunk import Chunk, Chunker, CountBasedChunker, SizeBasedChunker
from dataeval.shift._drift._thresholds import ConstantThreshold, Threshold
from dataeval.types import Output, set_metadata
from dataeval.utils.arrays import flatten_samples

logger = logging.getLogger(__name__)


def _validate(data: pl.DataFrame, expected_features: int | None = None) -> int:
    """Validate DataFrame has data and expected number of features."""
    if data.is_empty():
        raise ValueError("data contains no rows. Please provide a valid data set.")
    if expected_features is not None and data.shape[-1] != expected_features:
        raise ValueError(f"expected '{expected_features}' features in data set:\n\t{data}")
    return data.shape[-1]


DEFAULT_LGBM_HYPERPARAMS = {
    "boosting_type": "gbdt",
    "class_weight": None,
    "colsample_bytree": 1.0,
    "deterministic": True,
    "importance_type": "split",
    "learning_rate": 0.1,
    "max_depth": -1,
    "min_child_samples": 20,
    "min_child_weight": 0.001,
    "min_split_gain": 0.0,
    "n_estimators": 100,
    "num_leaves": 31,
    "objective": None,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "subsample": 1.0,
    "subsample_for_bin": 200000,
    "subsample_freq": 0,
    "verbosity": -1,
}


class DriftMVDCOutput(Output[pl.DataFrame]):
    """Results of the multivariate domain classifier drift detection."""

    def __init__(self, results_data: pl.DataFrame) -> None:
        """Initialize a DriftMVDCOutput results object.

        Parameters
        ----------
        results_data : pl.DataFrame
            Results data returned by a DomainClassifierCalculator.
        """
        self._data = results_data.clone()

    def data(self) -> pl.DataFrame:
        """Return a copy of the results data."""
        return self._data.clone()

    @property
    def empty(self) -> bool:
        """Check if the results are empty."""
        return self._data is None or self._data.is_empty()

    @property
    def plot_type(self) -> Literal["drift_mvdc"]:
        """Return the plot type identifier."""
        return "drift_mvdc"

    def __len__(self) -> int:
        """Return the number of rows in the results."""
        return 0 if self.empty else len(self._data)

    def filter(self, period: str = "all", metrics: str | None = None) -> Self:
        """Filter results by period and optionally by metric.

        Parameters
        ----------
        period : str, default "all"
            Filter by period: "all", "reference", or "analysis".
        metrics : str or None, default None
            Metric name to filter by. Currently only "domain_classifier_auroc" is supported.
            If None, all metrics are included.

        Returns
        -------
        DriftMVDCOutput
            Filtered results object.

        Raises
        ------
        ValueError
            If metrics parameter is not a string or None.
        KeyError
            If the requested metric is not available.
        """
        if metrics is not None and not isinstance(metrics, str):
            raise ValueError("metrics value provided is not a valid metric")

        data = self._data
        if period != "all":
            data = data.filter(pl.col("chunk_period") == period)

        # If a specific metric is requested, validate and filter columns
        if metrics is not None:
            if metrics != "domain_classifier_auroc":
                raise KeyError(f"Metric '{metrics}' not found. Available metric: 'domain_classifier_auroc'")

            # Select chunk columns and requested metric columns
            chunk_cols = [col for col in data.columns if col.startswith("chunk_")]
            metric_cols = [col for col in data.columns if col.startswith(f"{metrics}_")]
            data = data.select(chunk_cols + metric_cols)

        res = copy.deepcopy(self)
        res._data = data
        return res


class _DomainClassifierCalculator:
    """Internal calculator for domain classifier drift detection."""

    def __init__(
        self,
        chunker: Chunker | None = None,
        cv_folds_num: int = 5,
        hyperparameters: dict[str, Any] | None = None,
        threshold: Threshold = ConstantThreshold(lower=0.45, upper=0.65),
        logger_instance: Logger | None = None,
    ) -> None:
        """Create a new DomainClassifierCalculator instance.

        Parameters
        ----------
        chunker : Chunker, default=None
            The `Chunker` used to split the data sets into a lists of chunks.
        cv_folds_num: int, default=5
            Number of cross-validation folds to use when calculating DC discrimination value.
        hyperparameters : dict[str, Any], default = None
            A dictionary used to provide your own custom hyperparameters when training the discrimination model.
        threshold: Threshold, default=ConstantThreshold
            The threshold you wish to evaluate values on.
        logger_instance: Logger, default=None
            Logger instance to use for logging. If None, uses the module logger.
        """
        self.chunker = chunker if isinstance(chunker, Chunker) else CountBasedChunker(10)
        self.result: DriftMVDCOutput | None = None
        self.n_features: int | None = None
        self._logger = logger_instance if isinstance(logger_instance, Logger) else logger

        self.cv_folds_num = cv_folds_num
        self.hyperparameters = DEFAULT_LGBM_HYPERPARAMS if hyperparameters is None else hyperparameters
        self.threshold = threshold

    def fit(self, reference_data: pl.DataFrame) -> Self:
        """Train the calculator using reference data."""
        self.n_features = _validate(reference_data)
        self._logger.debug(f"fitting {str(self)}")

        self._x_ref = reference_data
        chunks = self.chunker.split(self._x_ref)

        # Create records with flattened column names
        records = [
            {f"chunk_{k}": v for k, v in chunk.dict().items()}
            | {
                "chunk_period": "reference",
                "domain_classifier_auroc_value": self._calculate_chunk(chunk=chunk),
            }
            for chunk in chunks
        ]

        res = pl.DataFrame(records)
        res = self._populate_alert_thresholds(res)
        self.result = DriftMVDCOutput(results_data=res)

        return self

    @set_metadata
    def calculate(self, data: pl.DataFrame) -> DriftMVDCOutput:
        """Perform calculation on the provided data."""
        if self.result is None:
            raise RuntimeError("must run fit with reference data before running calculate")
        _validate(data, self.n_features)
        self._logger.debug(f"calculating {str(self)}")

        chunks = self.chunker.split(data)

        # Create records with flattened column names
        records = [
            {f"chunk_{k}": v for k, v in chunk.dict().items()}
            | {
                "chunk_period": "analysis",
                "domain_classifier_auroc_value": self._calculate_chunk(chunk=chunk),
            }
            for chunk in chunks
        ]

        res = pl.DataFrame(records)
        res = self._populate_alert_thresholds(res)

        # Combine with reference results
        self.result = self.result.filter(period="reference")
        self.result._data = pl.concat([self.result._data, res])

        return self.result

    def _calculate_chunk(self, chunk: Chunk) -> float:
        """Calculate AUROC for a single chunk."""
        if self.result is None:
            # Use chunk indices to identify reference chunk's location
            df_X = self._x_ref
            y = np.zeros(len(df_X), dtype=np.intp)
            y[chunk.start_index : chunk.end_index + 1] = 1
        else:
            chunk_X = chunk.data
            reference_X = self._x_ref
            chunk_y = np.ones(len(chunk_X), dtype=np.intp)
            reference_y = np.zeros(len(reference_X), dtype=np.intp)
            df_X = pl.concat([reference_X, chunk_X])
            y = np.concatenate([reference_y, chunk_y])

        # Extract feature names from Polars DataFrame and convert to numpy
        feature_names = df_X.columns
        X_numpy = df_X.to_numpy()

        skf = StratifiedKFold(n_splits=self.cv_folds_num)
        all_preds: list[NDArray[np.float32]] = []
        all_tgts: list[NDArray[np.intp]] = []

        # Suppress sklearn's feature name mismatch warning.
        # We pass feature names to LightGBM.fit() but sklearn's predict_proba() validation
        # still complains because the test arrays are plain numpy without feature metadata.
        # This is expected behavior - we maintain feature order consistency via array indexing.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            for train_index, test_index in skf.split(X_numpy, y):
                _trx = X_numpy[train_index]
                _try = y[train_index]
                _tsx = X_numpy[test_index]
                _tsy = y[test_index]
                model = LGBMClassifier(
                    **self.hyperparameters,
                    n_jobs=get_max_processes(),
                    random_state=get_seed(),
                )
                # Pass feature names to LightGBM for internal bookkeeping
                model.fit(_trx, _try, feature_name=feature_names)
                preds = np.asarray(model.predict_proba(_tsx), dtype=np.float32)[:, 1]
                all_preds.append(preds)
                all_tgts.append(_tsy)

        np_all_preds = np.concatenate(all_preds, axis=0)
        np_all_tgts = np.concatenate(all_tgts, axis=0)
        result = roc_auc_score(np_all_tgts, np_all_preds)
        return 0.5 if result == np.nan else float(result)

    def _populate_alert_thresholds(self, result_data: pl.DataFrame) -> pl.DataFrame:
        """Populate alert threshold columns."""
        if self.result is None:
            self._threshold_values = self.threshold.calculate(
                data=result_data["domain_classifier_auroc_value"].to_numpy(),
                lower_limit=0.0,
                upper_limit=1.0,
                logger=self._logger,
            )

        result_data = result_data.with_columns(
            [
                pl.lit(self._threshold_values[1]).alias("domain_classifier_auroc_upper_threshold"),
                pl.lit(self._threshold_values[0]).alias("domain_classifier_auroc_lower_threshold"),
            ]
        )

        return result_data.with_columns(
            (
                (pl.col("domain_classifier_auroc_value") > pl.col("domain_classifier_auroc_upper_threshold"))
                | (pl.col("domain_classifier_auroc_value") < pl.col("domain_classifier_auroc_lower_threshold"))
            ).alias("domain_classifier_auroc_alert")
        )


class DriftMVDC:
    """Multivariant Domain Classifier

    Parameters
    ----------
    n_folds : int, default 5
        Number of cross-validation (CV) folds.
    chunk_size : int or None, default None
        Number of samples in a chunk used in CV, will get one metric & prediction per chunk.
    chunk_count : int or None, default None
        Number of total chunks used in CV, will get one metric & prediction per chunk.
    threshold : Tuple[float, float], default (0.45, 0.65)
        (lower, upper) metric bounds on roc_auc for identifying :term:`drift<Drift>`.
    config : DriftMVDC.Config or None, default None
        Optional configuration object with default parameters. Parameters
        specified directly in __init__ will override config defaults.

    Examples
    --------
    Using configuration:

    >>> config = DriftMVDC.Config(n_folds=10, threshold=(0.4, 0.6))
    >>> detector = DriftMVDC(config=config)
    """

    @dataclass
    class Config:
        """
        Configuration for DriftMVDC detector.

        Attributes
        ----------
        n_folds : int, default 5
            Number of cross-validation folds.
        chunk_size : int or None, default None
            Number of samples in a chunk.
        chunk_count : int or None, default None
            Number of total chunks.
        threshold : tuple[float, float], default (0.45, 0.65)
            (lower, upper) metric bounds on roc_auc for drift identification.
        """

        n_folds: int = 5
        chunk_size: int | None = None
        chunk_count: int | None = None
        threshold: tuple[float, float] = (0.45, 0.65)

    def __init__(
        self,
        n_folds: int | None = None,
        chunk_size: int | None = None,
        chunk_count: int | None = None,
        threshold: tuple[float, float] | None = None,
        config: Config | None = None,
    ) -> None:
        # Store config or create default
        self.config: DriftMVDC.Config = config or DriftMVDC.Config()

        # Use config defaults if parameters not specified
        n_folds = n_folds if n_folds is not None else self.config.n_folds
        chunk_size = chunk_size if chunk_size is not None else self.config.chunk_size
        chunk_count = chunk_count if chunk_count is not None else self.config.chunk_count
        threshold = threshold if threshold is not None else self.config.threshold

        self.threshold: tuple[float, float] = max(0.0, min(threshold)), min(1.0, max(threshold))
        chunker = (
            CountBasedChunker(10 if chunk_count is None else chunk_count)
            if chunk_size is None
            else SizeBasedChunker(chunk_size)
        )
        self._calc = _DomainClassifierCalculator(
            cv_folds_num=n_folds,
            chunker=chunker,
            threshold=ConstantThreshold(lower=self.threshold[0], upper=self.threshold[1]),
        )

    def fit(self, x_ref: ArrayLike) -> Self:
        """
        Fit the domain classifier on the training dataframe

        Parameters
        ----------
        x_ref : ArrayLike
            Reference data with dim[n_samples, n_features].

        Returns
        -------
        DriftMVDC

        """
        # for 1D input, assume that is 1 sample: dim[1,n_features]
        self.x_ref: pl.DataFrame = pl.DataFrame(flatten_samples(np.atleast_2d(np.asarray(x_ref))))
        self.n_features: int = self.x_ref.shape[-1]
        self._calc.fit(self.x_ref)
        return self

    def predict(self, x: ArrayLike) -> DriftMVDCOutput:
        """
        Perform :term:`inference<Inference>` on the test dataframe

        Parameters
        ----------
        x : ArrayLike
            Test (analysis) data with dim[n_samples, n_features].

        Returns
        -------
        DomainClassifierDriftResult
        """
        self.x_test: pl.DataFrame = pl.DataFrame(flatten_samples(np.atleast_2d(np.asarray(x))))
        if self.x_test.shape[-1] != self.n_features:
            raise ValueError("Reference and test embeddings have different number of features")

        return self._calc.calculate(self.x_test)
