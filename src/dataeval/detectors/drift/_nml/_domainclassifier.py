"""
Source code derived from NannyML 0.13.0
https://github.com/NannyML/nannyml/blob/main/nannyml/drift/multivariate/domain_classifier/calculator.py

Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from dataeval.config import get_max_processes, get_seed
from dataeval.detectors.drift._nml._base import AbstractCalculator, _create_multilevel_index
from dataeval.detectors.drift._nml._chunk import Chunk, Chunker
from dataeval.detectors.drift._nml._thresholds import ConstantThreshold, Threshold
from dataeval.outputs._base import set_metadata
from dataeval.outputs._drift import DriftMVDCOutput

logger = logging.getLogger(__name__)

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


class DomainClassifierCalculator(AbstractCalculator):
    """
    DomainClassifierCalculator implementation.

    Uses Drift Detection Classifier's cross validated performance as a measure of drift.
    """

    def __init__(
        self,
        chunker: Chunker | None = None,
        cv_folds_num: int = 5,
        hyperparameters: dict[str, Any] | None = None,
        threshold: Threshold = ConstantThreshold(lower=0.45, upper=0.65),
    ) -> None:
        """
        Create a new DomainClassifierCalculator instance.

        Parameters
        -----------
        chunker : Chunker, default=None
            The `Chunker` used to split the data sets into a lists of chunks.
        cv_folds_num: Optional[int]
            Number of cross-validation folds to use when calculating DC discrimination value.
        hyperparameters : dict[str, Any], default = None
            A dictionary used to provide your own custom hyperparameters when training the discrimination model.
            Check out the available hyperparameter options in the
            `LightGBM docs <https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html>`_.
        threshold: Threshold, default=ConstantThreshold
            The threshold you wish to evaluate values on. Defaults to a ConstantThreshold with lower value
            of 0.45 and upper value of 0.65.
        """
        super().__init__(chunker, logger)

        self.cv_folds_num = cv_folds_num
        self.hyperparameters = DEFAULT_LGBM_HYPERPARAMS if hyperparameters is None else hyperparameters
        self.threshold = threshold
        self.result: DriftMVDCOutput | None = None

    def _fit(self, reference_data: pd.DataFrame) -> DriftMVDCOutput:
        """Fits the DC calculator to a set of reference data."""
        self._x_ref = reference_data
        result = self._calculate(data=self._x_ref)
        result._data[("chunk", "period")] = "reference"

        return result

    @set_metadata
    def _calculate(self, data: pd.DataFrame) -> DriftMVDCOutput:
        """Calculate the data DC calculator metric for a given data set."""
        chunks = self.chunker.split(data)

        res = pd.DataFrame.from_records(
            [
                {
                    **chunk.dict(),
                    "period": "analysis",
                    "classifier_auroc_value": self._calculate_chunk(chunk=chunk),
                }
                for chunk in chunks
            ]
        )

        multilevel_index = _create_multilevel_index(chunks, "domain_classifier_auroc", ["value"])
        res.columns = multilevel_index
        res = res.reset_index(drop=True)

        res = self._populate_alert_thresholds(res)

        if self.result is None:
            self.result = DriftMVDCOutput(results_data=res)
        else:
            self.result = self.result.filter(period="reference")
            self.result._data = pd.concat([self.result._data, res], ignore_index=True)
        return self.result

    def _calculate_chunk(self, chunk: Chunk) -> float:
        if self.result is None:
            # Use information from chunk indices to identify reference chunk's location. This is possible because
            # both the internal reference data copy and the chunk data were sorted by timestamp, so these
            # indices align. This way we eliminate the need to combine these two data frames and drop duplicate rows,
            # which is a costly operation.
            df_X = self._x_ref
            y = np.zeros(len(df_X), dtype=np.intp)
            y[chunk.start_index : chunk.end_index + 1] = 1
        else:
            chunk_X = chunk.data
            reference_X = self._x_ref
            chunk_y = np.ones(len(chunk_X), dtype=np.intp)
            reference_y = np.zeros(len(reference_X), dtype=np.intp)
            df_X = pd.concat([reference_X, chunk_X], ignore_index=True)
            y = np.concatenate([reference_y, chunk_y])

        skf = StratifiedKFold(n_splits=self.cv_folds_num)
        all_preds: list[NDArray[np.float32]] = []
        all_tgts: list[NDArray[np.intp]] = []
        for i, (train_index, test_index) in enumerate(skf.split(df_X, y)):
            _trx = df_X.iloc[train_index]
            _try = y[train_index]
            _tsx = df_X.iloc[test_index]
            _tsy = y[test_index]
            model = LGBMClassifier(**self.hyperparameters, n_jobs=get_max_processes(), random_state=get_seed())
            model.fit(_trx, _try)
            preds = np.asarray(model.predict_proba(_tsx), dtype=np.float32)[:, 1]
            all_preds.append(preds)
            all_tgts.append(_tsy)

        np_all_preds = np.concatenate(all_preds, axis=0)
        np_all_tgts = np.concatenate(all_tgts, axis=0)
        result = roc_auc_score(np_all_tgts, np_all_preds)
        return 0.5 if result == np.nan else float(result)

    def _populate_alert_thresholds(self, result_data: pd.DataFrame) -> pd.DataFrame:
        if self.result is None:
            self._threshold_values = self.threshold.calculate(
                data=result_data.loc[:, ("domain_classifier_auroc", "value")],  # type: ignore | dataframe loc
                lower_limit=0.0,
                upper_limit=1.0,
                logger=self._logger,
            )

        result_data[("domain_classifier_auroc", "upper_threshold")] = self._threshold_values[1]
        result_data[("domain_classifier_auroc", "lower_threshold")] = self._threshold_values[0]
        result_data[("domain_classifier_auroc", "alert")] = result_data.apply(
            lambda row: bool(
                row["domain_classifier_auroc", "value"] > row["domain_classifier_auroc", "upper_threshold"]
                or row["domain_classifier_auroc", "value"] < row["domain_classifier_auroc", "lower_threshold"]
            ),
            axis=1,
        )
        return result_data
