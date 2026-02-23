"""
Core domain classifier math: LightGBM-based AUROC and per-point class-1 rate computation.

Shared by DriftDomainClassifier (AUROC aggregation) and OODDomainClassifier (per-point scoring).

Source code derived from NannyML 0.13.0
https://github.com/NannyML/nannyml/blob/main/nannyml/drift/multivariate/domain_classifier/calculator.py

Licensed under Apache Software License (Apache 2.0)
"""

__all__ = []

import warnings
from typing import Any

import numpy as np
from lightgbm import LGBMClassifier
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from dataeval.config import get_max_processes, get_seed

DEFAULT_LGBM_HYPERPARAMS: dict[str, Any] = {
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


def compute_auroc(
    x: NDArray[np.float32],
    y: NDArray[np.intp],
    n_folds: int = 5,
    hyperparameters: dict[str, Any] | None = None,
) -> tuple[float, NDArray[np.float32], NDArray[np.float32]]:
    """Compute AUROC of a domain classifier distinguishing two classes.

    Parameters
    ----------
    x : NDArray[np.float32]
        Combined feature matrix, shape (n_samples, n_features).
    y : NDArray[np.intp]
        Binary labels (0=reference, 1=test).
    n_folds : int
        Number of stratified k-fold cross-validation splits.
    hyperparameters : dict or None
        LightGBM hyperparameters.

    Returns
    -------
    tuple[float, NDArray[np.float32], NDArray[np.float32]]
        - Overall AUROC score (0.5 = no discrimination, 1.0 = perfect).
        - Per-fold AUROC values, shape (n_folds,).
        - Mean feature importances across folds, shape (n_features,).
    """
    hyperparameters = DEFAULT_LGBM_HYPERPARAMS if hyperparameters is None else hyperparameters
    feature_names = [f"f{i}" for i in range(x.shape[1])]

    skf = StratifiedKFold(n_splits=n_folds)
    all_preds: list[NDArray[np.float32]] = []
    all_tgts: list[NDArray[np.intp]] = []
    fold_aurocs: list[float] = []
    fold_importances: list[NDArray[np.float32]] = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        for train_index, test_index in skf.split(x, y):
            _trx = x[train_index]
            _try = y[train_index]
            _tsx = x[test_index]
            _tsy = y[test_index]
            model = LGBMClassifier(
                **hyperparameters,
                n_jobs=get_max_processes(),
                random_state=get_seed(),
            )
            model.fit(_trx, _try, feature_name=feature_names)
            preds = np.asarray(model.predict_proba(_tsx), dtype=np.float32)[:, 1]
            all_preds.append(preds)
            all_tgts.append(_tsy)
            fold_auroc = roc_auc_score(_tsy, preds)
            fold_aurocs.append(0.5 if fold_auroc == np.nan else float(fold_auroc))
            fold_importances.append(np.asarray(model.feature_importances_, dtype=np.float32))

    np_all_preds = np.concatenate(all_preds, axis=0)
    np_all_tgts = np.concatenate(all_tgts, axis=0)
    result = roc_auc_score(np_all_tgts, np_all_preds)
    auroc = 0.5 if result == np.nan else float(result)
    return (
        auroc,
        np.array(fold_aurocs, dtype=np.float32),
        np.mean(fold_importances, axis=0).astype(np.float32),
    )


def compute_class1_rates(
    x: NDArray[np.float32],
    y: NDArray[np.intp],
    n_folds: int = 5,
    n_repeats: int = 5,
    hyperparameters: dict[str, Any] | None = None,
) -> NDArray[np.float32]:
    """Return per-point average class-1 prediction probability across repeated k-fold CV.

    Each sample appears in a test fold exactly ``n_repeats`` times.  The
    returned array averages the class-1 predicted probability over those
    appearances, giving a per-instance "how likely is this point to look
    like class 1?" score.

    Parameters
    ----------
    x : NDArray[np.float32]
        Combined feature matrix, shape (n_samples, n_features).
    y : NDArray[np.intp]
        Binary labels (0=reference, 1=test).
    n_folds : int
        Number of stratified k-fold cross-validation splits per repeat.
    n_repeats : int
        Number of times to repeat the k-fold split (each with a different
        random partition).
    hyperparameters : dict or None
        LightGBM hyperparameters.  Defaults to ``DEFAULT_LGBM_HYPERPARAMS``.

    Returns
    -------
    NDArray[np.float32]
        Per-point average class-1 probability, shape (n_samples,).
    """
    hyperparameters = DEFAULT_LGBM_HYPERPARAMS if hyperparameters is None else hyperparameters
    feature_names = [f"f{i}" for i in range(x.shape[1])]

    n_samples = len(x)
    sum_preds = np.zeros(n_samples, dtype=np.float64)
    count_preds = np.zeros(n_samples, dtype=np.int32)

    rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=get_seed())

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        for train_index, test_index in rskf.split(x, y):
            model = LGBMClassifier(
                **hyperparameters,
                n_jobs=get_max_processes(),
                random_state=get_seed(),
            )
            model.fit(x[train_index], y[train_index], feature_name=feature_names)
            preds = np.asarray(model.predict_proba(x[test_index]), dtype=np.float32)[:, 1]
            sum_preds[test_index] += preds
            count_preds[test_index] += 1

    # Every sample should appear exactly n_repeats times; guard against zeros
    count_preds = np.maximum(count_preds, 1)
    return (sum_preds / count_preds).astype(np.float32)
