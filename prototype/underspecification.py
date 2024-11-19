from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats


def spearman_permutation(x: NDArray[np.float_], y: NDArray[np.float_]) -> Any:
    def statistic(x):
        rs = stats.spearmanr(x, y).statistic  # type: ignore 
        transformed = rs * np.sqrt(dof / ((rs+1.0)*(1.0-rs)))
        return transformed 
    dof = x.shape[0] - 2
    test_results = stats.permutation_test((x,), statistic, alternative='two-sided',permutation_type='pairings')
    return test_results


def spearman_test(x: NDArray[np.float_], y: NDArray[np.float_]) -> dict[str, NDArray[np.float_] | float]:
    n_samples = x.shape[0]
    test_info = spearman_permutation(x, y) if n_samples < 20 else stats.spearmanr(x, y)
    test_results = {
        "statistic": test_info.statistic, # type: ignore
        "pvalue": test_info.pvalue, # type: ignore
        "null_distribution": test_info.null_distribution, # type: ignore 
    }
    return test_results


def kappa_agreement_statistic(x: NDArray[np.float_], y: NDArray[np.float_]) -> float:
    x_pred = np.argmax(x, axis=0)
    y_pred = np.argmax(y, axis=0)
    relative_agreement = (x_pred == y_pred).sum() / x_pred.shape[0]
    agreement_chance = 1/x.shape[1]
    kappa = (relative_agreement - agreement_chance) / (1 - agreement_chance)
    return kappa




