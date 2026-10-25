"""
Average-precision bounds from risk and count endpoints.

An upper envelope turns lower bounds on cost-weighted risk into an AP ceiling no scorer can beat;
a lower staircase turns a frozen scorer's per-threshold count endpoints into an AP it provably reaches.
"""

__all__ = []

import numpy as np
from numpy.typing import NDArray

from dataeval._experimental import experimental


def _precision_upper(
    recall: NDArray[np.float64], risk_lower: NDArray[np.float64], costs: NDArray[np.float64], mu_upper: float
) -> NDArray[np.float64]:
    """Precision upper bound ``mu r / (mu r + F(r))`` with ``F(r) = max(0, max_cost (risk - mu (1 - r)) / cost)``."""
    pos = costs > 0
    risk = np.minimum(risk_lower[pos], mu_upper)[:, None]
    fp = ((risk - mu_upper * (1 - recall)) / costs[pos, None]).max(axis=0, initial=0.0)
    tp = mu_upper * recall
    return np.divide(tp, tp + fp, out=np.ones_like(tp), where=tp + fp > 0)


@experimental
def ap_upper_envelope(  # noqa: C901
    risk_lower: NDArray[np.float64], costs: NDArray[np.float64], mu_upper: float, *, intervals: int = 1000
) -> float:
    """
    Upper bound on the population AP of any scorer, from lower bounds on the Bayes one-vs-rest cost-weighted risk.

    .. warning::
       This feature is experimental and may change or be removed in future releases.

    Every threshold of every scorer has recall ``r`` and false positives per unit ``f`` with
    ``mu (1 - r) + cost * f >= risk(cost)`` [1]_. So at recall ``r`` at least
    ``F(r) = max(0, max_cost (risk(cost) - mu (1 - r)) / cost)`` false positives are needed and precision is at
    most ``mu r / (mu r + F(r))`` (1 at ``r = 0``) [3]_. This curve is nonincreasing, so it bounds the
    interpolated precision curve [2]_, and its left sum on a uniform grid of ``intervals`` overestimates its
    integral by at most ``1 / intervals``.

    Classification has no output cap, so the recall cap is 1 and the zero-cost risk is 0 (emitting everything).
    Risks above ``mu_upper`` (possible only off the confidence event) are clipped to it, which is conservative.

    Parameters
    ----------
    risk_lower : NDArray[np.float64]
        Shape ``(n_costs,)`` lower bound on the Bayes cost-weighted risk, the least
        ``E[missed positives] + cost * E[false positives]`` per unit that any scorer's threshold reaches, aligned
        with ``costs``; 0 at cost 0.
    costs : NDArray[np.float64]
        Shape ``(n_costs,)`` distinct non-negative costs of a false positive relative to a missed positive,
        exactly one of them 0.
    mu_upper : float
        Upper bound on the prevalence ``mu`` (positives per unit), at least 0.
    intervals : int, default 1000
        Number of uniform recall intervals in the left sum.

    Returns
    -------
    float
        AP upper bound in ``[0, 1]``; 1.0 when ``mu_upper == 0``.

    References
    ----------
    .. [1] Reid, M. D. and Williamson, R. C. (2011). Information, divergence and risk for
       binary experiments. Journal of Machine Learning Research, 12, 731-817.
    .. [2] Davis, J. and Goadrich, M. (2006). The relationship between precision-recall and ROC
       curves. Proceedings of the 23rd International Conference on Machine Learning, 233-240.
    .. [3] DFE manuscript, operating-point inequalities (precision-recall and AP bounds).
    """
    risk_lower = np.asarray(risk_lower, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    if risk_lower.ndim != 1 or risk_lower.shape != costs.shape or not risk_lower.size:
        raise ValueError("risk_lower and costs must be aligned nonempty 1D arrays.")
    if not np.isfinite(np.r_[risk_lower, costs, mu_upper]).all():
        raise ValueError("risk_lower, costs and mu_upper must be finite.")
    if np.any(risk_lower < 0) or np.any(costs < 0) or mu_upper < 0:
        raise ValueError("risk_lower, costs and mu_upper must be non-negative.")
    if len(np.unique(costs)) != len(costs) or np.count_nonzero(costs == 0) != 1:
        raise ValueError("costs must be distinct with exactly one zero cost.")
    if risk_lower[costs == 0][0] != 0:
        raise ValueError("risk_lower must be 0 at cost 0 (emitting everything misses nothing).")
    if not isinstance(intervals, int) or intervals < 1:
        raise ValueError(f"intervals must be a positive integer, got {intervals}.")
    precision = _precision_upper(np.arange(intervals) / intervals, risk_lower, costs, mu_upper)
    return min(1.0, float(precision.mean()))


@experimental
def ap_lower_staircase(tp_lower: NDArray[np.float64], fp_upper: NDArray[np.float64], mu_upper: float) -> float:
    """
    Lower bound on a frozen scorer's population AP, from per-threshold count endpoints.

    .. warning::
       This feature is experimental and may change or be removed in future releases.

    Threshold ``t`` reaches recall at least ``tp_lower[t] / mu_upper`` with precision at least
    ``tp_lower[t] / (tp_lower[t] + fp_upper[t])`` [2]_. The scorer's interpolated precision [1]_ at recall ``r`` is at
    least the best such precision among thresholds whose recall bound reaches ``r``; this staircase is integrated
    exactly.
    The counts may be in any common unit (per item or per group).

    Parameters
    ----------
    tp_lower : NDArray[np.float64]
        Shape ``(n_thresholds,)`` lower bound on true positives per unit at each threshold, at most ``mu_upper``.
    fp_upper : NDArray[np.float64]
        Shape ``(n_thresholds,)`` upper bound on false positives per unit at each threshold.
    mu_upper : float
        Upper bound on the prevalence ``mu`` (positives per unit), positive.

    Returns
    -------
    float
        AP lower bound in ``[0, 1]``.

    References
    ----------
    .. [1] Davis, J. and Goadrich, M. (2006). The relationship between precision-recall and ROC
       curves. Proceedings of the 23rd International Conference on Machine Learning, 233-240.
    .. [2] DFE manuscript, operating-point inequalities (lower witness).
    """
    tp = np.asarray(tp_lower, dtype=np.float64)
    fp = np.asarray(fp_upper, dtype=np.float64)
    if tp.ndim != 1 or tp.shape != fp.shape or not tp.size:
        raise ValueError("tp_lower and fp_upper must be aligned nonempty 1D arrays.")
    if not np.isfinite(np.r_[tp, fp, mu_upper]).all() or mu_upper <= 0:
        raise ValueError("tp_lower, fp_upper and mu_upper must be finite, with mu_upper positive.")
    if np.any(tp < 0) or np.any(fp < 0) or np.any(tp > mu_upper):
        raise ValueError("tp_lower and fp_upper must be non-negative, with tp_lower at most mu_upper.")
    recall = tp / mu_upper
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=tp + fp > 0)
    breaks = np.unique(np.r_[0.0, recall, 1.0])
    best = np.where(recall >= breaks[1:, None], precision, 0.0).max(axis=1)
    return float(np.diff(breaks) @ best)


@experimental
def witness_thresholds(  # noqa: C901
    fit_scores: NDArray[np.float64],
    fit_positive: NDArray[np.bool_],
    prevalence_fit: float,
    *,
    pos_below: float,
    quantiles: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Predeclared witness thresholds from the fit split only.

    .. warning::
       This feature is experimental and may change or be removed in future releases.

    For a rare class (``prevalence_fit < pos_below``) the quantiles of the fit-split positives' scores, so the
    thresholds spread over the recall range; otherwise (or when the fit split has no positives) the quantiles of
    all fit-split scores. Duplicates are removed.

    Parameters
    ----------
    fit_scores : NDArray[np.float64]
        Shape ``(n,)`` scorer outputs for the class on the fit split.
    fit_positive : NDArray[np.bool_]
        Shape ``(n,)`` whether each fit-split unit is a positive.
    prevalence_fit : float
        Fit-split prevalence of the class.
    pos_below : float
        Prevalence below which only positives' scores are used, in ``[0, 1]``.
    quantiles : NDArray[np.float64]
        Nonempty 1D quantile levels in ``[0, 1]``.

    Returns
    -------
    NDArray[np.float64]
        Sorted distinct thresholds (emit when ``score >= threshold``).

    Raises
    ------
    ValueError
        If ``fit_scores`` is not a nonempty 1D finite array, ``fit_positive`` is not aligned with it,
        ``prevalence_fit`` is not finite, or ``pos_below`` or a quantile is outside ``[0, 1]``.
    """
    scores = np.asarray(fit_scores, dtype=np.float64)
    positive = np.asarray(fit_positive, dtype=bool)
    q = np.asarray(quantiles, dtype=np.float64)
    if scores.ndim != 1 or not scores.size or positive.shape != scores.shape:
        raise ValueError("fit_scores and fit_positive must be aligned nonempty 1D arrays.")
    if not np.isfinite(scores).all() or not np.isfinite(prevalence_fit):
        raise ValueError("fit_scores and prevalence_fit must be finite.")
    if not 0 <= pos_below <= 1:
        raise ValueError(f"pos_below must be in [0, 1], got {pos_below}.")
    if q.ndim != 1 or not q.size or not ((q >= 0) & (q <= 1)).all():
        raise ValueError("quantiles must be a nonempty 1D array of levels in [0, 1].")
    if prevalence_fit < pos_below and positive.any():
        scores = scores[positive]
    return np.unique(np.quantile(scores, q))
