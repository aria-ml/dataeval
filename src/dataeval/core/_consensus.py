"""
Consensus values and disjoint panels.

A *panel* is ``m`` independent labels (replicas) of one unit. A consensus value scores how much
the panel disagrees, and its expectation lower-bounds the corresponding risk (DFE consensus
inequality). These are the per-panel values that ceilings are built from.
"""

__all__ = []

import math

import numpy as np
from numpy.typing import NDArray

from dataeval._experimental import experimental


def _check_m(m: int, minimum: int = 1) -> None:
    if not isinstance(m, (int, np.integer)) or m < minimum:
        raise ValueError(f"m must be an integer >= {minimum}, got {m}.")


def _check_cost(cost: float) -> None:
    if not math.isfinite(cost) or cost < 0:
        raise ValueError(f"cost must be finite and non-negative, got {cost}.")


@experimental
def consensus_max(n_classes: int, m: int) -> float:
    """
    Tight maximum of the 0-1 consensus value ``1 - maxcount / m``.

    .. warning::
       This feature is experimental and may change or be removed in future releases.

    Among ``m`` labels over ``n_classes`` classes the most common class holds at least
    ``ceil(m / n_classes)`` of them, so the 0-1 consensus value [1]_ is at most
    ``1 - ceil(m / n_classes) / m``.

    Parameters
    ----------
    n_classes : int
        Number of classes, at least 2.
    m : int
        Panel size, at least 1.

    Returns
    -------
    float
        The attainable maximum of the 0-1 consensus value.

    References
    ----------
    .. [1] DFE manuscript, replicated-consensus section (consensus inequality).
    """
    if n_classes < 2:
        raise ValueError(f"n_classes must be at least 2, got {n_classes}.")
    _check_m(m)
    return 1.0 - math.ceil(m / n_classes) / m


@experimental
def cost_consensus_max(cost: float, m: int) -> float:
    """
    Tight maximum of the one-vs-rest cost-weighted consensus value.

    .. warning::
       This feature is experimental and may change or be removed in future releases.

    Returns ``max_{k=0..m} min(k / m, cost * (1 - k / m))``, the largest one-vs-rest consensus
    value [1]_ of a cost-weighted risk [2]_, where ``k`` is the number of positive labels in a
    panel of ``m``.

    Parameters
    ----------
    cost : float
        Cost ``lambda >= 0`` of a false alarm relative to a missed positive.
    m : int
        Panel size, at least 1.

    Returns
    -------
    float
        The attainable maximum of the cost-weighted consensus value.

    References
    ----------
    .. [1] DFE manuscript, replicated-consensus section (consensus inequality).
    .. [2] Reid, M. D. and Williamson, R. C. (2011). Information, divergence and risk for
       binary experiments. Journal of Machine Learning Research, 12, 731-817.
    """
    _check_cost(cost)
    _check_m(m)
    k = np.arange(m + 1) / m
    return float(np.minimum(k, cost * (1 - k)).max())


@experimental
def kink_costs(m: int) -> NDArray[np.float64]:
    """
    Costs at which the cost-weighted consensus value changes slope.

    .. warning::
       This feature is experimental and may change or be removed in future releases.

    The break-even point ``k / m = cost / (1 + cost)`` gives ``cost = k / (m - k)``.

    Parameters
    ----------
    m : int
        Panel size, at least 1.

    Returns
    -------
    NDArray[np.float64]
        ``[0, 1/(m-1), ..., (m-1)/1]`` (just ``[0]`` when ``m == 1``).
    """
    _check_m(m)
    k = np.arange(1, m, dtype=np.float64)
    return np.r_[0.0, k / (m - k)]


@experimental
def zero_one_consensus(labels: NDArray[np.intp], n_classes: int) -> NDArray[np.float64]:
    """
    0-1 consensus value ``1 - maxcount / m`` of each panel.

    .. warning::
       This feature is experimental and may change or be removed in future releases.

    Its expectation lower-bounds the 0-1 risk of the best answer for the panel's label
    distribution (the consensus inequality [1]_).

    Parameters
    ----------
    labels : NDArray[np.intp]
        Shape ``(n, m)`` panels of class labels in ``[0, n_classes)``.
    n_classes : int
        Number of classes, at least 2.

    Returns
    -------
    NDArray[np.float64]
        Shape ``(n,)`` consensus value of each panel.

    References
    ----------
    .. [1] DFE manuscript, replicated-consensus section (consensus inequality).
    """
    if n_classes < 2:
        raise ValueError(f"n_classes must be at least 2, got {n_classes}.")
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.shape[1] < 1:
        raise ValueError(f"labels must have shape (n, m) with m >= 1, got {labels.shape}.")
    if labels.size and (labels.min() < 0 or labels.max() >= n_classes):
        raise ValueError(f"labels must lie in [0, {n_classes}).")
    counts = np.eye(n_classes, dtype=np.intp)[labels].sum(axis=1)
    return 1.0 - counts.max(axis=1) / labels.shape[1]


@experimental
def cost_consensus(positives: NDArray[np.intp], m: int, cost: float) -> NDArray[np.float64]:
    """
    One-vs-rest cost-weighted consensus value ``min(k / m, cost * (1 - k / m))`` of each panel.

    .. warning::
       This feature is experimental and may change or be removed in future releases.

    Its expectation lower-bounds the cost-weighted one-vs-rest risk [2]_ of the best answer for
    the panel's label distribution (the consensus inequality [1]_).

    Parameters
    ----------
    positives : NDArray[np.intp]
        Shape ``(n,)`` number ``k`` of positive labels in each panel, in ``[0, m]``.
    m : int
        Panel size, at least 1.
    cost : float
        Cost ``lambda >= 0`` of a false alarm relative to a missed positive.

    Returns
    -------
    NDArray[np.float64]
        Shape ``(n,)`` consensus value of each panel.

    References
    ----------
    .. [1] DFE manuscript, replicated-consensus section (consensus inequality).
    .. [2] Reid, M. D. and Williamson, R. C. (2011). Information, divergence and risk for
       binary experiments. Journal of Machine Learning Research, 12, 731-817.
    """
    _check_cost(cost)
    _check_m(m)
    positives = np.asarray(positives)
    if positives.ndim != 1:
        raise ValueError(f"positives must be 1D, got shape {positives.shape}.")
    if positives.size and (positives.min() < 0 or positives.max() > m):
        raise ValueError(f"positives must lie in [0, {m}].")
    k = positives / m
    return np.minimum(k, cost * (1 - k))


@experimental
def disjoint_panels(cells: NDArray[np.intp], m: int, rng: np.random.Generator) -> dict[int, NDArray[np.intp]]:
    """
    Random disjoint panels of ``m`` units within each cell.

    .. warning::
       This feature is experimental and may change or be removed in future releases.

    Only one integer seed is drawn from ``rng``, and it is drawn first, so the panels depend on
    ``rng`` only through its state at the call. Units left over in a cell (fewer than ``m``) are unused.

    Parameters
    ----------
    cells : NDArray[np.intp]
        Shape ``(n,)`` integer cell id of each unit.
    m : int
        Panel size, at least 2.
    rng : np.random.Generator
        Source of the panel seed.

    Returns
    -------
    dict[int, NDArray[np.intp]]
        For each cell id, shape ``(k_h, m)`` unit indices; ``k_h`` is ``0`` for a cell with fewer than ``m`` units.
    """
    _check_m(m, 2)
    cells = np.asarray(cells)
    if cells.ndim != 1 or not np.issubdtype(cells.dtype, np.integer):
        raise ValueError("cells must be a 1D integer array.")
    local = np.random.default_rng(int(rng.integers(2**32)))
    panels: dict[int, NDArray[np.intp]] = {}
    for cell in np.unique(cells):
        ix = local.permutation(np.flatnonzero(cells == cell))
        k = len(ix) // m
        panels[int(cell)] = ix[: k * m].reshape(k, m)
    return panels
