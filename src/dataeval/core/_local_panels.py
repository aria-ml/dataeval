"""
Label-blind cells and the post-stratified local-panel bound.

Audit units are grouped into cells fitted without labels; disjoint panels within each cell give
independent consensus values, and pooling the cell means with the audit's own cell weights
lower-bounds the risk.
"""

__all__ = []

import math
from collections.abc import Callable, Mapping
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans

from dataeval._experimental import experimental


class LocalPanelResult(TypedDict):
    """
    Type definition for a post-stratified local-panel lower bound.

    Attributes
    ----------
    bound : float
        Lower bound on the risk, floored at 0.
    within_radius : float
        Hoeffding radius of the within-cell step, at ``delta_within``.
    between_radius : float
        Hoeffding radius of the between-cell step, at ``delta_between``.
    n_units : int
        Number of audit units ``N``.
    n_panels : int
        Number of panels over all cells.
    panels_by_cell : dict[int, int]
        Number of panels ``k_h`` in each cell.
    deltas : tuple[float, float]
        The ledger: ``(delta_within, delta_between)``.
    value_max : float
        Upper bound of the panel values.
    """

    bound: float
    within_radius: float
    between_radius: float
    n_units: int
    n_panels: int
    panels_by_cell: dict[int, int]
    deltas: tuple[float, float]
    value_max: float


@experimental
def fit_partition(features_fit: NDArray, n_cells: int, seed: int) -> Callable[[NDArray], NDArray[np.intp]]:
    """
    Fit label-blind cells on the fit split and return the frozen cell assignment.

    .. warning::
       This feature is experimental and may change or be removed in future releases.

    Parameters
    ----------
    features_fit : NDArray
        Shape ``(n, d)`` features of the fit split only; labels never enter the fit.
    n_cells : int
        Number of cells.
    seed : int
        Random state of the k-means fit.

    Returns
    -------
    Callable[[NDArray], NDArray[np.intp]]
        Maps shape ``(n', d)`` features to their ``(n',)`` cell ids in ``[0, n_cells)``.
    """
    km = KMeans(n_clusters=n_cells, n_init=1, random_state=seed).fit(features_fit)  # type: ignore - n_init allows int but is typed as str
    return lambda features: np.asarray(km.predict(features), dtype=np.intp)


@experimental
def local_panel_bound(  # noqa: C901
    values_by_cell: Mapping[int, NDArray],
    n_by_cell: NDArray[np.intp],
    *,
    value_max: float,
    delta_within: float,
    delta_between: float,
    beta: float | NDArray | None = None,
) -> LocalPanelResult:
    """
    Post-stratified pooling of per-cell panel values into a risk lower bound.

    .. warning::
       This feature is experimental and may change or be removed in future releases.

    Returns ``sum_h (n_h/N) (gbar_h - beta_h) - r1 - r2``, floored at 0, where ``gbar_h`` is the
    mean of the ``k_h`` panel values of cell ``h`` (0 for a cell without panels) and ``beta_h`` is
    its approximation allowance, subtracted in every cell. The disjoint within-cell panels are the DFE's [2]_; this
    post-stratified pooling of them is DataEval's own construction (C3), proved in the Notes with Hoeffding [1]_
    twice.

    Parameters
    ----------
    values_by_cell : Mapping[int, NDArray]
        For each cell id in ``[0, H)``, the 1D values of its disjoint panels in ``[0, value_max]``.
        Missing or empty cells have no panels.
    n_by_cell : NDArray[np.intp]
        Shape ``(H,)`` number of audit units ``n_h`` in each cell.
    value_max : float
        Upper bound of the panel values. Values above ``value_max * (1 + 1e-12)`` raise; smaller
        excess (float rounding) is clipped.
    delta_within : float
        Failure probability of the within-cell step, in ``(0, 1)``.
    delta_between : float
        Failure probability of the between-cell step, in ``(0, 1)``.
    beta : float, NDArray or None, default None
        Non-negative allowance per cell (scalar or shape ``(H,)``); ``None`` means 0.

    Returns
    -------
    LocalPanelResult
        The bound, both radii and the ledger ``(delta_within, delta_between)``.

    Raises
    ------
    ValueError
        If a value is non-finite, negative or above ``value_max * (1 + 1e-12)``, if ``value_max`` is
        not positive and finite, if a delta is not in ``(0, 1)``, if ``n_by_cell`` has no units, or
        if ``beta`` is negative or non-finite.

    Notes
    -----
    Two Hoeffding [1]_ steps joined by a union bound, both valid for independent but not identically
    distributed terms. (1) Given the audit's cells the panel values of cell ``h`` are independent
    with mean ``g_h`` in ``[0, value_max]``, so ``sum_h (n_h/N) gbar_h`` is a weighted sum of
    independent bounded terms: it exceeds ``sum_h (n_h/N) g_h`` by more than
    ``r1 = value_max * sqrt(ln(1/delta_within) * sum_{k_h>0} n_h^2 / (N^2 k_h) / 2)`` with probability
    at most ``delta_within``; cells without panels drop out, which only lowers the left side since
    ``g_h >= 0``. (2) ``sum_h (n_h/N) (g_h - beta_h)`` is the mean over the ``N`` i.i.d. audit units of
    ``g - beta`` in ``[-beta_max, value_max - beta_min]``, so it exceeds its expectation by more than
    ``r2 = (value_max + beta_max - beta_min) * sqrt(ln(1/delta_between) / (2N))`` with probability at
    most ``delta_between``.

    References
    ----------
    .. [1] Hoeffding, W. (1963). Probability inequalities for sums of bounded random variables.
       Journal of the American Statistical Association, 58(301), 13-30.
    .. [2] DFE manuscript, local-pooling section (disjoint within-cell panels).

    Examples
    --------
    >>> import numpy as np
    >>> values = {0: np.array([0.0, 0.5, 0.5, 0.5]), 1: np.array([0.5, 0.5])}
    >>> r = local_panel_bound(values, np.array([600, 300, 100]), value_max=0.5, delta_within=0.025, delta_between=0.025)
    >>> round(r["bound"], 4)
    0.104
    """
    if not math.isfinite(value_max) or value_max <= 0:
        raise ValueError(f"value_max must be a positive finite number, got {value_max}.")
    for name, delta in (("delta_within", delta_within), ("delta_between", delta_between)):
        if not math.isfinite(delta) or not 0 < delta < 1:
            raise ValueError(f"{name} must be in (0, 1), got {delta}.")
    n = np.asarray(n_by_cell, dtype=np.float64)
    if n.ndim != 1 or n.size == 0 or n.min() < 0 or n.sum() <= 0:
        raise ValueError("n_by_cell must be a non-empty 1D array of non-negative counts with a positive total.")
    b = np.broadcast_to(np.asarray(0.0 if beta is None else beta, dtype=np.float64), n.shape)
    if not np.isfinite(b).all() or b.min() < 0:
        raise ValueError("beta must be finite and non-negative.")

    k = np.zeros(n.size, dtype=np.intp)
    gbar = np.zeros(n.size)
    for h in range(n.size):
        v = np.asarray(values_by_cell.get(h, ()), dtype=np.float64)
        if v.size == 0:
            continue
        if v.ndim != 1 or not np.isfinite(v).all() or v.min() < 0 or v.max() > value_max * (1 + 1e-12):
            raise ValueError(f"cell {h} values must be 1D, finite and within [0, value_max = {value_max}].")
        k[h], gbar[h] = v.size, np.minimum(v, value_max).mean()

    n_units = float(n.sum())
    has = k > 0
    r1 = value_max * math.sqrt(math.log(1 / delta_within) * float(np.sum(n[has] ** 2 / k[has])) / n_units**2 / 2)
    r2 = (value_max + float(b.max() - b.min())) * math.sqrt(math.log(1 / delta_between) / (2 * n_units))
    return LocalPanelResult(
        bound=max(0.0, float(n @ (gbar - b)) / n_units - r1 - r2),
        within_radius=r1,
        between_radius=r2,
        n_units=int(n_units),
        n_panels=int(k.sum()),
        panels_by_cell={h: int(k[h]) for h in range(n.size)},
        deltas=(float(delta_within), float(delta_between)),
        value_max=float(value_max),
    )
