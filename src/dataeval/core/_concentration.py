"""
Finite-sample confidence endpoints for the mean of a bounded variable.

The building blocks for evidence statements such as "with probability at least
``1 - delta`` the true rate is at least ``x``".
"""

__all__ = []

import math
from collections.abc import Callable
from typing import Literal, TypedDict

import numpy as np
from scipy.stats import beta

from dataeval._experimental import experimental
from dataeval.protocols import ArrayLike

_METHODS = ("hoeffding", "empirical_bernstein", "hybrid")
_CROSSING_METHODS = (*_METHODS, "exact_binomial")
_MAX_UNITS = 10**7


class EndpointResult(TypedDict):
    """
    Type definition for a one-sided confidence endpoint on a mean.

    Attributes
    ----------
    n : int
        Number of observations.
    mean : float
        Sample mean.
    variance : float | None
        Unbiased sample variance, or None when there is a single observation.
    radius_hoeffding : float
        Hoeffding radius at the tail budget used by ``method``.
    radius_bernstein : float | None
        Empirical Bernstein radius at the tail budget used by ``method``, or None when
        there is a single observation.
    endpoint : float
        The lower or upper endpoint, clipped to ``[0, support]``.
    method : str
        ``"hoeffding"``, ``"empirical_bernstein"``, ``"hybrid"`` or ``"exact_binomial"``.
    side : str
        ``"lower"`` or ``"upper"``.
    support : float
        Upper bound ``B`` of the observations, which lie in ``[0, B]``.
    delta : float
        Failure probability allocated to this endpoint.
    """

    n: int
    mean: float
    variance: float | None
    radius_hoeffding: float
    radius_bernstein: float | None
    endpoint: float
    method: str
    side: str
    support: float
    delta: float


def _check(
    support: float, delta: float, side: str = "lower", method: str = "hybrid", methods: tuple[str, ...] = _METHODS
) -> None:
    if side not in ("lower", "upper"):
        raise ValueError(f"side must be 'lower' or 'upper', got {side!r}.")
    if method not in methods:
        raise ValueError(f"method must be one of {methods}, got {method!r}.")
    if not math.isfinite(support) or support <= 0:
        raise ValueError(f"support must be a positive finite number, got {support}.")
    if not math.isfinite(delta) or not 0 < delta < 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}.")


def _check_crossing(mean: float, variance: float, support: float, delta: float, gap: float, method: str) -> None:
    _check(support, delta, method=method, methods=_CROSSING_METHODS)
    if not 0 <= mean <= support:
        raise ValueError(f"mean must lie in [0, {support}], got {mean}.")
    if not math.isfinite(variance) or variance < 0:
        raise ValueError(f"variance must be non-negative and finite, got {variance}.")
    if not math.isfinite(gap) or gap <= 0:
        raise ValueError(f"gap must be positive and finite, got {gap}.")


def _radii(n: int, variance: float | None, support: float, delta: float, method: str) -> tuple[float, float | None]:
    # Hybrid splits delta between the two bounds (union bound) and takes the smaller radius.
    d = delta / 2 if method == "hybrid" else delta
    hoeffding = support * math.sqrt(math.log(1 / d) / (2 * n))
    if variance is None or n < 2:
        return hoeffding, None
    # Maurer & Pontil (2009), Theorem 4.
    bernstein = math.sqrt(2 * variance * math.log(2 / d) / n) + 7 * support * math.log(2 / d) / (3 * (n - 1))
    return hoeffding, bernstein


def _radius(n: int, variance: float | None, support: float, delta: float, method: str) -> float:
    hoeffding, bernstein = _radii(n, variance, support, delta, method)
    fallback = support if bernstein is None else bernstein  # a single sample says nothing beyond the support
    return {"hoeffding": hoeffding, "empirical_bernstein": fallback}.get(method, min(hoeffding, fallback))


@experimental
def mean_endpoint(
    values: ArrayLike,
    support: float,
    delta: float,
    *,
    side: Literal["lower", "upper"] = "lower",
    method: Literal["hoeffding", "empirical_bernstein", "hybrid"] = "hybrid",
) -> EndpointResult:
    """
    One-sided finite-sample confidence endpoint for the mean of i.i.d. bounded observations.

    .. warning::
       This feature is experimental and may change or be removed in future releases.

    With probability at least ``1 - delta`` the true mean is at least the ``"lower"`` endpoint
    (or at most the ``"upper"`` endpoint). The ``"hybrid"`` method applies Hoeffding [1]_ and empirical
    Bernstein [2]_ each at ``delta / 2`` and uses the smaller radius. The hybrid and empirical Bernstein
    methods assume i.i.d. observations; ``method="hoeffding"`` remains valid for independent but not
    identically distributed observations bounded in ``[0, support]``.

    Parameters
    ----------
    values : ArrayLike
        1D array of i.i.d. observations in ``[0, support]``.
    support : float
        Upper bound ``B > 0`` on the observations.
    delta : float
        Failure probability of this one endpoint, in ``(0, 1)``.
    side : "lower" or "upper", default "lower"
        Which endpoint to compute.
    method : "hoeffding", "empirical_bernstein" or "hybrid", default "hybrid"
        Concentration inequality used for the radius.

    Returns
    -------
    EndpointResult
        The endpoint, with the sample statistics and radii behind it.

    Raises
    ------
    ValueError
        If ``values`` is empty, not 1D, non-finite or outside ``[0, support]``, if ``support <= 0``,
        if ``delta`` is not in ``(0, 1)``, or if ``side`` or ``method`` is unknown.

    Notes
    -----
    Hoeffding radius ``B * sqrt(ln(1/d) / (2n))``; empirical Bernstein radius
    ``sqrt(2 S^2 ln(2/d) / n) + 7 B ln(2/d) / (3 (n - 1))`` where ``S^2`` is the unbiased sample
    variance; ``d`` is ``delta`` (``delta / 2`` for ``"hybrid"``). With a single observation only
    the Hoeffding radius exists and the empirical Bernstein method falls back to ``support``.

    References
    ----------
    .. [1] Hoeffding, W. (1963). Probability inequalities for sums of bounded random variables.
       Journal of the American Statistical Association, 58(301), 13-30.
    .. [2] Maurer, A., & Pontil, M. (2009). Empirical Bernstein bounds and sample variance
       penalization. COLT 2009, Theorem 4.

    Examples
    --------
    >>> import numpy as np
    >>> r = mean_endpoint(np.ones(400), 1.0, 0.05)
    >>> round(r["endpoint"], 3)
    0.974
    """
    _check(support, delta, side, method)
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"values must be 1D, got {x.ndim}D.")
    if x.size == 0:
        raise ValueError("values must be non-empty.")
    if not np.isfinite(x).all():
        raise ValueError("values must be finite.")
    if x.min() < 0 or x.max() > support:
        raise ValueError(f"values outside the declared support [0, {support}].")

    n, mean = int(x.size), float(x.mean())
    variance = float(x.var(ddof=1)) if n >= 2 else None
    hoeffding, bernstein = _radii(n, variance, support, delta, method)
    radius = _radius(n, variance, support, delta, method)
    endpoint = max(0.0, mean - radius) if side == "lower" else min(support, mean + radius)
    return EndpointResult(
        n=n,
        mean=mean,
        variance=variance,
        radius_hoeffding=hoeffding,
        radius_bernstein=bernstein,
        endpoint=endpoint,
        method=method,
        side=side,
        support=float(support),
        delta=float(delta),
    )


def _projected_bound(p: float, n: int, delta: float, side: str) -> float:
    """Clopper-Pearson bound at the real-valued count ``k = p * n``, via the beta-quantile form."""
    k = p * n
    if side == "lower":
        return 0.0 if k <= 0 else float(beta.ppf(delta, k, n - k + 1))
    return 1.0 if p >= 1 else float(beta.ppf(1 - delta, k + 1, n - k))


def _is_count(x: object) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


@experimental
def binomial_endpoint(
    successes: int,
    trials: int,
    delta: float,
    *,
    side: Literal["lower", "upper"] = "lower",
) -> EndpointResult:
    """
    One-sided exact (Clopper-Pearson) confidence endpoint for a binomial proportion.

    .. warning::
       This feature is experimental and may change or be removed in future releases.

    With probability at least ``1 - delta`` the true success probability is at least the ``"lower"``
    endpoint (or at most the ``"upper"`` endpoint) of ``trials`` i.i.d. Bernoulli trials [1]_. The bounds
    are quantiles of the beta distribution, so they are exact rather than asymptotic and are tighter
    than Hoeffding's at the extremes.

    Parameters
    ----------
    successes : int
        Number of successes, in ``[0, trials]``.
    trials : int
        Number of i.i.d. Bernoulli trials; at least 1.
    delta : float
        Failure probability of this one endpoint, in ``(0, 1)``.
    side : "lower" or "upper", default "lower"
        Which endpoint to compute.

    Returns
    -------
    EndpointResult
        The endpoint with ``method="exact_binomial"`` and ``support=1.0``. ``variance`` is the unbiased
        sample variance (None for a single trial), ``radius_hoeffding`` is given for reference only and
        ``radius_bernstein`` is None.

    Raises
    ------
    ValueError
        If the counts are not integers with ``0 <= successes <= trials`` and ``trials >= 1``, if ``delta``
        is not in ``(0, 1)``, or if ``side`` is unknown.

    Notes
    -----
    Lower endpoint ``Beta(delta; k, n - k + 1)`` (0 when ``k = 0``); upper endpoint
    ``Beta(1 - delta; k + 1, n - k)`` (1 when ``k = n``), for ``k`` successes in ``n`` trials.

    References
    ----------
    .. [1] Clopper, C. J., & Pearson, E. S. (1934). The use of confidence or fiducial limits illustrated
       in the case of the binomial. Biometrika, 26(4), 404-413.

    Examples
    --------
    >>> r = binomial_endpoint(400, 400, 0.05)
    >>> round(r["endpoint"], 3)
    0.993
    """
    _check(1.0, delta, side)
    if not (_is_count(successes) and _is_count(trials)):
        raise ValueError(f"successes and trials must be integers, got {successes!r} and {trials!r}.")
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError(f"need trials >= 1 and 0 <= successes <= trials, got {successes} of {trials}.")
    k, n = int(successes), int(trials)
    return EndpointResult(
        n=n,
        mean=k / n,
        variance=k * (n - k) / (n * (n - 1)) if n >= 2 else None,
        radius_hoeffding=math.sqrt(math.log(1 / delta) / (2 * n)),
        radius_bernstein=None,
        endpoint=_projected_bound(k / n, n, delta, side),
        method="exact_binomial",
        side=side,
        support=1.0,
        delta=float(delta),
    )


@experimental
def ratio_endpoint(
    numerators: ArrayLike,
    denominators: ArrayLike,
    num_support: float,
    den_support: float,
    delta: float,
    *,
    side: Literal["lower", "upper"] = "lower",
    method: Literal["hoeffding", "empirical_bernstein", "hybrid"] = "hybrid",
) -> float:
    """
    One-sided confidence endpoint for the ratio ``E[numerator] / E[denominator]``.

    .. warning::
       This feature is experimental and may change or be removed in future releases.

    The ``"lower"`` endpoint divides a lower endpoint of the numerator mean by an upper endpoint of the
    denominator mean; the ``"upper"`` endpoint divides an upper endpoint of the numerator mean by a lower
    endpoint of the denominator mean. Each uses ``delta / 2``, so by a union bound the ratio endpoint
    holds with probability at least ``1 - delta``. Each mean endpoint uses Hoeffding [1]_, empirical
    Bernstein [2]_ or the smaller of the two, as ``method`` selects.

    Parameters
    ----------
    numerators : ArrayLike
        1D array of i.i.d. numerator observations in ``[0, num_support]``.
    denominators : ArrayLike
        1D array of the paired denominator observations in ``[0, den_support]``.
    num_support : float
        Upper bound on the numerator observations.
    den_support : float
        Upper bound on the denominator observations.
    delta : float
        Failure probability of the ratio endpoint, in ``(0, 1)``.
    side : "lower" or "upper", default "lower"
        Which endpoint to compute.
    method : "hoeffding", "empirical_bernstein" or "hybrid", default "hybrid"
        Concentration inequality used for each mean.

    Returns
    -------
    float
        The ratio endpoint. The ``"upper"`` endpoint is ``inf`` when the denominator's lower endpoint is zero.

    Raises
    ------
    ValueError
        If the arrays differ in length or either mean endpoint rejects its inputs.

    References
    ----------
    .. [1] Hoeffding, W. (1963). Probability inequalities for sums of bounded random variables.
       Journal of the American Statistical Association, 58(301), 13-30.
    .. [2] Maurer, A., & Pontil, M. (2009). Empirical Bernstein bounds and sample variance
       penalization. COLT 2009, Theorem 4.
    """
    _check(num_support, delta, side, method)
    _check(den_support, delta, side, method)
    num = np.asarray(numerators, dtype=np.float64)
    den = np.asarray(denominators, dtype=np.float64)
    if num.shape != den.shape:
        raise ValueError(f"numerators and denominators must have the same length, got {num.shape} and {den.shape}.")
    other = "upper" if side == "lower" else "lower"
    top = mean_endpoint(num, num_support, delta / 2, side=side, method=method)["endpoint"]
    bottom = mean_endpoint(den, den_support, delta / 2, side=other, method=method)["endpoint"]
    if bottom == 0:
        return 0.0 if top == 0 else float("inf")
    return top / bottom


def _first_true(pred: Callable[[int], bool], lo: int, hi: int) -> int:
    """Smallest n in [lo, hi] with ``pred(n)``, for a predicate that is monotone and true at ``hi``."""
    while lo < hi:
        mid = (lo + hi) // 2
        if pred(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def _units_to_cross_binomial(mean: float, support: float, delta: float, gap: float, side: str) -> int | None:
    _check(support, delta, side)
    if support != 1:
        raise ValueError(f"exact_binomial requires support == 1, got {support}.")
    target = mean - gap if side == "lower" else mean + gap

    def clears(n: int) -> bool:
        bound = _projected_bound(mean, n, delta, side)
        return bound > target if side == "lower" else bound < target

    return _first_true(clears, 1, _MAX_UNITS) if clears(_MAX_UNITS) else None


@experimental
def units_to_cross(
    mean: float,
    variance: float,
    support: float,
    delta: float,
    gap: float,
    *,
    method: Literal["hoeffding", "empirical_bernstein", "hybrid", "exact_binomial"] = "hybrid",
    side: Literal["lower", "upper"] = "lower",
) -> int | None:
    """
    Smallest number of observations whose confidence radius falls below a gap.

    .. warning::
       This feature is experimental and may change or be removed in future releases.

    Answers "how much more data would it take for the endpoint to clear the gap" assuming the
    observed mean and variance would persist. The radius is Hoeffding's [1]_, the empirical
    Bernstein one [2]_ or the smaller of the two, as ``method`` selects.

    ``method="exact_binomial"`` inverts the Clopper-Pearson bound [3]_ instead, which is asymmetric: the
    result is the smallest ``n`` whose ``side`` endpoint, evaluated at the real-valued count
    ``k = mean * n``, lies strictly beyond ``mean - gap`` (``"lower"``) or ``mean + gap`` (``"upper"``).
    It requires ``support == 1`` and ignores ``variance``. The bound moves monotonically toward ``mean``
    as ``n`` grows, which is checked by the tests, so a binary search finds the smallest ``n``.

    Parameters
    ----------
    mean : float
        Observed mean, in ``[0, support]``.
    variance : float
        Observed unbiased sample variance; unused by ``"exact_binomial"``.
    support : float
        Upper bound ``B > 0`` on the observations.
    delta : float
        Failure probability of the endpoint, in ``(0, 1)``.
    gap : float
        Target radius; must be positive.
    method : "hoeffding", "empirical_bernstein", "hybrid" or "exact_binomial", default "hybrid"
        Concentration inequality used for the radius, or the exact binomial bound.
    side : "lower" or "upper", default "lower"
        Which exact binomial endpoint to invert; ignored by the other methods.

    Returns
    -------
    int | None
        Smallest ``n`` whose radius is below ``gap`` (or whose exact endpoint clears it), or None if no
        ``n`` up to ``10**7`` does.

    Raises
    ------
    ValueError
        If ``mean`` is outside ``[0, support]``, ``variance < 0``, ``gap <= 0``, ``method="exact_binomial"``
        with ``support != 1``, or the other arguments are invalid.

    References
    ----------
    .. [1] Hoeffding, W. (1963). Probability inequalities for sums of bounded random variables.
       Journal of the American Statistical Association, 58(301), 13-30.
    .. [2] Maurer, A., & Pontil, M. (2009). Empirical Bernstein bounds and sample variance
       penalization. COLT 2009, Theorem 4.
    .. [3] Clopper, C. J., & Pearson, E. S. (1934). The use of confidence or fiducial limits illustrated
       in the case of the binomial. Biometrika, 26(4), 404-413.
    """
    _check_crossing(mean, variance, support, delta, gap, method)
    if method == "exact_binomial":
        return _units_to_cross_binomial(mean, support, delta, gap, side)

    def below(n: int) -> bool:
        return _radius(n, variance, support, delta, method) < gap

    # n = 1 has no variance term, so the radius is not monotone between 1 and 2; search from 2.
    if below(1):
        return 1
    if not below(_MAX_UNITS):
        return None
    return _first_true(below, 2, _MAX_UNITS)
