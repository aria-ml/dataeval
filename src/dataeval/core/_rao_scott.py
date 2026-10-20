"""
Cluster-robust statistics for one factor-by-class contingency table.

Rows (detections, frames) are clustered within groups (images, sequences), so a row-level
chi-square test overstates the evidence. :func:`grouped_table` tests label-proportion
homogeneity with the second-order Rao-Scott F (Rao & Scott 1984; Thomas & Rao 1987),
taking its denominator degrees of freedom from the Kish (1965) effective number of groups
behind the thinnest factor level. Each cell gets an exact, two-sided Korn & Graubard (1998)
binomial test on its effective counts. A cell's variance is never taken below a Kish
cluster-model variance, built from each group's size and a within-group correlation taken
from the class at other levels or from the level's other classes. Where those reflect the
cell's bursts, a cell holding few of them is not also read as precise; where a class bursts
at one level only, they cannot (see :func:`_pooled_cell_variance`).

References
----------
Rao, J. N. K., & Scott, A. J. (1984). On chi-squared tests for multiway contingency tables
with cell proportions estimated from survey data. Annals of Statistics, 12(1), 46-60.

Thomas, D. R., & Rao, J. N. K. (1987). Small-sample comparisons of level and power for
simple goodness-of-fit statistics under cluster sampling. JASA, 82(398), 630-636.

Korn, E. L., & Graubard, B. I. (1998). Confidence intervals for proportions with small
expected number of positive counts estimated from survey data. Survey Methodology, 24(2),
193-201.

Kish, L. (1965). Survey Sampling. Wiley.
"""

__all__ = []

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.special import betainc
from scipy.stats import f as f_dist
from scipy.stats import norm
from scipy.stats import t as t_dist

from dataeval._log import get_logger
from dataeval.core._association import bias_corrected_cramers_v, cochran_breach

_logger = get_logger(__name__)

# Relative tolerance below which a variance, or the trace of the design-effect matrix, is zero.
TOLERANCE = 1e-12
# Largest factor-by-class table tested: the dense covariance costs 8 * (r*K)**2 bytes.
MAX_GROUPED_CELLS = 10_000
# A cell is sparse below this effective expected count, or below this many design df.
MIN_SPARSE_EXPECTED = 5.0
MIN_SPARSE_DF = 4

_Z_975 = float(norm.ppf(0.975))


class GroupedTable(NamedTuple):
    """
    Cluster-robust statistics for one factor-by-class table.

    When the table is untestable, every float is NaN, ``insufficient`` is True,
    ``cell_p_values`` is all NaN and ``cell_sparse`` is all True.

    Attributes
    ----------
    observed : NDArray[np.float64]
        Row counts per (level, class), shape (r, K).
    expected : NDArray[np.float64]
        Counts expected under homogeneity, shape (r, K).
    chi_squared : float
        Pearson's X², not divided by the design effect.
    design_effect : float
        Mean generalized design effect, tr(GV)/df. Not floored: below 1 it signals
        negative within-group dependence.
    df_num : float
        Numerator degrees of freedom of the F reference, df/(1+a²).
    df_den : float
        Denominator degrees of freedom, ``df_num`` times the design df.
    p_value : float
        Survival function of the second-order Rao-Scott F.
    effective_groups : float
        Kish effective number of groups of the thinnest factor level, and of the thinnest
        class when the class is constant within groups. The design df is one less.
    score : float
        Bias-corrected Cramér's V, its null mean matched to the design effect.
    insufficient : bool
        Cochran's criterion fails on the expected counts divided by the design effect, or
        the design df is below ``MIN_SPARSE_DF``. A screen, not a validity guarantee.
    cell_p_values : NDArray[np.float64]
        Two-sided Korn-Graubard p-value per cell, shape (r, K). NaN where the cell's
        variance is zero, or where its class occurs in a single group.
    cell_sparse : NDArray[np.bool_]
        Whether each cell's effective expected count is below ``MIN_SPARSE_EXPECTED``, or
        its design df below ``MIN_SPARSE_DF``.
    testable : bool
        Whether the factor could be tested at all.
    """

    observed: NDArray[np.float64]
    expected: NDArray[np.float64]
    chi_squared: float
    design_effect: float
    df_num: float
    df_den: float
    p_value: float
    effective_groups: float
    score: float
    insufficient: bool
    cell_p_values: NDArray[np.float64]
    cell_sparse: NDArray[np.bool_]
    testable: bool


def _untestable(observed: NDArray[np.float64], expected: NDArray[np.float64]) -> GroupedTable:
    nan = float("nan")
    return GroupedTable(
        observed,
        expected,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan,
        True,
        np.full(observed.shape, np.nan),
        np.ones(observed.shape, dtype=np.bool_),
        False,
    )


def _sandwich(
    cells: NDArray[np.intp],
    group_index: NDArray[np.intp],
    n_groups: int,
    p_f: NDArray[np.float64],
    p_y: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Cluster sandwich covariance of the table's deviations from homogeneity.

    Each group's score is ``U_s = N_s - X_s p_yᵀ - p_f Y_sᵀ + m_s p_f p_yᵀ``, which is
    ``vec(N_s)`` times ``P_f ⊗ P_y`` with ``P = I - 1 pᵀ``. So ``V = S/(S-1) UᵀU`` is
    ``NᵀN`` centered along each of its four (level, class, level, class) axes. The
    group-by-cell counts ``N`` stay sparse; no dense S×rK matrix is built.

    Parameters
    ----------
    cells : NDArray[np.intp]
        Flat cell index per row, ``level * K + class``.
    group_index : NDArray[np.intp]
        Dense group index per row, ``0..n_groups-1``.
    n_groups : int
        Number of groups S.
    p_f, p_y : NDArray[np.float64]
        Level and class proportions over all rows.

    Returns
    -------
    NDArray[np.float64]
        The (r*K, r*K) covariance V.
    """
    r, k = p_f.size, p_y.size
    counts = sparse.csr_array((np.ones(cells.size), (group_index, cells)), shape=(n_groups, r * k))
    v = (counts.T @ counts).toarray().reshape(r, k, r, k)
    for axis, p in enumerate((p_f, p_y, p_f, p_y)):
        shape = [1, 1, 1, 1]
        shape[axis] = -1
        # ponytail: the product is one (rK)² temporary; block it if MAX_GROUPED_CELLS grows.
        v -= p.reshape(shape) * v.sum(axis=axis, keepdims=True)
    v *= n_groups / (n_groups - 1)
    return v.reshape(r * k, r * k)


def _group_counts(
    codes: NDArray[np.intp], group_index: NDArray[np.intp], size: int
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64]]:
    """Count the rows of each (group, code) pair that occurs; return its group, code and count."""
    pairs, counts = np.unique(group_index * size + codes, return_counts=True)
    return pairs // size, pairs % size, counts.astype(np.float64)


def _kish(code: NDArray[np.intp], counts: NDArray[np.float64], size: int) -> NDArray[np.float64]:
    """Kish effective number of groups per code, from the rows of each (group, code) pair."""
    return np.bincount(code, counts, size) ** 2 / np.bincount(code, counts**2, size)


def _correlation(excess: NDArray[np.float64], pairs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Moment estimate of a within-group correlation, clipped to [0, 1]; 0 where no pairs."""
    return np.clip(np.divide(excess, pairs, out=np.zeros_like(excess), where=pairs > 0), 0.0, 1.0)


def _max_over_others(x: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
    """Each entry's maximum of ``x`` along ``axis`` over the other entries (at least two along it)."""
    top2 = np.sort(x, axis=axis).take([-1, -2], axis=axis)
    first, second = top2.take([0], axis=axis), top2.take([1], axis=axis)
    return np.where(x == first, second, first)


def _pooled_cell_variance(
    var_cell: NDArray[np.float64],
    levels: tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64]],
    classes: tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64]],
    sizes: NDArray[np.float64],
    p_y: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Each cell's between-group variance under a design-effect model fitted away from the cell.

    A cell's sandwich variance comes from the groups that fill it, so a cell that happens
    to hold few of its class's bursts is both low and low-variance, and its test is
    anti-conservative. Between groups, a cell deviates by ``Σ_s (a_si - p_i) h_sj``, with
    level shares ``a_si = X_si/m_s`` and class deviations ``h_sj = Y_sj - m_s p_j``
    (exactly, for a factor constant within groups), so its variance is
    ``Σ_s (a_si - p_i)² E[h_sj²]``. Here ``E[h_sj²]`` is the Kish (1965) cluster model
    ``m_s q_j (1 + (m_s - 1) ρ)`` with ``q_j = p_j (1 - p_j)``: it takes the group's size,
    not its count of the class. For a group at level ``i'``, ``ρ`` is the largest of four
    moment estimates of the class's within-group correlation:

    - the class's correlation pooled over the groups outside level ``i'``;
    - the largest of the class's correlations at the other levels one by one, so a class
      that bursts at some levels only (scene context) keeps its bursts' variance;
    - level ``i'``'s correlation pooled over the other classes, so a level whose groups
      burst keeps that variance;
    - the largest of level ``i'``'s correlations in the other classes one by one, so a
      level that bursts in a few classes keeps their variance at every class.

    The first two never read level ``i'``. The last two read level ``i'`` but not class
    ``j``. With few classes the other classes' counts mirror class ``j``'s (with two,
    exactly), so they can raise the floor but give no protection when the cell falls
    short. A cell's own groups use their level's estimates; groups at other levels enter
    with weight ``p_i²``, and their estimates do read the cell's level. The maxima over
    levels and over classes are taken over noisy estimates, which widens every cell's
    reference when each level holds few bursts.

    The model assumes a class's within-group correlation does not depend on group size,
    and that for each level and class one of the four estimates is at least the cell's
    own correlation. That fails when a class bursts at one level only and that level's
    other classes do not burst: the cell's own groups are then the only evidence of its
    variance, and a cell that by chance holds none of its bursts cannot be told from a
    real shortage.

    Parameters
    ----------
    var_cell : NDArray[np.float64]
        Sandwich variance per cell, shape (r, K); sets the table's shape.
    levels, classes : tuple of NDArray
        :func:`_group_counts` of the factor and of the class.
    sizes : NDArray[np.float64]
        Rows per group, m_s.
    p_y : NDArray[np.float64]
        Class proportions over all rows.

    Returns
    -------
    NDArray[np.float64]
        The pooled variance per cell, shape (r, K).
    """
    (r, k), n_groups = var_cell.shape, sizes.size
    group, level, count = levels
    share = count / sizes[group]
    p_f = np.bincount(level, count, r) / sizes.sum()
    q = p_y * (1 - p_y)
    c_group, cls, c_count = classes
    squares, pairs = sizes**2, sizes * (sizes - 1)
    # Σ_s a_si h_sj²: h² on the (group, class) pairs that occur, plus p_j² m_s² for every pair
    a = sparse.csr_array((share, (group, level)), shape=(n_groups, r))
    h2 = sparse.csr_array((c_count * (c_count - 2 * p_y[cls] * sizes[c_group]), (c_group, cls)), shape=(n_groups, k))
    in_level = (a.T @ h2).toarray() + np.outer(np.bincount(level, share * squares[group], r), p_y**2)
    total = in_level.sum(0)
    rows_in, pairs_in = np.bincount(level, share * sizes[group], r), np.bincount(level, share * pairs[group], r)
    rows_out, pairs_out = float(sizes.sum()) - rows_in, float(pairs.sum()) - pairs_in
    q_other = q.sum() - q
    own = _correlation(in_level - np.outer(rows_in, q), np.outer(pairs_in, q))
    rho = np.maximum.reduce([
        _correlation(total - in_level - np.outer(rows_out, q), np.outer(pairs_out, q)),
        _max_over_others(own, axis=0),
        _correlation(
            in_level.sum(1, keepdims=True) - in_level - np.outer(rows_in, q_other), np.outer(pairs_in, q_other)
        ),
        _max_over_others(own, axis=1),
    ])
    # Σ_s (a_si - p_i)² m_s q_j (1 + (m_s - 1) Σ_i' a_si' ρ_i'j), expanded so only r×r level products are formed
    spread = np.bincount(level, share**2 * sizes[group], r) - 2 * p_f * rows_in + p_f**2 * sizes.sum()
    weighted = sparse.csr_array((share * pairs[group], (group, level)), shape=(n_groups, r))
    weighted_sq = sparse.csr_array((share**2 * pairs[group], (group, level)), shape=(n_groups, r))
    clustered = (
        (weighted_sq.T @ a) @ rho - 2 * p_f[:, None] * ((weighted.T @ a) @ rho) + np.outer(p_f**2, pairs_in @ rho)
    )
    return q * (spread[:, None] + clustered)


def grouped_table(
    factor_codes: NDArray[np.intp],
    class_codes: NDArray[np.intp],
    groups: NDArray[np.intp],
) -> GroupedTable:
    """
    Test one factor against the class axis when rows are clustered within groups.

    Groups are independent of each other; rows inside a group may depend on each other in
    any way. The null hypothesis is label-proportion homogeneity: every factor level has
    the same class distribution.

    The factor test is the second-order Rao-Scott F. With the cluster sandwich V of the
    deviations ``O - E`` and ``G = diag(1/E)``, the mean design effect is
    ``δ̄ = tr(GV)/df``, ``1 + a² = tr((GV)²)/(df·δ̄²)`` and ``ν = df/(1 + a²)``. The p-value
    is the survival function of ``F(ν, ν·d)`` at ``X²/(δ̄·df)``, where the design df ``d`` is
    the smallest Kish effective number of groups over the factor levels, less one. When
    the class is constant within groups, the classes' effective groups enter that minimum.

    Each cell is tested with a Korn-Graubard binomial test of ``O/D`` successes in
    ``n_i/D`` trials at the class proportion, where ``D`` is the cell's design effect
    (floored at 1) inflated by ``(t_{0.975, d_ij}/z_{0.975})²`` with
    ``d_ij = min(d, k_j - 1)``. The design effect's variance is the larger of the cell's
    sandwich variance ``V_(ij),(ij)`` and a cluster-model variance that reads each group's
    size and a within-group correlation taken from the class at other levels or from the
    level's other classes. Where that correlation reflects the cell's bursts, the model
    variance keeps a cell that holds few of them from being both low and falsely precise;
    the sandwich covers what the model assumes away. A class that bursts at one level only,
    whose level's other classes do not, is not covered.

    The table is untestable with fewer than two levels, classes or groups, a design df
    below 1, a zero design-effect trace, or more than ``MAX_GROUPED_CELLS`` cells.

    Parameters
    ----------
    factor_codes : NDArray[np.intp]
        Factor level per row, dense ``0..r-1`` (every code occurs).
    class_codes : NDArray[np.intp]
        Class per row, dense ``0..K-1`` (every code occurs).
    groups : NDArray[np.intp]
        Group id per row, any integers.

    Returns
    -------
    GroupedTable
        The table, the factor test and the cell tests.

    Examples
    --------
    A sequence-level factor over 40 sequences of 25 frames, each sequence showing one class
    drawn independently of the factor. A row-level chi-square on the 1000 frames gives
    p = 3e-8; the grouped test counts each sequence about once:

    >>> rng = np.random.default_rng(0)
    >>> groups = np.repeat(np.arange(40), 25)
    >>> table = grouped_table(groups % 2, rng.integers(0, 3, 40)[groups], groups)
    >>> round(table.design_effect, 1), table.effective_groups, round(table.p_value, 3)
    (25.6, 13.0, 0.519)
    """
    f = np.asarray(factor_codes, dtype=np.intp)
    y = np.asarray(class_codes, dtype=np.intp)
    r, k, n = int(f.max()) + 1, int(y.max()) + 1, f.size
    cells = f * k + y
    observed = np.bincount(cells, minlength=r * k).reshape(r, k).astype(np.float64)
    p_f, p_y = observed.sum(1) / n, observed.sum(0) / n
    expected = n * np.outer(p_f, p_y)

    if r * k > MAX_GROUPED_CELLS:
        _logger.warning(
            "A %d x %d table exceeds MAX_GROUPED_CELLS (%d); the factor is untestable.", r, k, MAX_GROUPED_CELLS
        )
        return _untestable(observed, expected)
    group_index = np.unique(groups, return_inverse=True)[1].ravel()
    n_groups = int(group_index.max()) + 1
    levels, classes = _group_counts(f, group_index, r), _group_counts(y, group_index, k)
    k_level, k_class = _kish(levels[1], levels[2], r), _kish(classes[1], classes[2], k)
    effective = float(k_level.min())
    if classes[0].size == n_groups:  # the class is constant within every group
        effective = min(effective, float(k_class.min()))
    design_df = effective - 1
    if min(r, k, n_groups) < 2 or design_df < 1:
        return _untestable(observed, expected)

    v = _sandwich(cells, group_index, n_groups, p_f, p_y)
    var_cell = v.diagonal().copy()
    g = 1 / expected.ravel()
    trace = float(g @ var_cell)
    if trace <= TOLERANCE:
        return _untestable(observed, expected)
    sqrt_g = np.sqrt(g)
    v *= sqrt_g[:, None]
    v *= sqrt_g
    trace_sq = float(np.vdot(v, v))  # tr((GV)²) = ‖G^½ V G^½‖_F²

    df = (r - 1) * (k - 1)
    chi_squared = float(g @ (observed - expected).ravel() ** 2)
    design_effect = trace / df
    a2_plus_1 = max(1.0, trace_sq / (df * design_effect**2))
    df_num = df / a2_plus_1
    df_den = df_num * design_df
    p_value = float(f_dist.sf(chi_squared / (design_effect * a2_plus_1 * df_num), df_num, df_den))
    score = float(bias_corrected_cramers_v(chi_squared, n, r, k, design_effect=design_effect))
    insufficient = cochran_breach(expected / design_effect) or design_df < MIN_SPARSE_DF

    # Korn-Graubard cells on effective counts
    p_i, p_j = p_f[:, None], p_y[None, :]
    var_cell = var_cell.reshape(r, k)
    sizes = np.bincount(group_index, minlength=n_groups).astype(np.float64)
    var_test = np.maximum(var_cell, _pooled_cell_variance(var_cell, levels, classes, sizes, p_y))
    deff = np.maximum(1.0, var_test / (n * p_i * (1 - p_i) * p_j * (1 - p_j)))
    cell_df = np.broadcast_to(np.minimum(design_df, k_class - 1), (r, k))
    scale = deff * (t_dist.ppf(0.975, cell_df) / _Z_975) ** 2  # NaN where cell_df = 0, masked below
    o_eff, n_eff = observed / scale, n * p_i / scale
    lower = np.where(o_eff < n_eff, betainc(n_eff - o_eff, o_eff + 1, 1 - p_j), 1.0)
    upper = np.where(o_eff > 0, betainc(o_eff, n_eff - o_eff + 1, p_j), 1.0)
    cell_p = np.minimum(1.0, 2 * np.minimum(lower, upper))
    cell_p[(var_cell <= TOLERANCE * var_cell.max()) | (cell_df <= 0)] = np.nan
    cell_sparse = (expected / deff < MIN_SPARSE_EXPECTED) | (cell_df < MIN_SPARSE_DF)

    return GroupedTable(
        observed,
        expected,
        chi_squared,
        design_effect,
        df_num,
        df_den,
        p_value,
        effective,
        score,
        insufficient,
        cell_p,
        cell_sparse,
        True,
    )
