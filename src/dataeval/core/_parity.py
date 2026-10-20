__all__ = []

from typing import Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

from dataeval._experimental import experimental
from dataeval._log import get_logger
from dataeval.core._association import (
    benjamini_hochberg,
    bias_corrected_cramers_v,
    cochran_breach,
    hypergeometric_cell_pvalues,
)
from dataeval.core._rao_scott import MIN_SPARSE_EXPECTED, GroupedTable, _untestable, grouped_table
from dataeval.types import Array1D, Array2D
from dataeval.utils._array import as_numpy
from dataeval.utils.scipy.stats import chi2_contingency

_logger = get_logger(__name__)

# D2: raised here and by the Parity evaluator when nothing establishes the independent unit.
NO_SAMPLING_UNIT = (
    "Cannot establish independent sampling units for this metadata. Provide `item_indices` "
    "(label -> item) or use `Metadata` levels, or pass `assume_independent=True` if each row "
    "is its own independent item. Row-level inference gives invalid p-values when several "
    "rows belong to one image or sequence."
)


class ParityCells(TypedDict):
    r"""
    Per-cell results of :func:`.parity`, as flat arrays with one entry per cell.

    A cell is one (factor, observed level, observed class) combination. Every combination
    of the levels and classes that occur is listed, observed zeros included, factor by
    factor, then level by level, then class by class.

    .. warning::
       This feature is experimental and may change or be removed in future releases.

    Attributes
    ----------
    factor : NDArray[np.intp]
        Position of the cell's factor in ``factor_data``.
    level : NDArray[np.intp]
        The factor's code for the cell.
    class_label : NDArray[np.intp]
        The class label of the cell.
    observed : NDArray[np.intp]
        Rows in the cell.
    expected : NDArray[np.float64]
        Rows expected under homogeneity, the level's rows times the class's share.
    deficit : NDArray[np.float64]
        ``expected - observed``: positive where the cell holds fewer rows than expected.
    lift : NDArray[np.float64]
        ``observed / expected``.
    p_values : NDArray[np.float64]
        Exact two-sided p-value of the cell, NaN where it is untestable.
    p_values_adjusted : NDArray[np.float64]
        Benjamini-Hochberg adjusted over the testable cells of the call, NaN elsewhere.
    sparse : NDArray[np.bool\_]
        Too little expected data, or too few effective groups, to rely on the cell.
    testable : NDArray[np.bool\_]
        Whether the cell could be tested at all.
    """

    factor: NDArray[np.intp]
    level: NDArray[np.intp]
    class_label: NDArray[np.intp]
    observed: NDArray[np.intp]
    expected: NDArray[np.float64]
    deficit: NDArray[np.float64]
    lift: NDArray[np.float64]
    p_values: NDArray[np.float64]
    p_values_adjusted: NDArray[np.float64]
    sparse: NDArray[np.bool_]
    testable: NDArray[np.bool_]


class ParityResult(TypedDict):
    r"""
    Type definition for parity output.

    .. warning::
       This feature is experimental and may change or be removed in future releases.

    Attributes
    ----------
    scores : NDArray[np.float64]
        Bias-corrected Cramér's V per factor, from 0 (no association) to 1 (perfect
        association). NaN where the factor is untestable.
    chi_squared : NDArray[np.float64]
        Pearson's X² per factor, divided by the design effect.
    df_num : NDArray[np.float64]
        Numerator degrees of freedom of the reference distribution.
    df_den : NDArray[np.float64]
        Denominator degrees of freedom; infinite for independent rows (a χ² reference).
    p_values : NDArray[np.float64]
        Raw p-value per factor, NaN where untestable.
    p_values_adjusted : NDArray[np.float64]
        Benjamini-Hochberg adjusted over the testable factors, NaN elsewhere.
    design_effects : NDArray[np.float64]
        Mean generalized design effect δ̄; 1 for independent rows.
    effective_groups : NDArray[np.float64]
        Kish effective number of groups behind the thinnest level; the row count for
        independent rows.
    insufficient : NDArray[np.bool\_]
        Whether the table is too thin to trust the reference distribution (a screen, not a
        validity guarantee). True for an untestable factor.
    testable : NDArray[np.bool\_]
        Whether the factor could be tested at all.
    cells : ParityCells
        Exact per-cell results.
    test : {"pearson", "rao_scott_f"}
        The factor test run: Pearson's χ² on independent rows, or the second-order
        Rao-Scott F on groups.
    n_rows : int
        Number of rows.
    n_groups : int
        Number of independent groups; the row count for independent rows.
    """

    scores: NDArray[np.float64]
    chi_squared: NDArray[np.float64]
    df_num: NDArray[np.float64]
    df_den: NDArray[np.float64]
    p_values: NDArray[np.float64]
    p_values_adjusted: NDArray[np.float64]
    design_effects: NDArray[np.float64]
    effective_groups: NDArray[np.float64]
    insufficient: NDArray[np.bool_]
    testable: NDArray[np.bool_]
    cells: ParityCells
    test: Literal["pearson", "rao_scott_f"]
    n_rows: int
    n_groups: int


def _independent(f: NDArray[np.intp], y: NDArray[np.intp], r: int, k: int) -> GroupedTable:
    """Pearson's χ² with exact hypergeometric cells, for independent rows."""
    observed = np.bincount(f * k + y, minlength=r * k).reshape(r, k).astype(np.float64)
    expected = observed.sum(1, keepdims=True) * observed.sum(0, keepdims=True) / f.size
    if r < 2 or k < 2:
        return _untestable(observed, expected)
    chi = chi2_contingency(observed, correction=False)
    return GroupedTable(
        observed,
        expected,
        float(chi.statistic),
        1.0,
        float((r - 1) * (k - 1)),
        float("inf"),
        float(chi.pvalue),
        float(f.size),
        float(bias_corrected_cramers_v(chi.statistic, f.size, r, k)),
        cochran_breach(expected),
        hypergeometric_cell_pvalues(observed),
        expected < MIN_SPARSE_EXPECTED,
        True,
    )


def _adjusted(p_values: NDArray[np.float64], testable: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Benjamini-Hochberg over the testable entries only; NaN elsewhere."""
    adjusted = np.full(p_values.shape, np.nan)
    if testable.any():
        adjusted[testable] = benjamini_hochberg(p_values[testable])
    return adjusted


def _validated(
    factor_data: Array2D[int],
    class_labels: Array1D[int],
    groups: Array1D[int] | None,
    assume_independent: bool,
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.intp], NDArray[np.intp] | None, int]:
    """
    Check the inputs and resolve the sampling unit.

    Returns
    -------
    tuple
        The factor codes, the dense class index per row, the classes, the group per row
        (None when every row is its own unit) and the number of groups.
    """
    if groups is not None and assume_independent:
        raise ValueError(
            "`groups` and `assume_independent=True` were both given: `groups` records the "
            "independent unit, so there is nothing to declare. Pass one or the other."
        )
    if groups is None and not assume_independent:
        raise ValueError(NO_SAMPLING_UNIT)

    factors = as_numpy(factor_data, dtype=np.intp, required_ndim=2)
    labels = as_numpy(class_labels, dtype=np.intp, required_ndim=1)
    n = labels.size
    if n == 0 or factors.shape[0] != n:
        raise ValueError(f"factor_data has {factors.shape[0]} rows and class_labels {n}; both need the same rows.")
    group_ids = None if groups is None else as_numpy(groups, dtype=np.intp, required_ndim=1)
    if group_ids is not None and group_ids.size != n:
        raise ValueError(f"groups has {group_ids.size} entries for {n} rows.")
    n_groups = n if group_ids is None else int(np.unique(group_ids).size)
    classes, y = np.unique(labels, return_inverse=True)
    # One row per group is the rows being independent.
    return factors, y.ravel(), classes, None if n_groups == n else group_ids, n_groups


def _factor_table(
    f: NDArray[np.intp], y: NDArray[np.intp], r: int, k: int, groups: NDArray[np.intp] | None
) -> GroupedTable:
    """One factor's test and cells, with ``chi_squared`` already divided by the design effect."""
    if groups is None or r < 2 or k < 2:
        return _independent(f, y, r, k)
    table = grouped_table(f, y, groups)
    return table._replace(chi_squared=table.chi_squared / table.design_effect)


@experimental
def parity(
    factor_data: Array2D[int],
    class_labels: Array1D[int],
    groups: Array1D[int] | None = None,
    *,
    assume_independent: bool = False,
) -> ParityResult:
    """
    Test whether each factor's levels share one class distribution, factor by factor and cell by cell.

    .. warning::
       This feature is experimental and may change or be removed in future releases.

    For each factor, tests label-proportion homogeneity (the class mix is the same at every
    level) and scores the association with the bias-corrected Cramér's V. Every
    (level, class) cell also gets an exact two-sided test of whether it holds more or fewer
    rows than homogeneity predicts. Both margins are conditioned on, so no distribution is
    assumed for either the factor or the class labels.

    Parameters
    ----------
    factor_data : Array2D[int]
        Binned metadata factor values. Shape should be (n_samples, n_factors).
    class_labels : Array1D[int]
        Observed class labels. Shape should be (n_samples,).
    groups : Array1D[int] or None, default None
        The independent group (image, sequence) each row belongs to, shape (n_samples,).
        Rows inside a group may depend on each other in any way. None requires
        ``assume_independent=True``.

        .. versionadded:: 1.2
    assume_independent : bool, default False
        Declare each row its own independent item. Required when ``groups`` is None, and
        contradictory alongside it.

        .. versionadded:: 1.2

    Returns
    -------
    ParityResult
        Per-factor arrays (``scores``, ``chi_squared``, ``df_num``, ``df_den``,
        ``p_values``, ``p_values_adjusted``, ``design_effects``, ``effective_groups``,
        ``insufficient``, ``testable``), the per-cell results in ``cells``, and the scalars
        ``test``, ``n_rows`` and ``n_groups``.

    Raises
    ------
    ValueError
        If ``groups`` is None and ``assume_independent`` is False, if both are given, or if
        the inputs do not have one entry per row.

    See Also
    --------
    :class:`~dataeval.bias.Parity`

    Notes
    -----
    **The null hypothesis.** The class distribution of rows is the same at every factor
    level. Groups are independent of each other; rows inside a group may depend on each
    other in any way. Dependence *between* groups is not corrected.

    **Independent rows** (``assume_independent=True``, or one row per group): Pearson's χ²
    without continuity correction against a χ² reference with ``(r-1)(K-1)`` df, the
    Bergsma (2013) bias-corrected Cramér's V, and Cochran's criterion on the expected
    counts as the ``insufficient`` screen. Each cell is the exact hypergeometric test of its
    cell-versus-rest 2×2 collapse; it is sparse when its expected count is below 5.

    **Groups**: the second-order Rao-Scott F (Rao & Scott 1984; Thomas & Rao 1987) with
    its denominator df scaled by the Kish effective number of groups behind the thinnest
    level, and an exact Korn & Graubard (1998) test per cell on effective counts. A
    cell's variance is never below a Kish cluster-model variance: each group's size, with a
    within-group correlation estimated from the class at other levels or from the level's
    other classes. Where either of those reflects the cell's bursts, a cell that holds few
    of them is not read as both short and precise; see the known limit below.
    ``chi_squared`` is Pearson's X² divided by the design effect δ̄, and V has its null bias
    matched to δ̄.

    **Known limit of the grouped cell tests.** They can raise false alarms when a class
    comes in bursts (most rows of an image or sequence are that class) at one level of a
    factor only, and that level's other classes do not burst; with two classes there are
    no others. A cell of that class at that level that by chance holds none of its bursts
    looks like a real shortage, and no estimate from the data tells the two apart. Look
    for it as a rare class seen in one or a few crowded images, all at one level, or a
    factor whose levels each have their own bursty class. In simulated nulls, the share of
    tables with a falsely flagged cell (nominal 5%) was 11-17% with
    two classes and about two bursty images at one of 3 to 10 levels, 3-10% for a single
    such cell in a 2×2 to 3×4 table, and 26-33% and 49-54% when each of 20 or 40 levels burst
    in its own class. Read an ``underrepresented`` cell of such a class at such a level as
    a reason to look at its images, not as a finding.

    A factor with fewer than two observed levels or classes, or a grouped table with fewer
    than two groups or too few effective groups, is **untestable**: its V, statistic,
    p-value, design effect and dfs are NaN, not a verdict of independence. A cell is
    untestable when its factor is, or when its own variance is degenerate.

    P-values are Benjamini-Hochberg adjusted across the testable factors, and separately
    across the testable cells of the call. Cells share their table's margins, so the
    adjustment does not carry an unconditional false discovery rate guarantee.

    References
    ----------
    Bergsma, W. (2013). A bias-correction for Cramér's V and Tschuprow's T.
    Journal of the Korean Statistical Society, 42(3), 323-328.

    Cochran, W. G. (1954). Some methods for strengthening the common chi-square tests.
    Biometrics, 10(4), 417-451.

    Rao, J. N. K., & Scott, A. J. (1984). On chi-squared tests for multiway contingency tables
    with cell proportions estimated from survey data. Annals of Statistics, 12(1), 46-60.

    Thomas, D. R., & Rao, J. N. K. (1987). Small-sample comparisons of level and power for
    simple goodness-of-fit statistics under cluster sampling. JASA, 82(398), 630-636.

    Korn, E. L., & Graubard, B. I. (1998). Confidence intervals for proportions with small
    expected number of positive counts estimated from survey data. Survey Methodology, 24(2),
    193-201.

    Kish, L. (1965). Survey Sampling. Wiley.

    Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a
    practical and powerful approach to multiple testing. Journal of the Royal
    Statistical Society: Series B, 57(1), 289-300.

    Examples
    --------
    A factor whose rare level holds none of class 0, where 30 rows were expected:

    >>> factor = np.repeat([0, 0, 1, 1], [0, 60, 1500, 1440])[:, None]
    >>> labels = np.repeat([0, 1, 0, 1], [0, 60, 1500, 1440])
    >>> result = parity(factor, labels, assume_independent=True)
    >>> round(float(result["scores"][0]), 3), result["test"]
    (0.142, 'pearson')
    >>> cells = result["cells"]
    >>> int(cells["observed"][0]), float(cells["expected"][0]), float(cells["deficit"][0])
    (0, 30.0, 30.0)
    >>> bool(cells["p_values_adjusted"][0] < 1e-15)
    True

    A sequence-level factor over 40 sequences of 25 frames, each sequence showing one class
    drawn independently of the factor. Read as 1000 independent rows it looks associated;
    grouped by sequence, it does not:

    >>> rng = np.random.default_rng(0)
    >>> groups = np.repeat(np.arange(40), 25)
    >>> factor, labels = (groups % 2)[:, None], rng.integers(0, 3, 40)[groups]
    >>> rows = parity(factor, labels, assume_independent=True)
    >>> clustered = parity(factor, labels, groups)
    >>> f"{rows['p_values'][0]:.0e}", round(float(clustered["p_values"][0]), 3), clustered["test"]
    ('3e-08', 0.519, 'rao_scott_f')
    """
    factors, y, classes, group_ids, n_groups = _validated(factor_data, class_labels, groups, assume_independent)
    _logger.info("Starting parity calculation")

    tables: list[GroupedTable] = []
    cell_levels: list[NDArray[np.intp]] = []
    for column in factors.T:
        levels, f = np.unique(column, return_inverse=True)
        tables.append(_factor_table(f.ravel(), y, levels.size, classes.size, group_ids))
        cell_levels.append(np.repeat(levels, classes.size))

    def per_factor(field: str) -> NDArray[np.float64]:
        return np.array([getattr(t, field) for t in tables], dtype=np.float64)

    testable = np.array([t.testable for t in tables], dtype=np.bool_)
    p_values = per_factor("p_value")

    cell_p = np.concatenate([t.cell_p_values.ravel() for t in tables])
    cell_testable = ~np.isnan(cell_p)
    observed = np.concatenate([t.observed.ravel() for t in tables])
    expected = np.concatenate([t.expected.ravel() for t in tables])
    cells = ParityCells(
        factor=np.repeat(np.arange(len(tables), dtype=np.intp), [t.observed.size for t in tables]),
        level=np.concatenate(cell_levels).astype(np.intp),
        class_label=np.tile(classes, sum(t.observed.shape[0] for t in tables)).astype(np.intp),
        observed=observed.astype(np.intp),
        expected=expected,
        deficit=expected - observed,
        lift=observed / expected,
        p_values=cell_p,
        p_values_adjusted=_adjusted(cell_p, cell_testable),
        sparse=np.concatenate([t.cell_sparse.ravel() for t in tables]),
        testable=cell_testable,
    )
    insufficient = np.array([t.insufficient for t in tables], dtype=np.bool_)

    _logger.info("Parity calculation complete: %d factors analyzed, %d testable", len(tables), testable.sum())
    if insufficient.any():
        _logger.warning("Found insufficient data for %d factor(s)", insufficient.sum())

    return ParityResult(
        scores=per_factor("score"),
        chi_squared=per_factor("chi_squared"),
        df_num=per_factor("df_num"),
        df_den=per_factor("df_den"),
        p_values=p_values,
        p_values_adjusted=_adjusted(p_values, testable),
        design_effects=per_factor("design_effect"),
        effective_groups=per_factor("effective_groups"),
        insufficient=insufficient,
        testable=testable,
        cells=cells,
        test="pearson" if group_ids is None else "rao_scott_f",
        n_rows=y.size,
        n_groups=n_groups,
    )
