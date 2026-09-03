__all__ = []

from collections.abc import Iterable
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import mutual_info_score
from sklearn.metrics.cluster import contingency_matrix, expected_mutual_information

from dataeval._log import get_logger
from dataeval.config import get_max_processes, get_seed
from dataeval.types import Array1D, Array2D
from dataeval.utils._array import as_numpy

_logger = get_logger(__name__)


class MutualInfoResult(TypedDict):
    """
    Type definition for normalized mutual information output.

    Attributes
    ----------
    class_to_factor : NDArray[np.float64]
        1D array of shape (F+1,) holding the share of the class label's entropy
        that each factor accounts for, corrected for chance, with the class label itself at
        index 0. 1.0 means the factor determines the class outright and 0.0 means it says
        nothing beyond what its cardinality would produce by chance. The entries are
        comparable to each other while the factors are of similar cardinality; a factor
        taking several hundred values is scored conservatively and can fall below a
        coarser factor that accounts for less. See Notes.
    interfactor : NDArray[np.float64]
        (F, F) symmetric matrix of normalized mutual information
        between metadata factors only (excluding class labels), corrected for chance on the
        same grounds. Neither factor in a pair is privileged, so these are normalized by the
        smaller of the two entropies rather than against the class entropy, which puts them
        on a different scale from ``class_to_factor``.
    """

    class_to_factor: NDArray[np.float64]
    interfactor: NDArray[np.float64]


def _validate_num_neighbors(num_neighbors: int) -> int:
    if not isinstance(num_neighbors, int | float):
        raise TypeError(
            f"Variable {num_neighbors} is not real-valued numeric type."
            "num_neighbors should be an int, greater than 0 and less than"
            "the number of samples in the dataset",
        )
    if num_neighbors < 1:
        raise ValueError(
            f"Invalid value for {num_neighbors}."
            "Choose a value greater than 0 and less than number of samples"
            "in the dataset.",
        )
    if isinstance(num_neighbors, float):
        num_neighbors = int(num_neighbors)
        _logger.warning(f"Variable {num_neighbors} is currently type float and will be truncated to type int.")

    return num_neighbors


def _entropy_of(values: NDArray[Any]) -> float:
    """Entropy of a discrete sample in nats, read off its observed value counts."""
    _, counts = np.unique(values, return_counts=True)
    return float(entropy(counts / counts.sum()))


def _adjusted_share(target: NDArray[Any], factor: NDArray[Any], target_entropy: float) -> float:
    """
    Share of a target's entropy that a discretized factor removes, corrected for chance.

    Mutual information estimated from a contingency table grows with the number of
    categories even when the two variables are independent: every extra category is
    another chance for the counts to line up by luck, and with finite data some of them
    do. Subtracting the value expected under a random assignment holding the same margins
    removes that floor, so an unrelated factor scores near zero whatever its cardinality
    while a genuinely informative one is barely touched.

    References
    ----------
    [1] Information theoretic measures for clusterings comparison: Variants, properties,
        normalization and correction for chance. Vinh, N. X., Epps, J., & Bailey, J.
        (2010). Journal of Machine Learning Research, 11, 2837-2854.
    """
    contingency = contingency_matrix(target, factor, sparse=True)
    observed = float(mutual_info_score(None, None, contingency=contingency))
    expected = float(expected_mutual_information(contingency, target.shape[0]))
    denominator = target_entropy - expected
    # A factor holding a value per entity leaves every cell of the table with a single
    # observation, so the target is recovered exactly and the expectation rises to meet
    # the target's own entropy. Both sides of the ratio collapse to rounding error, and
    # their quotient lands near 1.0 on the sign of that error -- an identifier column
    # would report itself as the dataset's strongest factor. It generalizes to nothing, so
    # it is scored as nothing. A factor that genuinely determines the target keeps its
    # cells populated and leaves the expectation well below the entropy, so this does not
    # catch it.
    if denominator <= target_entropy * 1e-6:
        return 0.0
    return float(np.clip((observed - expected) / denominator, 0.0, 1.0))


def _adjusted_association(
    first: NDArray[Any],
    second: NDArray[Any],
    first_bound: float,
    second_bound: float,
    first_entropy: float,
    second_entropy: float,
) -> float:
    """
    Association between two tabulable factors, corrected for chance.

    The chance correction is the one ``_adjusted_share`` applies, over a different
    denominator. Neither factor here is the thing being explained, so there is no reason
    to divide by one of the two entropies rather than the other; the smaller of them is
    the most either could account for, and it is what the pair is scored against. The
    class-to-factor direction is not symmetric in that way and uses the class entropy
    alone, so a value from that row and a value from this block answer different
    questions and are not comparable to each other.

    ``first_bound`` and ``second_bound`` are each factor's entropy where its alphabet is
    a property of the variable, and infinite where the alphabet came from binning. An
    infinite bound contributes no ceiling, because a binned factor's entropy measures the
    cut rather than the variable: it grows with the bin count, so dividing by it makes the
    reported association shrink as the same data is cut more finely. When neither factor
    offers a real bound the pair is scored by the Linfoot transformation instead, so the
    denominator no longer moves with the bin count. The numerator still does: cutting the
    same values more finely changes what the contingency table captures. See
    :func:`mutual_info` for the measurements.

    Dropping the denominator does not make the branch scale-free, which is what
    ``first_entropy`` and ``second_entropy`` are for. Mutual information is capped by the
    smaller of the two entropies however the codes arose, so the largest value the
    transformation can return is ``1 - exp(-2 * min(H1, H2))``: 0.75 for a pair of two-level
    factors, 0.99 by sixteen levels, and lower again where the levels are unevenly filled --
    0.47 for a binary split holding 90% of the rows on one side. Left as is, a coarsely cut
    pair is scored against a lower reachable maximum than a finely cut one, so a duplicated
    binary factor reads 0.75 while a duplicated sixteen-bin factor reads 0.996, and a fixed
    correlation threshold means something different for each.

    Dividing by that reachable maximum is what puts them back on one scale, and it is the
    same move the entropy branch makes -- both divide by the most the pair could have
    shared. It leaves a duplicate at 1.0 whatever the cut, and leaves the ceiling's
    *growth* with bin count removed, which is what the infinite bound is for. What it does
    not do, and must not, is restore resolution: a coarse cut genuinely shares less, and the
    numerator still says so. See :doc:`/concepts/Binning` for the measurements.
    """
    contingency = contingency_matrix(first, second, sparse=True)
    observed = float(mutual_info_score(None, None, contingency=contingency))
    expected = float(expected_mutual_information(contingency, first.shape[0]))
    # Clamped before either branch: the correction can drive an independent pair slightly
    # below zero, which the entropy branch clips away but the Linfoot branch would turn
    # into a small positive value by way of a negative exponent.
    adjusted = max(observed - expected, 0.0)

    ceiling = min(first_bound, second_bound)
    if np.isinf(ceiling):
        # The most this pair could have shared, on the same scale the value is reported on.
        # Subtracting the expectation matches the entropy branch below: the numerator is
        # chance-corrected, so the maximum it can reach is chance-corrected too.
        reachable = min(first_entropy, second_entropy) - expected
        if reachable <= min(first_entropy, second_entropy) * 1e-6:
            return 0.0
        return float(np.clip((1.0 - np.exp(-2.0 * adjusted)) / (1.0 - np.exp(-2.0 * reachable)), 0.0, 1.0))
    denominator = ceiling - expected
    # A constant factor has no entropy to share, and a pair of near-identifier columns
    # collapses both sides of the ratio to rounding error; see ``_adjusted_share``.
    if denominator <= ceiling * 1e-6:
        return 0.0
    return float(np.clip(adjusted / denominator, 0.0, 1.0))


def _target_to_factor(
    target: NDArray[Any],
    data: NDArray[np.intp],
    coded_list: list[bool],
    raw_mi: NDArray[Any],
) -> NDArray[np.float64]:
    """
    Score every factor by how much of one target's uncertainty it removes.

    Every entry is divided by the target's own entropy, which is what makes the row
    readable as a ranking: a factor scores 1.0 when it determines the target outright and
    0.0 when it says nothing the target did not already say. Dividing each factor by its
    own entropy instead -- the symmetric convention used between factors, where neither
    side is privileged -- would reward a factor for having few categories rather than for
    being informative, and reorder the row accordingly.

    The denominator is not identical across factors, since the chance correction is
    subtracted from it as well and that subtraction grows with a factor's cardinality.
    The effect is small over the range a binned factor occupies and grows from there, so
    the row ranks factors of comparable width reliably and scores a very wide one
    conservatively rather than generously.

    Because the denominator is the *target's* entropy rather than the factor's, this row
    is unaffected by how finely a factor was cut: refining a binned factor's cuts moves
    the numerator toward the mutual information the unbinned values carry and leaves the
    denominator alone, so the score converges instead of drifting. That is what separates
    this row from the factor-to-factor block, where both sides of the ratio move with the
    bin count unless the ceiling is dropped; see ``_adjusted_association``.

    ``coded_list`` says which columns hold codes a contingency table can be built over,
    which is read from the column's own values rather than declared by the caller. Coded
    columns are corrected for chance; see ``_adjusted_share``. A column of measured values
    has no contingency table to take that expectation over, so it is scored on the
    estimated mutual information alone. This is deliberately not the caller's
    ``discrete_features``: a column of distinct integer identifiers is coded whatever the
    caller calls it, and it is exactly the case the chance correction exists for.
    """
    row = np.zeros(len(coded_list), dtype=np.float64)
    row[0] = 1.0  # the target against itself
    target_entropy = _entropy_of(target)
    if target_entropy <= 0:
        # A target taking one value carries no uncertainty for a factor to explain.
        return row
    for j in range(1, len(coded_list)):
        share = (
            _adjusted_share(target, data[:, j], target_entropy)
            if coded_list[j]
            else float(np.clip(float(raw_mi[j]) / target_entropy, 0.0, 1.0))
        )
        row[j] = share
    return row


def _is_coded(column: NDArray[Any]) -> bool:
    """Whether a column holds category codes rather than measured values.

    Integral values are codes — bin indices, ordinals, counts, identifiers — and a
    contingency table over them is exact, which is what makes the chance correction
    available. A column carrying a fractional part was measured rather than coded, and
    tabulating it would give most rows a cell of their own.

    Read from the values rather than taken from the caller, because it is a fact about
    the array in hand and not a judgement about the variable behind it. Non-finite entries
    are ignored: a NaN is neither integral nor measured, and would otherwise make every
    column carrying one look measured.
    """
    finite = column[np.isfinite(column)] if np.issubdtype(column.dtype, np.inexact) else column
    return bool(finite.size == 0 or np.all(finite == np.floor(finite)))


def _merge_labels_and_factors(
    class_labels: NDArray[np.intp],
    factor_data: NDArray[np.intp],
    discrete_features: Iterable[bool],
) -> tuple[NDArray[np.intp], list[bool], list[bool], list[bool]]:
    """Stack the label axis onto the factors and answer three questions about each column.

    Two independent things have to be known about a column, and they come from different
    places:

    - **Can it be tabulated?** ``coded_list``, read from the values themselves by
      :func:`_is_coded`. This decides which estimator reads the column and whether the
      chance correction is available, and is never the caller's to state — a column either
      holds codes or it does not.
    - **Is its alphabet a property of the variable?** ``declared_list``, the caller's
      ``discrete_features``. This decides only whether the column's entropy is a legitimate
      ceiling to divide by. A binned continuous factor is coded but its alphabet is an
      artifact of where the cuts fell, so it answers True to the first question and False
      to the second — the combination the previous single flag could not express.

    Returns the stacked data, the list handed to sklearn, ``coded_list`` and
    ``declared_list``. The sklearn list is ``coded_list`` with one substitution: a column
    of all-distinct values is presented to the estimator as continuous so it does not
    treat every value as its own category. ``coded_list`` itself keeps that column coded,
    since a per-row identifier is exactly the case the chance correction exists for.
    """
    declared_list = [True] + list(discrete_features)

    # Use numeric data for MI
    data = np.hstack((class_labels[:, np.newaxis], factor_data))
    coded_list = [_is_coded(data[:, i]) for i in range(data.shape[1])]

    # Present coded features composed of distinct values as continuous for `mutual_info_classif`
    sklearn_list = list(coded_list)
    for i in range(len(sklearn_list)):
        if len(data) == len(np.unique(data[:, i])):
            sklearn_list[i] = False

    return data, sklearn_list, coded_list, declared_list


def mutual_info(  # noqa: C901
    class_labels: Array1D[int],
    factor_data: Array2D[int | float],
    discrete_features: Array1D[bool],
    num_neighbors: int = 5,
) -> MutualInfoResult:
    """
    Compute normalized mutual information between factors, transformed to lie in [0, 1].

    Factors include class label, metadata, and label/image properties.

    Parameters
    ----------
    class_labels : Array1D[int], shape - (N,)
        Target class labels as integer indices. Can be a 1D list, or array-like object.
    factor_data : Array2D[int | float], shape - (N, F)
        Factor values after binning or digitization. Can be a 2D list, or array-like object.
    discrete_features : Array1D[bool], shape - (F,)
        Whether each factor's set of values is a property of the *variable* rather than of
        how it was processed — True for a category, a count or any factor with a finite
        alphabet of its own, False for one whose values were produced by cutting a
        continuous quantity into bins. This does not select an estimator: which estimator
        reads a column is read from the column's own values. It selects only whether the
        column's entropy is used as a ceiling; see Notes. From a
        :class:`~dataeval.Metadata` this is ``[not b for b in metadata.is_binned]``. Can be
        a 1D list, or array-like object.

        .. versionchanged:: 1.2
            Required. v1.1 auto-detected it when unset, and that guess reads the column's
            values — the one thing they cannot answer: a factor cut into six bins and a
            factor with six categories are both the integers 0-5. Only the caller knows.
    num_neighbors : int, default 5
        Number of points to consider as neighbors. Consulted only for columns holding
        measured values, which are the only ones the neighbor-based estimator reads.

    Returns
    -------
    MutualInfoResult
        TypedDict containing:

        - class_to_factor: NDArray[np.float64], shape - (F+1,) - normalized MI between the
          class label and each factor, with the class label's own self-MI at index 0 and
          factor ``i`` of ``factor_data`` at index ``i+1``.
        - interfactor: NDArray[np.float64], shape - (F, F) - normalized MI between factors
          only; the class label is excluded, so row/column ``i`` is factor ``i`` of
          ``factor_data``.

        The two are indexed differently on purpose: the class label is one of the entries
        the first answers for and is not an entry of the second at all. See Notes for why
        they are also normalized differently.

    See Also
    --------
    :func:`sklearn.feature_selection.mutual_info_classif`
    :func:`sklearn.feature_selection.mutual_info_regression`
    :func:`sklearn.metrics.mutual_info_score`

    Notes
    -----
    We use `mutual_info_classif` from sklearn since class label is categorical.
    `mutual_info_classif` outputs are consistent up to O(1e-4) and depend on a random
    seed. MI is computed differently for categorical and continuous variables.

    Two decisions are made per column, from two different sources. **Which estimator reads
    it** is read from the column's own values: integral values are codes and are tabulated
    exactly, anything with a fractional part was measured and goes to the neighbor-based
    estimator. **Whether its entropy is a legitimate ceiling** is what ``discrete_features``
    declares, and nothing else depends on it.

    The two halves of the result answer different questions and are normalized
    accordingly. The class-to-factor row is directed -- it asks how much of the class label
    a factor accounts for -- so every entry is divided by the class entropy alone.
    Dividing each factor by its own entropy there would rank a factor by how few categories
    it has as much as by how much it explains. Because the denominator belongs to the class
    rather than to the factor, that row is unaffected by how finely a factor was cut.

    Between two factors neither side is privileged, so that block is divided by the
    smaller of the entropies **of the factors whose alphabet is their own**. A factor
    declared to have no alphabet of its own contributes no ceiling, because its entropy
    describes the cut rather than the variable: it grows with the bin count, so a ratio
    against it shrinks as the same data is cut more finely. On a bivariate normal pair with
    a true dependence of 0.81, dividing by a binned factor's entropy reports 0.39 at four
    bins falling to 0.09 at 128 -- on identical data, with only the cut changed. Where
    neither factor offers a real ceiling the pair is scored by the Linfoot transformation,
    which reads 0.70 at four bins and peaks at 0.79 by sixteen: still short of the truth at
    the coarse end, because binning genuinely destroyed the information, but no longer
    moving with a choice the caller did not make. It falls away again past 64 bins, where
    the sample can no longer fill the table and the chance correction has little left to
    keep.

    Whichever denominator applies, it is the most the pair could have shared rather than a
    fixed 1.0. For the entropy branch that is the smaller entropy; for the Linfoot branch it
    is that same entropy carried through the transformation, ``1 - exp(-2 * min(H1, H2))``,
    which is 0.75 for a two-level pair and 0.996 by sixteen levels. Without it a duplicated
    binary factor would read 0.75 and a duplicated sixteen-bin factor 0.996, and a fixed
    correlation threshold would mean something different for each.

    Both halves are corrected for chance wherever a contingency table exists. Mutual
    information read off one rises with the number of categories even when the factor and
    the class are independent, enough that an identifier column can outrank a genuine
    effect; see ``_adjusted_share``.

    References
    ----------
    [1] `Linfoot, E.H. (1957). "An Informational Measure of Correlation." Information and
    Control, 1(1), 85-89. <https://www.sciencedirect.com/science/article/pii/S001999585790116X>`_
    [2] Information theoretic measures for clusterings comparison: Variants, properties,
    normalization and correction for chance. Vinh, N. X., Epps, J., & Bailey, J. (2010).
    Journal of Machine Learning Research, 11, 2837-2854. https://jmlr.org/papers/v11/vinh10a.html

    Example
    -------
    Return balance (normalized mutual information) of factors with class_labels

    >>> rng = np.random.default_rng(175)
    >>> class_labels = rng.choice([0, 1, 2], size=2000)
    >>> factor_data = np.column_stack([
    ...     rng.choice([25, 35, 45, 55], size=2000),  # age, unrelated to the class
    ...     np.where(  # collection site, agreeing with the class 70% of the time
    ...         rng.random(2000) < 0.7, class_labels, rng.choice([0, 1, 2], size=2000)
    ...     ),
    ...     rng.choice([0, 1], size=2000),  # gender, unrelated to the class
    ... ])

    Each factor's integers stand for values of its own here — ages, site ids, a gender code
    — so every entry is True. A factor you had binned would be False:

    >>> result = mutual_info(
    ...     class_labels=class_labels,
    ...     factor_data=factor_data,
    ...     discrete_features=[True, True, True],
    ... )

    Only the site accounts for any of the class label; the two unrelated factors sit at
    zero rather than at the small positive value their cardinality alone would produce:

    >>> result["class_to_factor"]
    array([1.000e+00, 0.000e+00, 4.319e-01, 7.336e-04])
    >>> result["interfactor"]
    array([[1.000e+00, 3.752e-04, 0.000e+00],
           [3.752e-04, 1.000e+00, 0.000e+00],
           [0.000e+00, 0.000e+00, 1.000e+00]])
    """
    _logger.info("Starting mutual_info calculation with num_neighbors=%d", num_neighbors)

    class_labels_np = as_numpy(class_labels, dtype=np.intp, required_ndim=1)
    factor_data_np = as_numpy(factor_data, required_ndim=2)
    discrete_feat_np = as_numpy(discrete_features, dtype=np.bool_, required_ndim=1)

    _logger.debug("Input shapes: class_labels=%s, factor_data=%s", class_labels_np.shape, factor_data_np.shape)

    num_neighbors = _validate_num_neighbors(num_neighbors)
    data, sklearn_list, coded_list, declared_list = _merge_labels_and_factors(
        class_labels_np, factor_data_np, discrete_feat_np
    )
    # Counts columns of `data`, which is the class label followed by the F factors, so this
    # is F+1. `class_to_factor` is returned at this length; `interfactor` drops the class.
    num_columns = len(coded_list)

    _logger.debug(
        "Computing NMI over %d columns (%d factors, %d coded, %d with an alphabet of their own)",
        num_columns,
        num_columns - 1,
        sum(coded_list),
        sum(declared_list),
    )

    # initialize output matrix
    mi = np.full((num_columns, num_columns), np.nan, dtype=np.float32)

    # A factor whose alphabet is its own bounds what any pair containing it can share, so
    # its entropy is a legitimate ceiling. One whose values came out of a binning does not:
    # that entropy measures the cut and grows with the bin count, so it is left infinite
    # and contributes no ceiling at all.
    # `entropies` keeps what `norm_factor` throws away. A binned factor's entropy is not a
    # legitimate ceiling to divide by -- that is the whole point of the infinite bound -- but
    # it still caps what the pair can share, and the Linfoot branch needs it to report on a
    # scale a coarse cut and a fine one can both reach. Held per column rather than derived
    # per pair, which would be quadratic in the factors.
    norm_factor = np.zeros(num_columns)
    entropies = np.full(num_columns, np.inf)
    for i in range(num_columns):
        if coded_list[i] or declared_list[i]:
            entropies[i] = _entropy_of(data[:, i])
        norm_factor[i] = entropies[i] if declared_list[i] else np.inf

    # The estimator is consulted only where a column cannot be tabulated. With every column
    # holding codes -- which is every call arriving from :class:`~dataeval.bias.Balance`,
    # since `factor_data` is bin and category indices throughout -- nothing below reads
    # `mi`, and the pass is a full sklearn run per factor whose result is then overwritten.
    if all(coded_list):
        mi[:, :] = 0.0
    else:
        for idx in range(num_columns):
            mi[idx, :] = (mutual_info_classif if sklearn_list[idx] else mutual_info_regression)(
                data,
                data[:, idx],
                discrete_features=sklearn_list,  # type: ignore - sklearn function not typed
                n_neighbors=num_neighbors,
                random_state=get_seed(),
                n_jobs=get_max_processes(),  # type: ignore - added in 1.5
            )

    # Estimated mutual information in nats, kept before normalization because the
    # class-to-factor row is normalized differently from the factor-to-factor block.
    raw_mi = mi[0].copy()

    for idx in range(num_columns):
        # Normalization via entropy, pre-computed above
        for j in range(data.shape[1]):
            if np.isinf(norm_factor[j]) and np.isinf(norm_factor[idx]):
                mi[idx, j] = 1.0 - np.exp(-2.0 * float(mi[idx, j]))  # Linfoot transformation, mi in nats
            elif norm_factor[j] == 0 or norm_factor[idx] == 0:
                mi[idx, j] = 0.0
            else:
                # Clipped like every other branch. The neighbor estimator is not bounded by
                # the entropy it is divided by here -- it reads a measured column, whose
                # ceiling is not the coded partner's -- so a near-deterministic pair can
                # overshoot and report above 1.0 on a scale documented as [0, 1].
                mi[idx, j] = np.clip(mi[idx, j] / min(norm_factor[j], norm_factor[idx]), 0.0, 1.0)

    full_matrix = 0.5 * (mi + mi.T).astype(np.float64)
    interfactor = full_matrix[1:, 1:]

    # Every pair with a contingency table behind it is corrected for chance, on the same
    # grounds as the class row: plug-in mutual information rises with cardinality even
    # under independence, and this block carries a correlation threshold, so an uncorrected
    # value turns a finely binned factor into a reported correlation with everything. The
    # ceiling each pair is scored against comes from `norm_factor`, so a pair of binned
    # factors falls through to Linfoot rather than being divided by an artifact. Pairs
    # involving a column of measured values have no table and keep the estimator's own
    # normalization above.
    for i in range(1, num_columns):
        if not coded_list[i]:
            continue
        for j in range(i + 1, num_columns):
            if not coded_list[j]:
                continue
            adjusted = _adjusted_association(
                data[:, i], data[:, j], norm_factor[i], norm_factor[j], entropies[i], entropies[j]
            )
            interfactor[i - 1, j - 1] = interfactor[j - 1, i - 1] = adjusted

    # A factor shares all of itself with itself, whatever it is made of. Stated rather than
    # left to whichever branch above happened to run, which reported 1.0 for a tabulated
    # factor and a Linfoot-transformed self-estimate for a measured one -- two different
    # answers to a question with only one. A factor holding a single value has nothing to
    # share and stays at zero.
    for i in range(1, num_columns):
        interfactor[i - 1, i - 1] = 1.0 if _entropy_of(data[:, i]) > 0 else 0.0

    # Between two factors neither side is privileged, so that block is scored against the
    # smaller of the two entropies. The class-to-factor row asks a directed question --
    # how much of the class label does this factor account for -- and is scored accordingly.
    class_to_factor = _target_to_factor(data[:, 0], data, coded_list, raw_mi)

    _logger.info(
        "Mutual info calculation complete: %d factors, mean class_to_factor NMI=%.4f",
        num_columns - 1,
        np.mean(class_to_factor[1:]),
    )

    return MutualInfoResult(
        class_to_factor=class_to_factor,
        interfactor=interfactor,
    )


def mutual_info_classwise(
    class_labels: Array1D[int],
    factor_data: Array2D[int | float],
    num_neighbors: int = 5,
) -> NDArray[np.float64]:
    """
    Compute normalized mutual information (NMI) between factors.

    Factors include class label, metadata, and label/image properties.

    Parameters
    ----------
    class_labels : Array1D[int], shape - (N,)
        Target class labels as integer indices. Can be a 1D list, or array-like object.
    factor_data : Array2D[int | float], shape - (N, F)
        Factor values after binning or digitization. Can be a 2D list, or array-like object.
    num_neighbors : int, default 5
        Number of points to consider as neighbors. Consulted only for columns holding
        measured values, which are the only ones the neighbor-based estimator reads.

    Returns
    -------
    NDArray[np.float64]
        (num_classes, F+1) array holding, for each class taken against the
        rest, the share of that one-against-the-rest split's entropy which each factor
        accounts for, corrected for chance. Each row shares a single denominator, so it
        ranks the factors for that class the way ``class_to_factor`` ranks them for the
        label as a whole. Column 0 holds the class label measured against itself, which is
        1.0 for every class by construction and so carries no per-class information.

    See Also
    --------
    :func:`sklearn.feature_selection.mutual_info_classif`
    :func:`sklearn.feature_selection.mutual_info_regression`
    :func:`sklearn.metrics.mutual_info_score`

    Notes
    -----
    We use `mutual_info_classif` from sklearn since class label is categorical.
    `mutual_info_classif` outputs are consistent up to O(1e-4) and depend on a random
    seed. MI is computed differently for categorical and continuous variables. In all cases,
    we return either a normalization or transformation of MI onto the interval [0, 1].

    Each row takes one class against the rest and scores the factors against that split
    the way :func:`mutual_info` scores them against the label as a whole: a denominator
    shared across the row, and a correction for the mutual information that the factor's
    cardinality would produce by chance.

    .. versionchanged:: 1.2
        ``discrete_features`` was removed. It was accepted without effect and warned when
        set: every row here is divided by the entropy of one class against the rest, which
        belongs to the class label rather than to any factor, so there was no factor
        entropy for the declaration to select. :func:`mutual_info` does use it, for the
        factor-to-factor block this function does not return, and there it is required.

    Example
    -------
    Return classwise balance (normalized mutual information) of factors with individual class_labels

    >>> rng = np.random.default_rng(175)
    >>> class_labels = rng.choice([0, 1, 2], size=2000)
    >>> factor_data = np.column_stack([
    ...     rng.choice([25, 35, 45, 55], size=2000),  # age, unrelated to the class
    ...     np.where(  # collection site, agreeing with the class 70% of the time
    ...         rng.random(2000) < 0.7, class_labels, rng.choice([0, 1, 2], size=2000)
    ...     ),
    ...     rng.choice([0, 1], size=2000),  # gender, unrelated to the class
    ... ])
    >>> mutual_info_classwise(class_labels=class_labels, factor_data=factor_data)
    array([[1.000e+00, 0.000e+00, 4.071e-01, 1.361e-03],
           [1.000e+00, 0.000e+00, 3.943e-01, 0.000e+00],
           [1.000e+00, 0.000e+00, 4.312e-01, 8.627e-04]])
    """
    _logger.info("Starting mutual_info_classwise calculation with num_neighbors=%d", num_neighbors)

    class_labels_np = as_numpy(class_labels, dtype=np.intp, required_ndim=1)
    factor_data_np = as_numpy(factor_data, required_ndim=2)

    num_neighbors = _validate_num_neighbors(num_neighbors)
    # A constant rather than a real declaration: the declaration is discarded below, so a
    # list nothing reads is what gets built.
    data, sklearn_list, coded_list, _ = _merge_labels_and_factors(
        class_labels_np, factor_data_np, np.ones(factor_data_np.shape[1], dtype=np.bool_)
    )
    # Columns of `data`: the class label followed by the F factors, so F+1. Each returned
    # row is this wide, with the class label's self-MI at index 0.
    num_columns = len(coded_list)
    u_classes = np.unique(class_labels_np)
    num_classes = len(u_classes)

    _logger.debug("Computing classwise NMI for %d classes and %d factors", num_classes, num_columns - 1)

    if num_classes == 0:
        # No class to take against the rest, so there is no row to score. Returned rather
        # than stacked: np.stack has no shape to infer from an empty list.
        return np.zeros((0, num_columns), dtype=np.float64)

    # classwise targets (binary indicators)
    tgt_bin = data[:, 0][:, None] == u_classes

    # Compute MI. The estimate is read only where a column holds measured values, since
    # every other column is scored off its contingency table instead; with every column
    # coded the whole loop is an estimator pass nothing reads.
    classwise_mi = np.zeros((num_classes, num_columns), dtype=np.float32)
    if not all(coded_list[1:]):
        for idx in range(num_classes):
            classwise_mi[idx, :] = mutual_info_classif(
                data,
                tgt_bin[:, idx],
                discrete_features=sklearn_list,  # type: ignore - sklearn function not typed
                n_neighbors=num_neighbors,
                random_state=get_seed(),
                n_jobs=get_max_processes(),  # type: ignore - added in 1.5
            )

    # Each row asks the same directed question the class-to-factor row asks, with one
    # class against the rest standing in for the class label, so it is scored the same way:
    # a shared denominator per row, and chance correction wherever a factor is tabulable.
    # Like that row it divides by the target's entropy rather than the factor's, so it does
    # not move with a factor's bin count and `discrete_features` does not reach it.
    normalized = np.stack([
        _target_to_factor(tgt_bin[:, idx], data, coded_list, classwise_mi[idx]) for idx in range(num_classes)
    ])

    _logger.info("Mutual info classwise calculation complete: %d classes x %d factors", num_classes, num_columns - 1)

    return normalized.astype(np.float64)
