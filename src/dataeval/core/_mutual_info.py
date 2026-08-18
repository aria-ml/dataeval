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
from dataeval.core._bin import is_continuous
from dataeval.types import Array1D, Array2D
from dataeval.utils._internal import as_numpy, opt_as_numpy

_logger = get_logger(__name__)


class MutualInfoResult(TypedDict):
    """
    Type definition for normalized mutual information output.

    Attributes
    ----------
    class_to_factor : NDArray[np.float64]
        1D array of length (num_factors+1) holding the share of the class label's entropy
        that each factor accounts for, corrected for chance, with the class label itself at
        index 0. 1.0 means the factor determines the class outright and 0.0 means it says
        nothing beyond what its cardinality would produce by chance. The entries are
        comparable to each other while the factors are of similar cardinality; a factor
        taking several hundred values is scored conservatively and can fall below a
        coarser factor that accounts for less. See Notes.
    interfactor : NDArray[np.float64]
        (num_factors) x (num_factors) symmetric matrix of normalized mutual information
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


def _adjusted_association(first: NDArray[Any], second: NDArray[Any]) -> float:
    """
    Association between two discretized factors, corrected for chance.

    The chance correction is the one ``_adjusted_share`` applies, over a different
    denominator. Neither factor here is the thing being explained, so there is no reason
    to divide by one of the two entropies rather than the other; the smaller of them is
    the most either could account for, and it is what the pair is scored against. The
    class-to-factor direction is not symmetric in that way and uses the class entropy
    alone, so a value from that row and a value from this block answer different
    questions and are not comparable to each other.
    """
    ceiling = min(_entropy_of(first), _entropy_of(second))
    contingency = contingency_matrix(first, second, sparse=True)
    observed = float(mutual_info_score(None, None, contingency=contingency))
    expected = float(expected_mutual_information(contingency, first.shape[0]))
    denominator = ceiling - expected
    # A constant factor has no entropy to share, and a pair of near-identifier columns
    # collapses both sides of the ratio to rounding error; see ``_adjusted_share``.
    if denominator <= ceiling * 1e-6:
        return 0.0
    return float(np.clip((observed - expected) / denominator, 0.0, 1.0))


def _target_to_factor(
    target: NDArray[Any],
    data: NDArray[np.intp],
    declared_list: list[bool],
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

    Discretized factors are additionally corrected for chance; see ``_adjusted_share``.
    A factor arriving unbinned has no contingency table to take that expectation over, so
    it is scored on the estimated mutual information alone.

    ``declared_list`` is the discreteness the caller declared, not the one
    ``_merge_labels_and_factors`` hands sklearn: a column of distinct values is
    presented to the estimator as continuous, but it is exactly the identifier case the
    chance correction exists for, so the branch here is chosen before that substitution.
    """
    row = np.zeros(len(declared_list), dtype=np.float64)
    row[0] = 1.0  # the target against itself
    target_entropy = _entropy_of(target)
    if target_entropy <= 0:
        # A target taking one value carries no uncertainty for a factor to explain.
        return row
    for j in range(1, len(declared_list)):
        share = (
            _adjusted_share(target, data[:, j], target_entropy)
            if declared_list[j]
            else float(np.clip(float(raw_mi[j]) / target_entropy, 0.0, 1.0))
        )
        row[j] = share
    return row


def _merge_labels_and_factors(
    class_labels: NDArray[np.intp],
    factor_data: NDArray[np.intp],
    discrete_features: Iterable[bool] | None,
) -> tuple[NDArray[np.intp], list[bool], list[bool]]:
    """Stack the label axis onto the factors and say which columns are discrete.

    Returns the stacked data, the list handed to sklearn, and the list as declared. The
    two differ only for a column of all-distinct values: sklearn is told it is continuous
    so the estimator does not treat every value as its own category, while the declared
    list keeps the caller's word so the chance correction still applies to it.
    """
    declared_list = [True] + (
        [not is_continuous(d) for d in factor_data.T] if discrete_features is None else list(discrete_features)
    )

    # Use numeric data for MI
    data = np.hstack((class_labels[:, np.newaxis], factor_data))
    # Present discrete features composed of distinct values as continuous for `mutual_info_classif`
    discrete_list = list(declared_list)
    for i in range(len(discrete_list)):
        if len(data) == len(np.unique(data[:, i])):
            discrete_list[i] = False

    return data, discrete_list, declared_list


def mutual_info(  # noqa: C901
    class_labels: Array1D[int],
    factor_data: Array2D[int | float],
    discrete_features: Array1D[bool] | None = None,
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
    discrete_features : Array1D[bool] | None, shape - (F,), default None
        Boolean array defining whether or not the feature set is discretized.
        Can be a 1D list, or array-like object.
    num_neighbors : int, default 5
        Number of points to consider as neighbors.

    Returns
    -------
    MutualInfoResult
        TypedDict containing:

        - class_to_factor: NDArray[np.float64] - 1D array of normalized MI between class labels and each factor
        - interfactor: NDArray[np.float64] - (num_factors) x (num_factors) matrix of normalized MI between factors only

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

    The two halves of the result answer different questions and are normalized
    accordingly. Between two factors neither side is privileged, so that block is divided
    by the smaller of the two entropies, and pairs involving a variable left continuous
    use the Linfoot transformation instead, there being no upper limit to the entropy of a
    continuous distribution to divide by. The class-to-factor row is directed -- it asks
    how much of the class label a factor accounts for -- so every entry is divided by the
    class entropy alone. Dividing each factor by its own entropy there would rank a factor
    by how few categories it has as much as by how much it explains.

    That row is also corrected for chance. Mutual information read off a contingency table
    rises with the number of categories even when the factor and the class are
    independent, enough that an identifier column can outrank a genuine effect; see
    ``_adjusted_share``.

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
    >>> result = mutual_info(class_labels=class_labels, factor_data=factor_data)

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
    discrete_feat_np = opt_as_numpy(discrete_features, dtype=np.bool_, required_ndim=1)

    _logger.debug("Input shapes: class_labels=%s, factor_data=%s", class_labels_np.shape, factor_data_np.shape)

    num_neighbors = _validate_num_neighbors(num_neighbors)
    data, discrete_list, declared_list = _merge_labels_and_factors(class_labels_np, factor_data_np, discrete_feat_np)
    num_factors = len(discrete_list)

    _logger.debug("Computing NMI for %d factors (%d discrete)", num_factors, sum(discrete_list))

    # initialize output matrix
    mi = np.full((num_factors, num_factors), np.nan, dtype=np.float32)

    # pre-compute normalization factor and use it for discrete-discrete continuous-discrete cases.
    norm_factor = np.zeros(len(discrete_list))
    for i in range(len(discrete_list)):
        if not discrete_list[i]:
            # Ensure that bogus entropies from a continuous variable will not be chosen ever.
            norm_factor[i] = np.inf
        else:
            norm_factor[i] = _entropy_of(data[:, i])

    # Only the factor-to-factor block of `mi` is returned, so row 0 is needed solely to
    # score a factor the caller declared continuous, which is the one case the class row
    # cannot read off a contingency table. With every factor declared discrete it is a
    # whole estimator pass over the data whose result nothing reads.
    row_zero_is_read = not all(declared_list[1:])
    for idx, is_discrete in enumerate(discrete_list):
        if idx == 0 and not row_zero_is_read:
            mi[idx, :] = 0.0
            continue
        mi[idx, :] = (mutual_info_classif if is_discrete else mutual_info_regression)(
            data,
            data[:, idx],
            discrete_features=discrete_list,  # type: ignore - sklearn function not typed
            n_neighbors=num_neighbors,
            random_state=get_seed(),
            n_jobs=get_max_processes(),  # type: ignore - added in 1.5
        )

    # Estimated mutual information in nats, kept before normalization because the
    # class-to-factor row is normalized differently from the factor-to-factor block.
    raw_mi = mi[0].copy()

    for idx, is_discrete in enumerate(discrete_list):
        # Normalization via entropy, pre-computed above
        for j in range(data.shape[1]):
            if discrete_list[j] or is_discrete:
                if norm_factor[j] == 0 or norm_factor[idx] == 0:
                    mi[idx, j] = 0.0
                else:
                    mi[idx, j] /= min(norm_factor[j], norm_factor[idx])
            else:
                mi[idx, j] = 1.0 - np.exp(-2.0 * float(mi[idx, j]))  # Linfoot transformation, mi in nats

    full_matrix = 0.5 * (mi + mi.T).astype(np.float64)
    interfactor = full_matrix[1:, 1:]

    # Every pair with a contingency table behind it is corrected for chance, on the same
    # grounds as the class row: plug-in mutual information rises with cardinality even
    # under independence, and this block carries a correlation threshold, so an uncorrected
    # value turns a finely binned factor into a reported correlation with everything. Pairs
    # involving a factor the caller declared continuous have no such table and keep the
    # estimator's own normalization above.
    for i in range(1, num_factors):
        if not declared_list[i]:
            continue
        for j in range(i + 1, num_factors):
            if not declared_list[j]:
                continue
            adjusted = _adjusted_association(data[:, i], data[:, j])
            interfactor[i - 1, j - 1] = interfactor[j - 1, i - 1] = adjusted

    # Between two factors neither side is privileged, so that block is scored against the
    # smaller of the two entropies. The class-to-factor row asks a directed question --
    # how much of the class label does this factor account for -- and is scored accordingly.
    class_to_factor = _target_to_factor(data[:, 0], data, declared_list, raw_mi)

    _logger.info(
        "Mutual info calculation complete: %d factors, mean class_to_factor NMI=%.4f",
        num_factors - 1,
        np.mean(class_to_factor[1:]),
    )

    return MutualInfoResult(
        class_to_factor=class_to_factor,
        interfactor=interfactor,
    )


def mutual_info_classwise(
    class_labels: Array1D[int],
    factor_data: Array2D[int | float],
    discrete_features: Array1D[bool] | None = None,
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
    discrete_features : Array1D[bool] | None, shape - (F,), default None
        Boolean array defining whether or not the feature set is discretized.
        Can be a 1D list, or array-like object.
    num_neighbors : int, default 5
        Number of points to consider as neighbors.

    Returns
    -------
    NDArray[np.float64]
        (num_classes) x (num_factors+1) array holding, for each class taken against the
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
    discrete_feat_np = opt_as_numpy(discrete_features, dtype=np.bool_, required_ndim=1)

    num_neighbors = _validate_num_neighbors(num_neighbors)
    data, discrete_list, declared_list = _merge_labels_and_factors(class_labels_np, factor_data_np, discrete_feat_np)
    num_factors = len(discrete_list)
    u_classes = np.unique(class_labels_np)
    num_classes = len(u_classes)

    _logger.debug("Computing classwise NMI for %d classes and %d factors", num_classes, num_factors)

    if num_classes == 0:
        # No class to take against the rest, so there is no row to score. Returned rather
        # than stacked: np.stack has no shape to infer from an empty list.
        return np.zeros((0, num_factors), dtype=np.float64)

    # classwise targets (binary indicators)
    tgt_bin = data[:, 0][:, None] == u_classes

    # Compute MI. The estimate is read only where a factor was declared continuous, since
    # every other column is scored off its contingency table instead; with none of them
    # declared continuous the whole loop is an estimator pass nothing reads.
    classwise_mi = np.zeros((num_classes, num_factors), dtype=np.float32)
    if not all(declared_list[1:]):
        for idx in range(num_classes):
            classwise_mi[idx, :] = mutual_info_classif(
                data,
                tgt_bin[:, idx],
                discrete_features=discrete_list,  # type: ignore - sklearn function not typed
                n_neighbors=num_neighbors,
                random_state=get_seed(),
                n_jobs=get_max_processes(),  # type: ignore - added in 1.5
            )

    # Each row asks the same directed question the class-to-factor row asks, with one
    # class against the rest standing in for the class label, so it is scored the same way:
    # a shared denominator per row, and chance correction wherever a factor is discretized.
    normalized = np.stack([
        _target_to_factor(tgt_bin[:, idx], data, declared_list, classwise_mi[idx]) for idx in range(num_classes)
    ])

    _logger.info("Mutual info classwise calculation complete: %d classes x %d factors", num_classes, num_factors)

    return normalized.astype(np.float64)
