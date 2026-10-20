__all__ = []

import inspect
import warnings
from typing import Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

from dataeval._log import get_logger
from dataeval.core._parity import parity
from dataeval.exceptions import DeprecatedWarning
from dataeval.types import Array1D
from dataeval.utils._array import as_numpy

_logger = get_logger(__name__)


class LabelParityResult(TypedDict):
    """
    Type definition for label parity output.

    Attributes
    ----------
    chi_squared : float
        Pearson's chi-squared statistic of the 2 x K table of class counts, divided by the design
        effect when groups were given. NaN if the grouped test is untestable.
    p_value : float
        The p-value of the test. NaN if the grouped test is untestable.
    score : float
        The bias-corrected Cramer's V between dataset membership and class, from 0 (the two
        datasets share one class distribution) to 1 (they share no class).
    insufficient : bool
        Whether the table is too thin to trust the reference distribution (a screen, not a
        validity guarantee). True for a single class and for an untestable table.
    df_num : float
        Numerator degrees of freedom of the reference distribution.
    df_den : float
        Denominator degrees of freedom; infinite for a chi-squared reference.
    test : {"pearson", "rao_scott_f"}
        The test run: Pearson's chi-squared on independent labels, or the second-order
        Rao-Scott F when groups were given.
    """

    chi_squared: float
    p_value: float
    score: float
    insufficient: bool
    df_num: float
    df_den: float
    test: Literal["pearson", "rao_scott_f"]


def _labels(name: str, labels: Array1D[int]) -> NDArray[np.intp]:
    array = as_numpy(labels, dtype=np.intp, required_ndim=1)
    if array.size == 0:
        raise ValueError(f"No labels found in the {name} dataset")
    return array


def _group_ids(name: str, groups: Array1D[int], size: int) -> NDArray[np.intp]:
    ids = as_numpy(groups, dtype=np.intp, required_ndim=1)
    if ids.size != size:
        raise ValueError(f"{name}_groups has {ids.size} entries for {size} {name} labels.")
    return ids


def _stacked_groups(
    expected_groups: Array1D[int] | None, observed_groups: Array1D[int] | None, n_expected: int, n_observed: int
) -> NDArray[np.intp] | None:
    """Stack the groups of both datasets, keeping them distinct; None if no groups were given."""
    if (expected_groups is None) != (observed_groups is None):
        raise ValueError(
            "Pass expected_groups and observed_groups both or neither: one dataset's labels cannot be "
            "clustered while the other's are assumed independent."
        )
    if expected_groups is None or observed_groups is None:
        return None
    first = _group_ids("expected", expected_groups, n_expected)
    second = _group_ids("observed", observed_groups, n_observed)
    # Shift the observed ids above every expected id, so the two datasets never share a group.
    return np.concatenate([first, second - second.min() + first.max() + 1])


def _check_independence(grouped: bool, assume_independent: bool | None) -> None:
    """Refuse a contradictory declaration, and warn that omitting one will raise from 2.0."""
    if grouped and assume_independent:
        raise ValueError(
            "Groups and assume_independent=True were both given: the groups record the independent "
            "unit, so there is nothing to declare. Pass one or the other."
        )
    if grouped or assume_independent:
        return
    if assume_independent is False:
        raise ValueError(
            "Cannot establish independent sampling units without groups. Pass expected_groups and "
            "observed_groups (item indices for detections, sequence ids for tracking), or "
            "assume_independent=True if each label is its own independent item."
        )
    warnings.warn(
        "label_parity was called without groups or assume_independent=True. It ran as if every "
        "label were an independent item, which gives invalid p-values for detections, where one "
        "image holds several labels. From 2.0 this raises: pass expected_groups and "
        "observed_groups for detections, or assume_independent=True when each label is its own item.",
        DeprecatedWarning,
        stacklevel=3,  # the caller of label_parity
    )


def label_parity(
    expected_labels: Array1D[int],
    observed_labels: Array1D[int],
    *,
    num_classes: int | None = None,
    expected_groups: Array1D[int] | None = None,
    observed_groups: Array1D[int] | None = None,
    assume_independent: bool | None = None,
) -> LabelParityResult:
    """
    Test whether two label samples share one class distribution.

    Assess the :term:`parity<Parity>` between expected and observed label distributions with a
    two-sample test of homogeneity: the null hypothesis is that both samples were drawn from the
    same class distribution. The test is symmetric, so swapping the two samples gives the same
    result, and neither sample is treated as the known population.

    Parameters
    ----------
    expected_labels : Array1D[int]
        Class labels of the expected (reference) dataset. Can be a 1D list, or array-like object.
    observed_labels : Array1D[int]
        Class labels of the observed dataset. Can be a 1D list, or array-like object.
    num_classes : int or None, default None
        Ignored: the classes are the ones that occur in either sample.

        .. deprecated:: 1.2
            The argument is no longer needed and is removed in 2.0. Drop it.
    expected_groups : Array1D[int] or None, default None
        The independent group (image, sequence) each expected label belongs to, one id per label.
        Labels inside a group may depend on each other in any way. Required together with
        ``observed_groups`` for detection or tracking labels, where one image or sequence
        contributes many labels.

        .. versionadded:: 1.2
    observed_groups : Array1D[int] or None, default None
        The independent group each observed label belongs to. Its ids need not differ from
        ``expected_groups``: the two datasets' groups are always kept distinct.

        .. versionadded:: 1.2
    assume_independent : bool or None, default None
        Declare each label its own independent item. Pass True when no groups are given and each
        label is a whole item (classification). Contradictory alongside groups. Leaving it as
        None without groups still runs, but warns.

        From 2.0, omitting both groups and ``assume_independent=True`` raises.

        .. versionadded:: 1.2

    Returns
    -------
    LabelParityResult
        Mapping with keys:

        - chi_squared: float - Pearson's chi-squared statistic (divided by the design effect for groups)
        - p_value: float - The p-value of the test
        - score: float - The bias-corrected Cramer's V between dataset and class
        - insufficient: bool - Whether the table is too thin to trust the reference distribution
        - df_num: float - Numerator degrees of freedom of the reference distribution
        - df_den: float - Denominator degrees of freedom; infinite for a chi-squared reference
        - test: str - ``"pearson"`` or ``"rao_scott_f"``

    Raises
    ------
    ValueError
        If either label sample is empty, if only one of ``expected_groups`` and
        ``observed_groups`` is given, if a group array does not have one id per label, or if
        ``assume_independent`` contradicts the groups or is False without groups.

    Warns
    -----
    DeprecatedWarning
        If ``num_classes`` is passed, or if neither groups nor ``assume_independent=True`` is given.

    See Also
    --------
    :func:`scipy.stats.chi2_contingency`
    :func:`~dataeval.core.label_stats`

    Notes
    -----
    **The test.** The labels of both samples form a 2 x K table of class counts, K being the number
    of classes that occur in either. Without groups it is Pearson's chi-squared test of
    homogeneity without continuity correction, identical to
    :func:`scipy.stats.chi2_contingency` with ``correction=False``. With groups it is the
    second-order Rao-Scott F, which accounts for labels inside an image or sequence resembling each
    other (Rao & Scott 1984; Thomas & Rao 1987). The groups of the two datasets are independent of
    each other and distinct, however they are numbered. Dependence between groups is not corrected.

    **Detections need groups.** One image contributes many labels, and they are not independent.
    Read as independent rows, two samples with the same class distribution but different numbers
    of images and different image-size distributions were declared different 84-91% of the time at
    a 5% level; grouped by image, 4.4-4.7%. Pass the item index of each label as its group.

    **The population** is the frequency of detection labels. Images with no detections contribute
    no labels and are outside it. Image-level class *presence* is a different question, answered by
    ``image_counts_per_class`` of :func:`~dataeval.core.label_stats`.

    **The vocabulary** must be the same in both samples: equal integer codes do not make equal
    classes. Use :class:`~dataeval.data.Relabel` to conform two datasets first.

    **A single class** across both samples returns a chi-squared of 0, a p-value of 1 and a score of
    0, flagged ``insufficient``. Read it as *no shift observed in the sampled labels*, not as
    evidence of parity: with one observed class the test cannot speak to alternatives the samples
    did not show.

    A grouped table with fewer than two groups, or too few effective groups, is untestable: the
    statistic and p-value are NaN, not a verdict of parity.

    References
    ----------
    [1] On the criterion that a given system of deviations from the probable in the
        case of a correlated system of variables is such that it can be reasonably
        supposed to have arisen from random sampling.
        Pearson, K. (1900). Philosophical Magazine, 50(302), 157-175.

    [2] On chi-squared tests for multiway contingency tables with cell proportions estimated
        from survey data. Rao, J. N. K., & Scott, A. J. (1984). Annals of Statistics, 12(1), 46-60.

    [3] Small-sample comparisons of level and power for simple goodness-of-fit statistics under
        cluster sampling. Thomas, D. R., & Rao, J. N. K. (1987). JASA, 82(398), 630-636.

    Examples
    --------
    Randomly creating some label distributions using :func:`np.random.default_rng <numpy.random.default_rng>`

    >>> rng = np.random.default_rng(175)
    >>> expected_labels = rng.choice([0, 1, 2, 3, 4], (100))
    >>> observed_labels = rng.choice([2, 3, 0, 4, 1], (100))
    >>> result = label_parity(expected_labels, observed_labels, assume_independent=True)
    >>> round(result["chi_squared"], 3), round(result["p_value"], 3), result["test"]
    (5.456, 0.244, 'pearson')

    Detections from 40 images of 5 labels each, every image showing a single class and both datasets
    drawn from one distribution. Read as 400 independent labels they look shifted; grouped by image,
    they do not:

    >>> rng = np.random.default_rng(0)
    >>> expected_labels = np.repeat(rng.integers(0, 3, 40), 5)
    >>> observed_labels = np.repeat(rng.integers(0, 3, 40), 5)
    >>> image = np.repeat(np.arange(40), 5)
    >>> rows = label_parity(expected_labels, observed_labels, assume_independent=True)
    >>> grouped = label_parity(expected_labels, observed_labels, expected_groups=image, observed_groups=image)
    >>> round(rows["p_value"], 3), round(grouped["p_value"], 3), grouped["test"]
    (0.004, 0.338, 'rao_scott_f')
    """
    _logger.info("Starting label_parity calculation")

    expected = _labels("expected", expected_labels)
    observed = _labels("observed", observed_labels)
    groups = _stacked_groups(expected_groups, observed_groups, expected.size, observed.size)
    _check_independence(groups is not None, assume_independent)
    if num_classes is not None:
        warnings.warn(
            "label_parity's num_classes is ignored: the classes are the ones that occur in either "
            "dataset. It will be removed in 2.0.",
            DeprecatedWarning,
            stacklevel=2,
        )

    labels = np.concatenate([expected, observed])
    test: Literal["pearson", "rao_scott_f"] = "pearson" if groups is None else "rao_scott_f"
    if np.unique(labels).size < 2:
        # No shift observed, and nothing to test: `parity` calls a single class untestable.
        return LabelParityResult(
            chi_squared=0.0,
            p_value=1.0,
            score=0.0,
            insufficient=True,
            df_num=0.0,
            df_den=float("inf"),
            test=test,
        )

    membership = np.repeat([0, 1], [expected.size, observed.size])[:, None]
    # `parity` is experimental but this function is stable, so its callers did not opt in. Call the
    # undecorated function: silencing the warning instead would still spend the decorator's one
    # notice per process, and a later direct call to `parity` would never see it.
    result = inspect.unwrap(parity)(membership, labels, groups, assume_independent=groups is None)

    _logger.info(
        "Label parity calculation complete: chi_squared=%.4f, p_value=%.4f",
        result["chi_squared"][0],
        result["p_values"][0],
    )

    return LabelParityResult(
        chi_squared=float(result["chi_squared"][0]),
        p_value=float(result["p_values"][0]),
        score=float(result["scores"][0]),
        insufficient=bool(result["insufficient"][0]),
        df_num=float(result["df_num"][0]),
        df_den=float(result["df_den"][0]),
        test=result["test"],
    )
