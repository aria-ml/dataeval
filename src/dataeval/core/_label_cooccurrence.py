__all__ = []

from collections.abc import Iterable, Mapping, Sequence
from itertools import chain
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.stats import chi2

from dataeval._log import get_logger
from dataeval.core._association import benjamini_hochberg, bias_corrected_cramers_v, cochran_breach_2x2
from dataeval.core._label_stats import LabelStatsResult, label_stats

_logger = get_logger(__name__)

# The keys of a LabelStatsResult that label_cooccurrence reads; everything else it needs
# (images per class, labeled images) is derived from ``classes_per_image``.
_REQUIRED_STATS_KEYS = ("classes_per_image", "index2label", "image_count")


class LabelCooccurrencePairs(TypedDict):
    """
    Statistics for each pair of classes that appear together in at least one image.

    Every array has one entry per pair, ordered by ``(class_a, class_b)`` with
    ``class_a < class_b``. Pairs of classes that never appear together are not listed.
    ``N`` below is the number of images with at least one label: images without labels say
    nothing about which classes appear together, and counting them would make any two
    classes look associated.

    Attributes
    ----------
    class_a : NDArray[np.intp]
        The first class of each pair.
    class_b : NDArray[np.intp]
        The second class of each pair.
    counts : NDArray[np.intp]
        Number of images containing both classes.
    expected : NDArray[np.float64]
        Number of images expected to contain both classes if they were independent,
        ``n_a * n_b / N``.
    lift : NDArray[np.float64]
        ``counts / expected``. Above 1, the classes appear together more often than
        chance; below 1, less often.
    jaccard : NDArray[np.float64]
        ``counts / (n_a + n_b - counts)``, the share of images containing either class
        that contain both. Symmetric and bounded in [0, 1].
    p_b_given_a : NDArray[np.float64]
        Probability that class b is present given that class a is.
    p_a_given_b : NDArray[np.float64]
        Probability that class a is present given that class b is.
    scores : NDArray[np.float64]
        Signed, bias-corrected Cramér's V of the 2x2 presence table, from -1 to 1. The
        magnitude is on the same scale as :func:`~dataeval.core.parity` and the sign is
        the direction of the association: positive when the classes appear together more
        often than chance, negative when less often.
    p_values : NDArray[np.float64]
        P-value of Pearson's chi-square test of independence on the 2x2 table.
    p_values_adjusted : NDArray[np.float64]
        ``p_values`` adjusted to control the false discovery rate across every pair of
        classes present in the data (Benjamini-Hochberg), not only those listed here.
    insufficient : NDArray[np.bool_]
        True where the 2x2 table breaches Cochran's criterion, so the chi-square
        p-value should not be trusted.
    """

    class_a: NDArray[np.intp]
    class_b: NDArray[np.intp]
    counts: NDArray[np.intp]
    expected: NDArray[np.float64]
    lift: NDArray[np.float64]
    jaccard: NDArray[np.float64]
    p_b_given_a: NDArray[np.float64]
    p_a_given_b: NDArray[np.float64]
    scores: NDArray[np.float64]
    p_values: NDArray[np.float64]
    p_values_adjusted: NDArray[np.float64]
    insufficient: NDArray[np.bool_]


class LabelCooccurrenceResult(TypedDict):
    """
    Type definition for label co-occurrence output.

    Attributes
    ----------
    pairs : LabelCooccurrencePairs
        Statistics for each pair of classes that appear together.
    partner_counts : Mapping[int, int]
        Mapping from class to the number of distinct other classes that appear with that
        class in at least one image.
    isolated_classes : Sequence[int]
        Sorted classes that appear in the dataset but always appear alone.
    classes : Sequence[int] | None
        None unless ``return_matrix`` was set. The class labels, in the order of the rows
        and columns of the matrices. Every class in the ``index2label`` of the statistics
        is included, even if it has no instances.
    cooccurrence_matrix : NDArray[np.intp] | None
        None unless ``return_matrix`` was set. Matrix of co-occurrence counts with shape
        (n_classes, n_classes), where element (i, j) is the number of images containing both
        class i and class j. The diagonal is the number of images containing each class.
    cooccurrence_probability : NDArray[np.float64] | None
        Matrix of joint probabilities P(i AND j) with shape (n_classes, n_classes), equal
        to ``cooccurrence_matrix / image_count``. It is the share of *all* images, empty
        ones included.
    conditional_probability : NDArray[np.float64] | None
        Matrix of conditional probabilities P(j | i) with shape (n_classes, n_classes),
        the probability of class j being present given that class i is. Rows of classes
        that never appear are all zero.
    n_tests : int
        Number of class pairs, over the classes that appear, that the false discovery rate
        is controlled across.
    """

    pairs: LabelCooccurrencePairs
    partner_counts: Mapping[int, int]
    isolated_classes: Sequence[int]
    classes: Sequence[int] | None
    cooccurrence_matrix: NDArray[np.intp] | None
    cooccurrence_probability: NDArray[np.float64] | None
    conditional_probability: NDArray[np.float64] | None
    n_tests: int


def _resolve_stats(
    class_labels: LabelStatsResult | Iterable[int],
    item_indices: Iterable[int] | None,
    index2label: Mapping[int, str] | None,
    image_count: int | None,
) -> LabelStatsResult:
    """Return the label statistics, computing them unless a result was passed in."""
    if not isinstance(class_labels, Mapping):
        return label_stats(class_labels, item_indices, index2label, image_count)

    given = [
        name
        for name, value in (("item_indices", item_indices), ("index2label", index2label), ("image_count", image_count))
        if value is not None
    ]
    if given:
        raise ValueError(
            f"{', '.join(given)} cannot be combined with a LabelStatsResult, which already holds them. "
            "Pass either a LabelStatsResult or the raw labels.",
        )
    missing = [key for key in _REQUIRED_STATS_KEYS if key not in class_labels]
    if missing:
        raise ValueError(f"The LabelStatsResult is missing the required key(s) {missing}.")
    stats: Any = class_labels
    n_entries = len(stats["classes_per_image"])
    if n_entries != stats["image_count"]:
        raise ValueError(f"classes_per_image has {n_entries} entries but image_count is {stats['image_count']}.")
    return stats


def _incidence_matrix(classes_per_image: Sequence[Sequence[int]], classes: Sequence[int]) -> sparse.csr_matrix:
    """
    Build the binary image-by-class presence matrix.

    Duplicate and unsorted classes within an image are tolerated, so a hand-built
    ``classes_per_image`` gives the same result as one from ``label_stats``.
    """
    n_images = len(classes_per_image)
    lengths = np.fromiter(map(len, classes_per_image), dtype=np.int64, count=n_images)
    flat = np.fromiter(chain.from_iterable(classes_per_image), dtype=np.int64, count=int(lengths.sum()))
    class_array = np.asarray(classes, dtype=np.int64)  # sorted, as taken from index2label

    columns = np.searchsorted(class_array, flat)
    if flat.size and (class_array.size == 0 or np.any(class_array[np.minimum(columns, class_array.size - 1)] != flat)):
        unknown = sorted(set(flat.tolist()) - set(classes))
        raise ValueError(f"classes_per_image holds classes missing from index2label: {unknown}")

    indptr = np.concatenate(([0], np.cumsum(lengths)))
    incidence = sparse.csr_matrix(
        (np.ones(flat.size, dtype=np.int64), columns, indptr),
        shape=(n_images, class_array.size),
    )
    incidence.sum_duplicates()
    incidence.data[:] = 1
    return incidence


def _pair_statistics(
    counts: sparse.csr_matrix,
    classes: Sequence[int],
    class_images: NDArray[np.intp],
    n_labeled: int,
) -> tuple[LabelCooccurrencePairs, int]:
    """Compute the 2x2 association statistics of every pair of classes that co-occur."""
    upper = sparse.triu(counts, k=1, format="coo")
    order = np.lexsort((upper.col, upper.row))
    rows, cols = upper.row[order], upper.col[order]
    both = upper.data[order].astype(np.float64)

    n = float(n_labeled)
    n_a = class_images[rows].astype(np.float64)
    n_b = class_images[cols].astype(np.float64)

    # 2x2 presence table of (a present, b present): cells both, a-both, b-both, N-a-b+both,
    # whose determinant simplifies to both * N - a * b.
    determinant = both * n - n_a * n_b
    denominator = n_a * (n - n_a) * n_b * (n - n_b)
    with np.errstate(divide="ignore", invalid="ignore"):
        chi_squared = np.where(denominator > 0, n * determinant**2 / denominator, 0.0)
        expected = n_a * n_b / n
        lift = both / expected
    p_values = chi2.sf(chi_squared, 1)

    n_present = int(np.count_nonzero(class_images))
    n_tests = n_present * (n_present - 1) // 2
    class_array = np.asarray(classes, dtype=np.intp)

    pairs = LabelCooccurrencePairs(
        class_a=class_array[rows],
        class_b=class_array[cols],
        counts=both.astype(np.intp),
        expected=expected,
        lift=lift,
        jaccard=both / (n_a + n_b - both),
        p_b_given_a=both / n_a,
        p_a_given_b=both / n_b,
        scores=np.sign(determinant) * bias_corrected_cramers_v(chi_squared, n, 2, 2),
        p_values=p_values,
        p_values_adjusted=benjamini_hochberg(p_values, n_tests),
        insufficient=cochran_breach_2x2(n_a, n_b, n),
    )
    return pairs, n_tests


def label_cooccurrence(
    class_labels: LabelStatsResult | Iterable[int],
    item_indices: Iterable[int] | None = None,
    index2label: Mapping[int, str] | None = None,
    image_count: int | None = None,
    return_matrix: bool = False,
) -> LabelCooccurrenceResult:
    """
    Calculate label co-occurrence and the interaction statistics between classes.

    Computes which labels tend to appear together in the same images,
    including co-occurrence counts, joint probabilities, conditional
    probabilities, Jaccard similarity, Lift, Cochran's sufficiency criterion,
    partner counts, and isolated classes.

    Parameters
    ----------
    class_labels : LabelStatsResult | Iterable[int]
        Either a precomputed :class:`~dataeval.core.LabelStatsResult` dictionary, or an
        iterable of integer class labels (one per annotation/target).
    item_indices : Iterable[int] | None, optional
        A sequence mapping each label to its source image index. If None and
        ``class_labels`` is not a LabelStatsResult, a 1:1 mapping is assumed (one label
        per image). Must not be given together with a LabelStatsResult.
    index2label : Mapping[int, str] | None, optional
        Mapping from label integers to human-readable names. Defines the complete set of
        classes reported in the matrix dimensions, even those with no instances. Must not
        be given together with a LabelStatsResult.
    image_count : int | None, optional
        Total number of images, including those without labels. Pass it when images
        without labels exist after the last labeled one. Must not be given together
        with a LabelStatsResult.
    return_matrix : bool, default False
        Whether to build the dense ``classes``, ``cooccurrence_matrix``,
        ``cooccurrence_probability`` and ``conditional_probability`` outputs. They grow
        with the square of the number of classes, so they are None unless requested.

    Returns
    -------
    LabelCooccurrenceResult
        Dictionary containing co-occurrence statistics with keys:

        - classes: Sequence[int] | None - Class labels in the order of the matrix rows and columns
        - cooccurrence_matrix: NDArray[np.intp] | None - Matrix of joint co-occurrence counts
        - cooccurrence_probability: NDArray[np.float64] | None - Matrix of joint probabilities P(i AND j)
        - conditional_probability: NDArray[np.float64] | None - Matrix of conditional probabilities P(j | i)
        - partner_counts: Mapping[int, int] - Number of other classes co-occurring with each class
        - isolated_classes: Sequence[int] - Classes that only appear alone in images
        - pairs: LabelCooccurrencePairs - Association statistics of each pair that co-occurs
        - n_tests: int - Number of class pairs the false discovery rate is controlled across

    Raises
    ------
    ValueError
        If a LabelStatsResult is combined with ``item_indices``, ``index2label`` or
        ``image_count``, lacks a required key, or is inconsistent (``classes_per_image``
        holds classes missing from ``index2label``, or its length differs from
        ``image_count``). Errors from :func:`~dataeval.core.label_stats` otherwise.

    Notes
    -----
    The association statistics treat each pair of classes as a 2x2 table of which images
    contain neither, either, or both, over the images that have at least one label. The
    sign of ``scores`` gives the direction of the association, which Cramér's V alone does
    not. Pairs that never appear together are not listed, so mutually exclusive classes are
    not reported, and every p-value is adjusted for all pairs of classes that appear, not
    just those listed.

    The matrices have one row and column per class, so they grow with the square of the
    number of classes. Imbalance ratios and SCUMBLE are reported by
    :func:`~dataeval.core.label_stats`.

    References
    ----------
    Bergsma, W. (2013). A bias-correction for Cramér's V and Tschuprow's T.
    Journal of the Korean Statistical Society, 42(3), 323-328.

    Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical
    and powerful approach to multiple testing. Journal of the Royal Statistical Society:
    Series B, 57(1), 289-300.

    Charte, F., Rivera, A. J., del Jesus, M. J., & Herrera, F. (2019). Dealing with difficult
    minority labels in imbalanced multilabel data sets. Neurocomputing, 326-327, 39-53.

    Examples
    --------
    Calculate label co-occurrence from annotations:

    >>> from dataeval.core import label_cooccurrence
    >>> labels = [0, 0, 1, 1, 2, 0, 1, 2, 4, 4]
    >>> item_indices = [0, 0, 0, 1, 1, 3, 3, 3, 4, 4]
    >>> index2label = {0: "horse", 1: "cow", 2: "sheep", 3: "pig", 4: "cat", 5: "dog"}
    >>> cooc = label_cooccurrence(labels, item_indices, index2label, image_count=5, return_matrix=True)
    >>> cooc["classes"]
    [0, 1, 2, 3, 4, 5]
    >>> cooc["cooccurrence_matrix"].shape
    (6, 6)
    >>> int(cooc["cooccurrence_matrix"][0, 1])  # horse and cow together
    2
    >>> cooc["partner_counts"][0]  # horse appears with cow and sheep
    2
    >>> cooc["isolated_classes"]  # cat only ever appears alone
    [4]

    Each pair that appears together, with how much more often than chance:

    >>> pairs = cooc["pairs"]
    >>> list(zip(pairs["class_a"].tolist(), pairs["class_b"].tolist()))
    [(0, 1), (0, 2), (1, 2)]
    >>> pairs["counts"].tolist()
    [2, 1, 2]
    >>> np.round(pairs["lift"], 3).tolist()
    [1.333, 1.0, 1.333]

    Four classes appear, so the false discovery rate is controlled across their six
    possible pairs, although only three co-occur:

    >>> cooc["n_tests"]
    6

    Calculate label co-occurrence from precomputed label_stats:

    >>> from dataeval.core import label_stats
    >>> stats = label_stats(labels, item_indices, index2label, image_count=5)
    >>> cooc = label_cooccurrence(stats)
    >>> cooc["isolated_classes"]
    [4]
    """
    _logger.info("Starting label_cooccurrence calculation")

    stats = _resolve_stats(class_labels, item_indices, index2label, image_count)
    classes = [int(cls) for cls in sorted(stats["index2label"])]
    incidence = _incidence_matrix(stats["classes_per_image"], classes)
    n_images = incidence.shape[0]  # type: ignore
    n_labeled = int(np.count_nonzero(np.diff(incidence.indptr)))

    counts = (incidence.T @ incidence).tocsr()
    class_images = np.asarray(counts.diagonal(), dtype=np.intp)

    partners = counts.getnnz(axis=1) - (class_images > 0)
    partner_counts = dict(zip(classes, partners.tolist(), strict=True))
    isolated_classes = [cls for cls, n, p in zip(classes, class_images, partners, strict=True) if n > 0 and p == 0]

    pairs, n_tests = _pair_statistics(counts, classes, class_images, n_labeled)

    matrix = None
    joint = None
    conditional = None
    if return_matrix:
        matrix = counts.toarray().astype(np.intp, copy=False)

        joint = matrix / n_images if n_images > 0 else np.zeros(matrix.shape, dtype=np.float64)
        denominator = class_images[:, None]
        conditional = np.divide(
            matrix, denominator, out=np.zeros(matrix.shape, dtype=np.float64), where=denominator > 0
        )

    _logger.info(
        "Label co-occurrence calculation complete: %d classes, %d co-occurring pairs, %d isolated classes",
        len(classes),
        len(pairs["counts"]),
        len(isolated_classes),
    )

    return {
        "pairs": pairs,
        "partner_counts": partner_counts,
        "isolated_classes": isolated_classes,
        "classes": classes if return_matrix else None,
        "cooccurrence_matrix": matrix,
        "cooccurrence_probability": joint,
        "conditional_probability": conditional,
        "n_tests": n_tests,
    }
