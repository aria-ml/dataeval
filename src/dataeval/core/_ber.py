__all__ = []

from typing import Any, TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dataeval._log import get_logger
from dataeval.exceptions import ShapeMismatchError
from dataeval.types import Array1D, ArrayND
from dataeval.utils._array import as_numpy
from dataeval.utils._internal import EPSILON

_logger = get_logger(__name__)

_DEVROYE_ALPHA = 0.3399
_DEVROYE_BETA = 0.9749


class BERResult(TypedDict):
    """
    Type definition for Bayes Error Rate bounds output.

    Attributes
    ----------
    upper_bound : float
        The upper bound of the Bayes Error Rate
    lower_bound : float
        The lower bound of the Bayes Error Rate
    """

    upper_bound: float
    lower_bound: float


def _validate_inputs(
    embeddings: ArrayND[float],
    class_labels: Array1D[int],
) -> tuple[NDArray[np.float32], NDArray[np.intp]]:
    embeddings = as_numpy(embeddings, dtype=np.float32)
    class_labels = as_numpy(class_labels, dtype=np.intp, required_ndim=1)

    if len(embeddings) != len(class_labels):
        raise ShapeMismatchError(
            f"Length of embeddings ({len(embeddings)}) does not match length of class_labels ({len(class_labels)}).",
        )

    return embeddings, class_labels


def _cover_hart_lower(error: ArrayLike | float, classes: int) -> NDArray[np.float64]:
    """Invert Cover & Hart (1967) elementwise: the least Bayes error an asymptotic 1-NN error allows."""
    ratio = classes / (classes - 1)
    return (1 - np.sqrt(np.clip(1 - ratio * np.asarray(error, dtype=np.float64), 0.0, 1.0))) / ratio


def _binary_knn_constant(k: int) -> float:
    """Return the tightest cited a_k with R_k <= (1 + a_k) R* for a two-class majority vote over an odd k >= 3."""
    a_k = 1 / np.sqrt(k)  # Renggli et al. (2021), FeeBee, for k > 2
    if k >= 5:  # Devroye (1981) states his constant for odd k >= 5
        a_k = min(a_k, _DEVROYE_ALPHA * np.sqrt(k) / (k - 3.25) * (1 + _DEVROYE_BETA / np.sqrt(k - 3)))
    return float(a_k)


def _knn_lowerbound(value: float, classes: int, k: int) -> float:
    """Lower bound on the Bayes error from a k-NN error, or from the 1-NN error when that is what applies.

    With ties broken toward the nearest tied neighbor an even k votes like k - 1, so it takes k - 1's constant.
    Two classes and an effective k above 1 use the k-NN constants; otherwise ``value`` must be the 1-NN error and
    Cover–Hart applies for any number of classes.
    """
    if value <= EPSILON:
        return 0.0
    k_eff = k - 1 if k % 2 == 0 else k
    if classes == 2 and k_eff > 1:
        return value / (1 + _binary_knn_constant(k_eff))
    return float(_cover_hart_lower(value, classes))


def _plurality_nearest_tie(neighbor_labels: NDArray[np.intp]) -> NDArray[np.intp]:
    """Plurality label of each row; a tie goes to the tied label whose member comes first (is nearest).

    ``neighbor_labels`` holds non-negative codes, nearest neighbor first.
    """
    rows = np.arange(len(neighbor_labels))[:, None]
    counts = np.zeros((len(neighbor_labels), int(neighbor_labels.max()) + 1), dtype=np.intp)
    np.add.at(counts, (rows, neighbor_labels), 1)
    tied = counts == counts.max(axis=1, keepdims=True)
    first = np.argmax(tied[rows, neighbor_labels], axis=1)
    return neighbor_labels[rows[:, 0], first]


def _neighbor_indices(data: NDArray[np.float32], k: int, groups: ArrayND[Any] | None) -> NDArray[np.intp]:
    """Return each sample's k nearest neighbors as an (n, k) array, from other groups when ``groups`` is given."""
    from dataeval.core._mst import compute_neighbors, grouped_neighbors

    if groups is None:
        indices = compute_neighbors(data, k=k)
        return indices[:, None] if indices.ndim == 1 else indices
    indices, _, valid = grouped_neighbors(data, groups, k)
    if not valid.all():
        raise ValueError(f"Every sample needs at least k={k} samples outside its group.")
    return indices


def _get_classes_counts(labels: NDArray[np.intp]) -> tuple[int, int]:
    classes, counts = np.unique(labels, return_counts=True)
    m = len(classes)
    if m < 2:
        raise ValueError("Label vector contains less than 2 classes!")
    n = int(np.sum(counts))
    return m, n


def ber_mst(embeddings: ArrayND[float], class_labels: Array1D[int]) -> BERResult:
    """
    Estimate Multi-class :term:`Bayes error rate<Bayes Error Rate (BER)>` using a minimum spanning tree.

    BER bounds the irreducible classification error given the current feature
    representation — the error attributable to class overlap in embedding space.
    Uses FR with a minimum spanning tree (MST) test statistic basis.

    Parameters
    ----------
    embeddings : ArrayND[float]
        Array of image :term:`embeddings<Embeddings>`. Can be an N dimensional list, or array-like object.
    class_labels : Array1D[int]
        Array of class labels for each image. Can be a 1D list, or array-like object.

    Returns
    -------
    BERResult
        Mapping with keys:

        - upper_bound: float - The upper bound of the Bayes Error Rate
        - lower_bound: float - The lower bound of the Bayes Error Rate

    Warnings
    --------
    The upper bound is not the error of any classifier, so it does not show that an accuracy is achievable; use
    :func:`ber_knn` for that. Nothing establishes the estimator's consistency when neighbors from the same group (a
    sequence, a near-duplicate cluster) are excluded, so samples that are not independent make both bounds
    unreliable.

    References
    ----------
    [1] `Learning to Bound the Multi-class Bayes Error (Th. 3 and Th. 4) <https://arxiv.org/abs/1811.06419>`_

    Examples
    --------
    >>> import sklearn.datasets as dsets
    >>> from dataeval.core import ber_mst

    >>> images, labels = dsets.make_blobs(n_samples=50, centers=2, n_features=2, random_state=0)
    >>> ber_mst(images, labels)
    {'upper_bound': 0.02, 'lower_bound': 0.010102051443364402}
    """
    _logger.info("Starting ber_mst calculation")

    from dataeval.core._mst import minimum_spanning_tree

    data_np, labels_np = _validate_inputs(embeddings, class_labels)

    m, n = _get_classes_counts(labels_np)
    _logger.debug("Number of classes: %d, Number of samples: %d", m, n)

    # Get MST
    mst_result = minimum_spanning_tree(data_np)
    source, target = mst_result["source"], mst_result["target"]
    # MST forces every group to connect, so remove the minimum number of forced connections
    mismatches = np.sum(labels_np[source] != labels_np[target]) - (m - 1)
    # BER sample scaling
    deltas = mismatches / (2 * n)
    # Get BER upper and lower values
    upper = float(2 * deltas)
    lower = float(((m - 1) / (m)) * (1 - max(1 - 2 * ((m) / (m - 1)) * deltas, 0) ** 0.5))

    _logger.info("BER_mst complete: upper_bound=%.4f, lower_bound=%.4f, mismatches=%d", upper, lower, mismatches)

    return {"upper_bound": upper, "lower_bound": lower}


def ber_knn(
    embeddings: ArrayND[float], class_labels: Array1D[int], k: int, groups: ArrayND[Any] | None = None
) -> BERResult:
    """
    Estimate Multi-class :term:`Bayes error rate<Bayes Error Rate (BER)>` using KNN.

    BER bounds the irreducible classification error given the current feature
    representation — the error attributable to class overlap in embedding space.
    Uses the k-NN classification error. The upper bound is the leave-one-out k-NN error, with plurality ties broken
    toward the nearest tied neighbor. The lower bound inverts the asymptotic relation between the k-NN error and the
    Bayes error:

    - Two classes: Cover & Hart for k = 1 (and for k = 2, which votes like k = 1); for odd k >= 3 the tighter of
      FeeBee's 1 / sqrt(k) and, for k >= 5, Devroye's constant; an even k uses k - 1's constant.
    - More than two classes: Cover & Hart applied to the 1-NN error, whatever k is.

    The lower bound is clamped to the upper bound: the k-NN error is achieved, so a lower bound above it is a
    finite-sample artifact.

    Parameters
    ----------
    embeddings : ArrayND[float]
        Array of image :term:`embeddings<Embeddings>`. Can be an N dimensional list, or array-like object.
    class_labels : Array1D[int]
        Array of class labels for each image. Can be a 1D list, or array-like object.
    k : int
        Number of nearest neighbors for KNN estimator. Should be between 1 and the number of samples.
    groups : ArrayND or None, default None
        One group id per sample. When given, neighbors come only from other groups, so near-duplicates and samples
        from the same sequence cannot vouch for each other.

    Returns
    -------
    BERResult
        Mapping with keys:

        - upper_bound: float - The upper bound of the Bayes Error Rate
        - lower_bound: float - The lower bound of the Bayes Error Rate

    Raises
    ------
    ValueError
        If ``groups`` leaves a sample fewer than ``k`` neighbors outside its group.

    References
    ----------
    [1] `Learning to Bound the Multi-class Bayes Error (Th. 3 and Th. 4) <https://arxiv.org/abs/1811.06419>`_

    [2] Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. IEEE Trans. Inf. Theory 13(1), 21-27.

    [3] Devroye, L. (1981). On the asymptotic probability of error in nonparametric discrimination. Ann. Statist.
        9(6), 1320-1327.

    [4] `Evaluating Bayes Error Estimators on Real-World Datasets with FeeBee <https://arxiv.org/abs/2108.13034>`_

    Examples
    --------
    >>> import sklearn.datasets as dsets
    >>> from dataeval.core import ber_knn

    >>> images, labels = dsets.make_blobs(n_samples=50, centers=2, n_features=2, random_state=0)
    >>> ber_knn(images, labels, 1)
    {'upper_bound': 0.04, 'lower_bound': 0.020416847668728033}
    """
    _logger.info("Starting ber_knn calculation with k=%d", k)

    data_np, labels_np = _validate_inputs(embeddings, class_labels)

    m, n = _get_classes_counts(labels_np)
    _logger.debug("Number of classes: %d, Number of samples: %d", m, n)

    codes = np.unique(labels_np, return_inverse=True)[1].reshape(-1)
    nn_indices = _neighbor_indices(data_np, k, groups)
    neighbor_codes = codes[nn_indices]
    upper = float(np.mean(_plurality_nearest_tie(neighbor_codes) != codes))
    nn_error = float(np.mean(neighbor_codes[:, 0] != codes))
    lower = _knn_lowerbound(upper, m, k) if m == 2 else _knn_lowerbound(nn_error, m, 1)
    # The k-NN error is achieved, so a lower bound above it is a finite-sample artifact that it refutes.
    lower = min(lower, upper)

    _logger.info("BER_knn complete: upper_bound=%.4f, lower_bound=%.4f", upper, lower)

    return {"upper_bound": upper, "lower_bound": lower}
