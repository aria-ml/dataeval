import math
import warnings
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import xxhash as xxh
from numpy.typing import NDArray
from PIL import Image
from scipy.fftpack import dct
from scipy.sparse import coo_matrix
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2_contingency, mode
from scipy.stats import entropy as sp_entropy
from sklearn.metrics import average_precision_score

from dataeval._internal.metrics.utils import (
    compute_neighbors,
    get_classes_counts,
    minimum_spanning_tree,
    normalize_image_shape,
    rescale,
)

HASH_SIZE = 8
MAX_FACTOR = 4


def ber_mst(X: NDArray, y: NDArray, _: int) -> Tuple[float, float]:
    """Calculates the Bayes Error Rate using a minimum spanning tree

    Parameters
    ----------
    X : NDArray, shape - (N, ... )
        n_samples containing n_features
    y : NDArray, shape - (N, 1)
        Labels corresponding to each sample

    Returns
    -------
    Tuple[float, float]
        The upper and lower bounds of the bayes error rate
    """
    M, N = get_classes_counts(y)

    tree = coo_matrix(minimum_spanning_tree(X))
    matches = np.sum([y[tree.row[i]] != y[tree.col[i]] for i in range(N - 1)])
    deltas = matches / (2 * N)
    upper = 2 * deltas
    lower = ((M - 1) / (M)) * (1 - max(1 - 2 * ((M) / (M - 1)) * deltas, 0) ** 0.5)
    return upper, lower


def ber_knn(X: NDArray, y: NDArray, k: int) -> Tuple[float, float]:
    """Calculates the Bayes Error Rate using K-nearest neighbors

    Parameters
    ----------
    X : NDArray, shape - (N, ... )
        n_samples containing n_features
    y : NDArray, shape - (N, 1)
        Labels corresponding to each sample

    Returns
    -------
    Tuple[float, float]
        The upper and lower bounds of the bayes error rate
    """
    M, N = get_classes_counts(y)

    # All features belong on second dimension
    X = X.reshape((X.shape[0], -1))
    nn_indices = compute_neighbors(X, X, k=k)
    nn_indices = np.expand_dims(nn_indices, axis=1) if nn_indices.ndim == 1 else nn_indices
    modal_class = mode(y[nn_indices], axis=1, keepdims=True).mode.squeeze()
    upper = float(np.count_nonzero(modal_class - y) / N)
    lower = knn_lowerbound(upper, M, k)
    return upper, lower


def knn_lowerbound(value: float, classes: int, k: int) -> float:
    """Several cases for computing the BER lower bound"""
    if value <= 1e-10:
        return 0.0

    if classes == 2 and k != 1:
        if k > 5:
            # Property 2 (Devroye, 1981) cited in Snoopy paper, not in snoopy repo
            alpha = 0.3399
            beta = 0.9749
            a_k = alpha * np.sqrt(k) / (k - 3.25) * (1 + beta / (np.sqrt(k - 3)))
            return value / (1 + a_k)
        if k > 2:
            return value / (1 + (1 / np.sqrt(k)))
        # k == 2:
        return value / 2

    return ((classes - 1) / classes) * (1 - np.sqrt(max(0, 1 - ((classes / (classes - 1)) * value))))


def coverage(
    embeddings: np.ndarray,
    radius_type: Literal["adaptive", "naive"] = "adaptive",
    k: int = 20,
    percent: np.float64 = np.float64(0.01),
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Perform a one-way chi-squared test between observation frequencies and expected frequencies that
    tests the null hypothesis that the observed data has the expected frequencies.

    Parameters
    ----------
    embeddings : ArrayLike, shape - (N, P)
        A dataset in an ArrayLike format.
        Function expects the data to have 2 dimensions, N number of observations in a P-dimesionial space.
    radius_type : Literal["adaptive", "naive"], default "adaptive"
        The function used to determine radius.
    k: int, default 20
        Number of observations required in order to be covered.
        [1] suggests that a minimum of 20-50 samples is necessary.
    percent: np.float64, default np.float(0.01)
        Percent of observations to be considered uncovered. Only applies to adaptive radius.

    Returns
    -------
    np.ndarray
        Array of uncovered indices
    np.ndarray
        Array of critical value radii
    float
        Radius for coverage

    Raises
    ------
    ValueError
        If length of embeddings is less than or equal to k
    ValueError
        If radius_type is unknown

    Note
    ----
    Embeddings should be on the unit interval.

    Reference
    ---------
    This implementation is based on https://dl.acm.org/doi/abs/10.1145/3448016.3457315.
    [1] Seymour Sudman. 1976. Applied sampling. Academic Press New York (1976).
    """

    # Calculate distance matrix, look at the (k+1)th farthest neighbor for each image.
    n = len(embeddings)
    if n <= k:
        raise ValueError("Number of observations less than or equal to the specified number of neighbors.")
    mat = squareform(pdist(embeddings))
    sorted_dists = np.sort(mat, axis=1)
    crit = sorted_dists[:, k + 1]

    d = np.shape(embeddings)[1]
    if radius_type == "naive":
        rho = (1 / math.sqrt(math.pi)) * ((2 * k * math.gamma(d / 2 + 1)) / (n)) ** (1 / d)
        pvals = np.where(crit > rho)[0]
    elif radius_type == "adaptive":
        # Use data adaptive cutoff as rho
        rho = int(n * percent)
        pvals = np.argsort(crit)[::-1][:rho]
    else:
        raise ValueError("Invalid radius type.")
    return pvals, crit, rho


def divergence_mst(data: np.ndarray, labels: np.ndarray) -> int:
    mst = minimum_spanning_tree(data).toarray()
    edgelist = np.transpose(np.nonzero(mst))
    errors = np.sum(labels[edgelist[:, 0]] != labels[edgelist[:, 1]])
    return errors


def divergence_fnn(data: np.ndarray, labels: np.ndarray) -> int:
    nn_indices = compute_neighbors(data, data)
    errors = np.sum(np.abs(labels[nn_indices] - labels))
    return errors


def pchash(image: np.ndarray) -> str:
    """
    Performs a perceptual hash on an image by resizing to a square NxN image
    using the Lanczos algorithm where N is 32x32 or the largest multiple of
    8 that is smaller than the input image dimensions.  The resampled image
    is compressed using a discrete cosine transform and the lowest frequency
    component is encoded as a bit array of greater or less than median value
    and returned as a hex string.

    Parameters
    ----------
    image : np.ndarray
        An image as a numpy array in CxHxW format

    Returns
    -------
    str
        The hex string hash of the image using perceptual hashing
    """
    # Verify that the image is at least larger than an 8x8 image
    min_dim = min(image.shape[-2:])
    if min_dim < HASH_SIZE + 1:
        raise ValueError(f"Image must be larger than {HASH_SIZE}x{HASH_SIZE} for fuzzy hashing.")

    # Calculates the dimensions of the resized square image
    resize_dim = HASH_SIZE * min((min_dim - 1) // HASH_SIZE, MAX_FACTOR)

    # Normalizes the image to CxHxW and takes the mean over all the channels
    normalized = np.mean(normalize_image_shape(image), axis=0).squeeze()

    # Rescales the pixel values to an 8-bit 0-255 image
    rescaled = rescale(normalized, 8).astype(np.uint8)

    # Resizes the image using the Lanczos algorithm to a square image
    im = np.array(Image.fromarray(rescaled).resize((resize_dim, resize_dim), Image.Resampling.LANCZOS))

    # Performs discrete cosine transforms to compress the image information and takes the lowest frequency component
    transform = dct(dct(im.T).T)[:HASH_SIZE, :HASH_SIZE]

    # Encodes the transform as a bit array over the median value
    diff = transform > np.median(transform)

    # Pads the front of the bit array to a multiple of 8 with False
    padded = np.full(int(np.ceil(diff.size / 8) * 8), False)
    padded[-diff.size :] = diff.ravel()

    # Converts the bit array to a hex string and strips leading 0s
    hash_hex = np.packbits(padded).tobytes().hex().lstrip("0")
    return hash_hex if hash_hex else "0"


def xxhash(image: np.ndarray) -> str:
    """
    Performs a fast non-cryptographic hash using the xxhash algorithm
    (xxhash.com) against the image as a flattened bytearray.  The hash
    is returned as a hex string.

    Parameters
    ----------
    image : np.ndarray
        An image as a numpy array

    Returns
    -------
    str
        The hex string hash of the image using the xxHash algorithm
    """
    return xxh.xxh3_64_hexdigest(image.ravel().tobytes())


def uap(labels: np.ndarray, scores: np.ndarray):
    return float(average_precision_score(labels, scores, average="weighted"))


def normalize_expected_dist(expected_dist: np.ndarray, observed_dist: np.ndarray) -> np.ndarray:
    exp_sum = np.sum(expected_dist)
    obs_sum = np.sum(observed_dist)

    if exp_sum == 0:
        raise ValueError(
            f"Expected label distribution {expected_dist} is all zeros. "
            "Ensure that Parity.expected_dist is set to a list "
            "with at least one nonzero element"
        )

    # Renormalize expected distribution to have the same total number of labels as the observed dataset
    if exp_sum != obs_sum:
        expected_dist = expected_dist * obs_sum / exp_sum

    return expected_dist


def digitize_factor_bins(continuous_values: np.ndarray, bins: int, factor_name: str):
    """
    Digitizes a list of values into a given number of bins.

    Parameters
    ----------
    continuous_values: np.ndarray
        The values to be digitized.
    bins: int
        The number of bins for the discrete values that continuous_values will be digitized into.
    factor_name: str
        The name of the factor to be digitized.

    Returns
    -------
    np.ndarray
        The digitized values

    """
    if not np.all([np.issubdtype(type(n), np.number) for n in continuous_values]):
        raise TypeError(
            f"Encountered a non-numeric value for factor {factor_name}, but the factor"
            " was specified to be continuous. Ensure all occurrences of this factor are numeric types,"
            f" or do not specify {factor_name} as a continuous factor."
        )

    _, bin_edges = np.histogram(continuous_values, bins=bins)
    bin_edges[-1] = np.inf
    bin_edges[0] = -np.inf
    return np.digitize(continuous_values, bin_edges)


def format_discretize_factors(
    data_factors: dict[str, np.ndarray], continuous_factor_bincounts: Dict[str, int]
) -> Tuple[dict, np.ndarray]:
    """
    Sets up the internal list of metadata factors.

    Parameters
    ----------
    data_factors: Dict[str, np.ndarray]
        The dataset factors, which are per-image attributes including class label and metadata.
        Each key of dataset_factors is a factor, whose value is the per-image factor values.
    continuous_factor_bincounts : Dict[str, int]
        The factors in data_factors that have continuous values and the array of bin counts to
        discretize values into. All factors are treated as having discrete values unless they
        are specified as keys in this dictionary. Each element of this array must occur as a key
        in data_factors.

    Returns
    -------
    Dict[str, np.ndarray]
        Intrinsic per-image metadata information with the formatting that input data_factors uses.
        Each key is a metadata factor, whose value is the discrete per-image factor values.
    np.ndarray
        Per-image labels, whose ith element is the label for the ith element of the dataset.
    """
    invalid_keys = set(continuous_factor_bincounts.keys()) - set(data_factors.keys())
    if invalid_keys:
        raise KeyError(
            f"The continuous factor(s) {invalid_keys} do not exist in data_factors. Delete these "
            "keys from `continuous_factor_names` or add corresponding entries to `data_factors`."
        )

    metadata_factors = {}

    # make sure each factor has the same number of entries
    lengths = []
    for arr in data_factors.values():
        lengths.append(arr.shape)

    if lengths[1:] != lengths[:-1]:
        raise ValueError("The lengths of each entry in the dictionary are not equal." f" Found lengths {lengths}")

    labels = data_factors["class"]

    metadata_factors = {
        name: val
        if name not in continuous_factor_bincounts
        else digitize_factor_bins(val, continuous_factor_bincounts[name], name)
        for name, val in data_factors.items()
        if name != "class"
    }

    return metadata_factors, labels


def parity(factors: dict[str, np.ndarray], labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluates the statistical independence of metadata factors from class labels.
    This performs a chi-square test, which provides a score and a p-value for
    statistical independence between each pair of a metadata factor and a class label.
    A high score with a low p-value suggests that a metadata factor is strongly
    correlated with a class label.

    Parameters
    ----------
    factors: Dict[str, np.ndarray]
        Intrinsic per-image metadata information.
        factors['key'][i] is the value of the metadata factor 'key' at the ith element of the dataset.
    labels: np.ndarray
        Dataset labels.
        Labels[i] is the label for the ith element of the dataset.

    Returns
    -------
    np.ndarray
        Array of length (num_factors) whose (i)th element corresponds to
        the chi-square score for the relationship between factor i
        and the class labels in the dataset.
    np.ndarray
        Array of length (num_factors) whose (i)th element corresponds to
        the p-value value for the chi-square test for the relationship between
        factor i and the class labels in the dataset.
    """

    chi_scores = np.zeros(len(factors))
    p_values = np.zeros(len(factors))
    n_cls = len(np.unique(labels))
    for i, (current_factor_name, factor_values) in enumerate(factors.items()):
        unique_factor_values = np.unique(factor_values)
        contingency_matrix = np.zeros((len(unique_factor_values), n_cls))
        # Builds a contingency matrix where entry at index (r,c) represents
        # the frequency of current_factor_name achieving value unique_factor_values[r]
        # at a data point with class c.

        # TODO: Vectorize this nested for loop
        for fi, factor_value in enumerate(unique_factor_values):
            for label in range(n_cls):
                with_both = np.bitwise_and((labels == label), factor_values == factor_value)
                contingency_matrix[fi, label] = np.sum(with_both)
                if 0 < contingency_matrix[fi, label] < 5:
                    warnings.warn(
                        f"Factor {current_factor_name} value {factor_value} co-occurs "
                        f"only {contingency_matrix[fi, label]} times with label {label}. "
                        "This can cause inaccurate chi_square calculation. Recommend"
                        "ensuring each label occurs either 0 times or at least 5 times. "
                        "Alternatively, digitize any continuous-valued factors "
                        "into fewer bins."
                    )

        # This deletes rows containing only zeros,
        # because scipy.stats.chi2_contingency fails when there are rows containing only zeros.
        rowsums = np.sum(contingency_matrix, axis=1)
        rowmask = np.where(rowsums)
        contingency_matrix = contingency_matrix[rowmask]

        chi2, p, _, _ = chi2_contingency(contingency_matrix)

        chi_scores[i] = chi2
        p_values[i] = p
    return chi_scores, p_values


def get_counts(
    data: np.ndarray, names: List[str], is_categorical: List[bool], subset_mask: Optional[np.ndarray] = None
) -> tuple[Dict, Dict]:
    """
    Initialize dictionary of histogram counts --- treat categorical values
    as histogram bins.

    Parameters
    ----------
    subset_mask: Optional[np.ndarray[bool]]
        Boolean mask of samples to bin (e.g. when computing per class).  True -> include in histogram counts

    Returns
    -------
    counts: Dict
        histogram counts per metadata factor in `factors`.  Each
        factor will have a different number of bins.  Counts get reused
        across metrics, so hist_counts are cached but only if computed
        globally, i.e. without masked samples.
    """

    hist_counts, hist_bins = {}, {}
    # np.where needed to satisfy linter
    mask = np.where(subset_mask if subset_mask is not None else np.ones(data.shape[0], dtype=bool))

    for cdx, fn in enumerate(names):
        # linter doesn't like double indexing
        col_data = data[mask, cdx].squeeze()
        if is_categorical[cdx]:
            # if discrete, use unique values as bins
            bins, cnts = np.unique(col_data, return_counts=True)
        else:
            bins = hist_bins.get(fn, "auto")
            cnts, bins = np.histogram(col_data, bins=bins, density=True)

        hist_counts[fn] = cnts
        hist_bins[fn] = bins

    return hist_counts, hist_bins


def entropy(
    data: np.ndarray,
    names: List[str],
    is_categorical: List[bool],
    normalized: bool = False,
    subset_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Meant for use with Bias metrics, Balance, Diversity, ClasswiseBalance,
    and Classwise Diversity.

    Compute entropy for discrete/categorical variables and, through standard
    histogram binning, for continuous variables.

    Parameters
    ----------
    normalized: bool
        Flag that determines whether or not to normalize entropy by log(num_bins)
    subset_mask: Optional[np.ndarray[bool]]
        Boolean mask of samples to bin (e.g. when computing per class).  True -> include in histogram counts

    Notes
    -----
    For continuous variables, histogram bins are chosen automatically.  See
    numpy.histogram for details.

    Returns
    -------
    ent: np.ndarray[float]
        Entropy estimate per column of X

    See Also
    --------
    numpy.histogram
    scipy.stats.entropy
    """

    num_factors = len(names)
    hist_counts, _ = get_counts(data, names, is_categorical, subset_mask)

    ev_index = np.empty(num_factors)
    for col, cnts in enumerate(hist_counts.values()):
        # entropy in nats, normalizes counts
        ev_index[col] = sp_entropy(cnts)
        if normalized:
            if len(cnts) == 1:
                # log(0)
                ev_index[col] = 0
            else:
                ev_index[col] /= np.log(len(cnts))
    return ev_index


def get_num_bins(
    data: np.ndarray, names: List[str], is_categorical: List[bool], subset_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Number of bins or unique values for each metadata factor, used to
    normalize entropy/diversity.

    Parameters
    ----------
    subset_mask: Optional[np.ndarray[bool]]
        Boolean mask of samples to bin (e.g. when computing per class).  True -> include in histogram counts
    """
    # likely cached
    hist_counts, _ = get_counts(data, names, is_categorical, subset_mask)
    num_bins = np.empty(len(hist_counts))
    for idx, cnts in enumerate(hist_counts.values()):
        num_bins[idx] = len(cnts)

    return num_bins


def infer_categorical(X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Compute fraction of feature values that are unique --- intended to be used
    for inferring whether variables are categorical.
    """
    if X.ndim == 1:
        X = np.expand_dims(X, axis=1)
    num_samples = X.shape[0]
    pct_unique = np.empty(X.shape[1])
    for col in range(X.shape[1]):  # type: ignore
        uvals = np.unique(X[:, col], axis=0)
        pct_unique[col] = len(uvals) / num_samples
    return pct_unique < threshold


def diversity_simpson(
    data: np.ndarray,
    names: List[str],
    is_categorical: List[bool],
    subset_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute diversity for discrete/categorical variables and, through standard
    histogram binning, for continuous variables.

    We define diversity as a normalized form of the inverse Simpson diversity
    index.

    diversity = 1 implies that samples are evenly distributed across a particular factor
    diversity = 1/num_categories implies that all samples belong to one category/bin

    Parameters
    ----------
    subset_mask: Optional[np.ndarray[bool]]
        Boolean mask of samples to bin (e.g. when computing per class).  True -> include in histogram counts

    Notes
    -----
    For continuous variables, histogram bins are chosen automatically.  See
        numpy.histogram for details.
    The expression is undefined for q=1, but it approaches the Shannon entropy
        in the limit.
    If there is only one category, the diversity index takes a value of 1 =
        1/N = 1/1.  Entropy will take a value of 0.

    Returns
    -------
    np.ndarray
        Diversity index per column of X

    See Also
    --------
    numpy.histogram
    """

    hist_counts, _ = get_counts(data, names, is_categorical, subset_mask)
    # normalize by global counts, not classwise counts
    num_bins = get_num_bins(data, names, is_categorical)

    ev_index = np.empty(len(names))
    # loop over columns for convenience
    for col, cnts in enumerate(hist_counts.values()):
        # relative frequencies
        p_i = cnts / cnts.sum()
        # inverse Simpson index normalized by (number of bins)
        ev_index[col] = 1 / np.sum(p_i**2) / num_bins[col]

    return ev_index


def preprocess_metadata(class_labels: Sequence[int], metadata: List[Dict]) -> Tuple[np.ndarray, List[str], List[bool]]:
    # convert class_labels and list of metadata dicts to dict of ndarrays
    metadata_dict: Dict[str, np.ndarray] = {
        "class_label": np.asarray(class_labels, dtype=int),
        **{k: np.array([d[k] for d in metadata]) for k in metadata[0]},
    }

    # map columns of dict that are not numeric (e.g. string) to numeric values
    # that mutual information and diversity functions can accommodate.  Each
    # unique string receives a unique integer value.
    for k, v in metadata_dict.items():
        # if not numeric
        if not np.issubdtype(v.dtype, np.number):
            _, mapped_vals = np.unique(v, return_inverse=True)
            metadata_dict[k] = mapped_vals

    data = np.stack(list(metadata_dict.values()), axis=-1)
    names = list(metadata_dict.keys())
    is_categorical = [infer_categorical(metadata_dict[var], 0.25)[0] for var in names]

    return data, names, is_categorical


def diversity_shannon(
    data: np.ndarray,
    names: List[str],
    is_categorical: List[bool],
    subset_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute diversity for discrete/categorical variables and, through standard
    histogram binning, for continuous variables.

    We define diversity as a normalized form of the Shannon entropy.

    diversity = 1 implies that samples are evenly distributed across a particular factor
    diversity = 0 implies that all samples belong to one category/bin

    Parameters
    ----------
    subset_mask: Optional[np.ndarray[bool]]
        Boolean mask of samples to bin (e.g. when computing per class).  True -> include in histogram counts

    Notes
    -----
    - For continuous variables, histogram bins are chosen automatically.  See
    numpy.histogram for details.

    Returns
    -------
    diversity_index: np.ndarray
        Diversity index per column of X

    See Also
    --------
    numpy.histogram
    """

    # entropy computed using global auto bins so that we can properly normalize
    ent_unnormalized = entropy(data, names, is_categorical, normalized=False, subset_mask=subset_mask)
    # normalize by global counts rather than classwise counts
    num_bins = get_num_bins(data, names, is_categorical=is_categorical, subset_mask=subset_mask)
    return ent_unnormalized / np.log(num_bins)
