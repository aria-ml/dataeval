from __future__ import annotations

from typing import Any, Callable, Literal, Mapping, NamedTuple

import numpy as np
import xxhash as xxh
from numpy.typing import ArrayLike, NDArray
from PIL import Image
from scipy.fftpack import dct
from scipy.signal import convolve2d
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree as mst
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy as sp_entropy
from sklearn.neighbors import NearestNeighbors

from dataeval._internal.interop import to_numpy

EPSILON = 1e-5
EDGE_KERNEL = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.int8)
BIT_DEPTH = (1, 8, 12, 16, 32)
HASH_SIZE = 8
MAX_FACTOR = 4


def get_method(method_map: dict[str, Callable], method: str) -> Callable:
    if method not in method_map:
        raise ValueError(f"Specified method {method} is not a valid method: {method_map}.")
    return method_map[method]


def get_counts(
    data: NDArray, names: list[str], is_categorical: list[bool], subset_mask: NDArray[np.bool_] | None = None
) -> tuple[dict, dict]:
    """
    Initialize dictionary of histogram counts --- treat categorical values
    as histogram bins.

    Parameters
    ----------
    subset_mask: NDArray[np.bool_] | None
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
    data: NDArray,
    names: list[str],
    is_categorical: list[bool],
    normalized: bool = False,
    subset_mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """
    Meant for use with Bias metrics, Balance, Diversity, ClasswiseBalance,
    and Classwise Diversity.

    Compute entropy for discrete/categorical variables and for continuous variables through standard
    histogram binning.

    Parameters
    ----------
    normalized: bool
        Flag that determines whether or not to normalize entropy by log(num_bins)
    subset_mask: NDArray[np.bool_] | None
        Boolean mask of samples to bin (e.g. when computing per class).  True -> include in histogram counts

    Note
    ----
    For continuous variables, histogram bins are chosen automatically.  See
    numpy.histogram for details.

    Returns
    -------
    ent: NDArray[np.float64]
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
    data: NDArray, names: list[str], is_categorical: list[bool], subset_mask: NDArray[np.bool_] | None = None
) -> NDArray[np.float64]:
    """
    Number of bins or unique values for each metadata factor, used to
    normalize entropy/diversity.

    Parameters
    ----------
    subset_mask: NDArray[np.bool_] | None
        Boolean mask of samples to bin (e.g. when computing per class).  True -> include in histogram counts

    Returns
    -------
    NDArray[np.float64]
    """
    # likely cached
    hist_counts, _ = get_counts(data, names, is_categorical, subset_mask)
    num_bins = np.empty(len(hist_counts))
    for idx, cnts in enumerate(hist_counts.values()):
        num_bins[idx] = len(cnts)

    return num_bins


def infer_categorical(X: NDArray, threshold: float = 0.2) -> NDArray:
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


def preprocess_metadata(
    class_labels: ArrayLike, metadata: Mapping[str, ArrayLike], cat_thresh: float = 0.2
) -> tuple[NDArray, list[str], list[bool]]:
    # convert class_labels and dict of lists to matrix of metadata values
    preprocessed_metadata = {"class_label": np.asarray(class_labels, dtype=int)}

    # map columns of dict that are not numeric (e.g. string) to numeric values
    # that mutual information and diversity functions can accommodate.  Each
    # unique string receives a unique integer value.
    for k, v in metadata.items():
        # if not numeric
        v = to_numpy(v)
        if not np.issubdtype(v.dtype, np.number):
            _, mapped_vals = np.unique(v, return_inverse=True)
            preprocessed_metadata[k] = mapped_vals
        else:
            preprocessed_metadata[k] = v

    data = np.stack(list(preprocessed_metadata.values()), axis=-1)
    names = list(preprocessed_metadata.keys())
    is_categorical = [infer_categorical(preprocessed_metadata[var], cat_thresh)[0] for var in names]

    return data, names, is_categorical


def flatten(X: NDArray):
    """
    Flattens input array from (N, ... ) to (N, -1) where all samples N have all data in their last dimension

    Parameters
    ----------
    X : NDArray, shape - (N, ... )
        Input array

    Returns
    -------
    NDArray, shape - (N, -1)
    """

    return X.reshape((X.shape[0], -1))


def minimum_spanning_tree(X: NDArray) -> Any:
    """
    Returns the minimum spanning tree from a NumPy image array.

    Parameters
    ----------
    X : NDArray
        Numpy image array

    Returns
    -------
        Data representing the minimum spanning tree
    """
    # All features belong on second dimension
    X = flatten(X)
    # We add a small constant to the distance matrix to ensure scipy interprets
    # the input graph as fully-connected.
    dense_eudist = squareform(pdist(X)) + EPSILON
    eudist_csr = csr_matrix(dense_eudist)
    return mst(eudist_csr)


def get_classes_counts(labels: NDArray) -> tuple[int, int]:
    """
    Returns the classes and counts of from an array of labels

    Parameters
    ----------
    label : NDArray
        Numpy labels array

    Returns
    -------
        Classes and counts

    Raises
    ------
    ValueError
        If the number of unique classes is less than 2
    """
    classes, counts = np.unique(labels, return_counts=True)
    M = len(classes)
    if M < 2:
        raise ValueError("Label vector contains less than 2 classes!")
    N = np.sum(counts).astype(int)
    return M, N


def compute_neighbors(
    A: NDArray,
    B: NDArray,
    k: int = 1,
    algorithm: Literal["auto", "ball_tree", "kd_tree"] = "auto",
) -> NDArray:
    """
    For each sample in A, compute the nearest neighbor in B

    Parameters
    ----------
    A, B : NDArray
        The n_samples and n_features respectively
    k : int
        The number of neighbors to find
    algorithm : Literal
        Tree method for nearest neighbor (auto, ball_tree or kd_tree)

    Note
    ----
        Do not use kd_tree if n_features > 20

    Returns
    -------
    List:
        Closest points to each point in A and B

    Raises
    ------
    ValueError
        If algorithm is not "auto", "ball_tree", or "kd_tree"

    See Also
    --------
    sklearn.neighbors.NearestNeighbors
    """

    if k < 1:
        raise ValueError("k must be >= 1")
    if algorithm not in ["auto", "ball_tree", "kd_tree"]:
        raise ValueError("Algorithm must be 'auto', 'ball_tree', or 'kd_tree'")

    A = flatten(A)
    B = flatten(B)

    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm=algorithm).fit(B)
    nns = nbrs.kneighbors(A)[1]
    nns = nns[:, 1:].squeeze()

    return nns


class BitDepth(NamedTuple):
    depth: int
    pmin: float | int
    pmax: float | int


def get_bitdepth(image: NDArray) -> BitDepth:
    """
    Approximates the bit depth of the image using the
    min and max pixel values.
    """
    pmin, pmax = np.min(image), np.max(image)
    if pmin < 0:
        return BitDepth(0, pmin, pmax)
    else:
        depth = ([x for x in BIT_DEPTH if 2**x > pmax] or [max(BIT_DEPTH)])[0]
        return BitDepth(depth, 0, 2**depth - 1)


def rescale(image: NDArray, depth: int = 1) -> NDArray:
    """
    Rescales the image using the bit depth provided.
    """
    bitdepth = get_bitdepth(image)
    if bitdepth.depth == depth:
        return image
    else:
        normalized = (image + bitdepth.pmin) / (bitdepth.pmax - bitdepth.pmin)
        return normalized * (2**depth - 1)


def normalize_image_shape(image: NDArray) -> NDArray:
    """
    Normalizes the image shape into (C,H,W).
    """
    ndim = image.ndim
    if ndim == 2:
        return np.expand_dims(image, axis=0)
    elif ndim == 3:
        return image
    elif ndim > 3:
        # Slice all but the last 3 dimensions
        return image[(0,) * (ndim - 3)]
    else:
        raise ValueError("Images must have 2 or more dimensions.")


def normalize_box_shape(bounding_box: NDArray) -> NDArray:
    """
    Normalizes the bounding box shape into (N,4).
    """
    ndim = bounding_box.ndim
    if ndim == 1:
        return np.expand_dims(bounding_box, axis=0)
    elif ndim > 2:
        raise ValueError("Bounding boxes must have 2 dimensions: (# of boxes in an image, [X,Y,W,H]) -> (N,4)")
    else:
        return bounding_box


def edge_filter(image: NDArray, offset: float = 0.5) -> NDArray:
    """
    Returns the image filtered using a 3x3 edge detection kernel:
    [[ -1, -1, -1 ],
     [ -1,  8, -1 ],
     [ -1, -1, -1 ]]
    """
    edges = convolve2d(image, EDGE_KERNEL, mode="same", boundary="symm") + offset
    np.clip(edges, 0, 255, edges)
    return edges


def pchash(image: NDArray) -> str:
    """
    Performs a perceptual hash on an image by resizing to a square NxN image
    using the Lanczos algorithm where N is 32x32 or the largest multiple of
    8 that is smaller than the input image dimensions.  The resampled image
    is compressed using a discrete cosine transform and the lowest frequency
    component is encoded as a bit array of greater or less than median value
    and returned as a hex string.

    Parameters
    ----------
    image : NDArray
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


def xxhash(image: NDArray) -> str:
    """
    Performs a fast non-cryptographic hash using the xxhash algorithm
    (xxhash.com) against the image as a flattened bytearray.  The hash
    is returned as a hex string.

    Parameters
    ----------
    image : NDArray
        An image as a numpy array

    Returns
    -------
    str
        The hex string hash of the image using the xxHash algorithm
    """
    return xxh.xxh3_64_hexdigest(image.ravel().tobytes())
