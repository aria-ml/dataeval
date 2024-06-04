import hashlib
import os
import typing
from copy import deepcopy
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform


def _validate_file(fpath, file_hash, chunk_size=65535):
    hasher = hashlib.sha256()
    with open(fpath, "rb") as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b""):
            hasher.update(chunk)

    return str(hasher.hexdigest()) == str(file_hash)


def _get_file(
    fname: str,
    origin: str,
    file_hash: typing.Optional[str] = None,
):
    cache_dir = os.path.join(os.path.expanduser("~"), ".keras")
    datadir_base = os.path.expanduser(cache_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join("/tmp", ".keras")
    datadir = os.path.join(datadir_base, "datasets")
    os.makedirs(datadir, exist_ok=True)

    fname = os.fspath(fname) if isinstance(fname, os.PathLike) else fname
    fpath = os.path.join(datadir, fname)

    download = False
    if os.path.exists(fpath):
        if file_hash is not None and not _validate_file(fpath, file_hash):
            download = True
    else:
        download = True

    if download:
        try:
            error_msg = "URL fetch failure on {}: {} -- {}"
            try:
                urlretrieve(origin, fpath)
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg)) from e
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason)) from e
        except (Exception, KeyboardInterrupt):
            if os.path.exists(fpath):
                os.remove(fpath)
            raise

        if (
            os.path.exists(fpath)
            and file_hash is not None
            and not _validate_file(fpath, file_hash)
        ):
            raise ValueError(
                "Incomplete or corrupted file detected. "
                f"The sha256 file hash does not match the provided value "
                f"of {file_hash}.",
            )
    return fpath


def square_to_condensed(square_matrix):
    """Convert a square distance matrix to a condensed distance matrix."""
    assert square_matrix.shape[0] == square_matrix.shape[1], "Matrix must be square"
    n = square_matrix.shape[0]
    tri_indices = np.triu_indices(n, 1)
    condensed_matrix = square_matrix[tri_indices]
    return condensed_matrix


def L2_distance_matrix(x):
    """
    Takes advantage that Euclidean distance can be written as
    ||a-b||**2 = ||a||**2 + ||b||**2 - 2 a @ b
    Also, does not compute the sqrt at the end since the squared distance
    is still going to produce an equivalent nearest neighbor matrix
    """
    # Adjusting dtype to be friendlier for speed and memory
    # x = x.astype(np.float32)
    # Compute the squared magnitude of the vector
    x2 = np.sum(np.square(x), axis=1, keepdims=True, dtype=x.dtype)
    # Compute the matrix product between x and its transpose
    mat = np.matmul(x, x.T) * -2
    # Utilize broadcasting to perform the addition
    mat += x2
    mat += x2.T

    return mat


def get_duplicate(link_arr, distance):
    link_std = link_arr.std()

    if distance <= link_std / 1e3:
        return "exact duplicate"
    elif distance <= link_std:
        return "near duplicate"
    else:
        return ""


def get_outlier(level, distance, dist_arr):
    dist_mean, dist_std = dist_arr[level].mean(), dist_arr[level].std()

    if abs(dist_mean - distance) < dist_std * 2:
        return "outlier"
    elif level >= dist_arr.shape[0] * 2 / 3:
        return "potential outlier"
    else:
        return ""


def get_distance(
    cluster, level, sample, distance_array, distance_matrix, sample_clusters
):
    # Convert the condensed distance matrix to a square form
    square_distance_matrix = squareform(distance_matrix)

    # For each cluster, check if it is active at the current level
    for cluster_num, level_info in sample_clusters.items():
        if cluster_num != cluster and level in level_info:
            samples = level_info[level]["samples"]

            for other_sample in samples:
                if (
                    square_distance_matrix[sample, other_sample]
                    < distance_array[cluster]
                ):
                    distance_array[cluster] = square_distance_matrix[
                        sample, other_sample
                    ]

    return distance_array


def reorganize_clusters(clusters):
    """
    Reorganize the clusters dictionary to be nested by cluster_num, then by level,
    and include avg_dist, sample_dist, and samples within each level.

    Parameters:
    - clusters: A dictionary containing the original clusters information.

    Returns:
    - new_structure: A dictionary reorganized by cluster_num,
                      then by level, with details.
    """
    new_structure = {}

    for _, info in clusters.items():
        # Extract necessary information
        cluster_num = info["cluster_num"]
        level = info["level"]
        samples = info.get("samples_added", [])

        # Initialize the structure if not present
        if cluster_num not in new_structure:
            new_structure[cluster_num] = {}

        if level not in new_structure[cluster_num] and level == 1:
            new_structure[cluster_num][level] = {"samples": []}
        elif level not in new_structure[cluster_num] and level > 1:
            new_structure[cluster_num][level] = {
                "samples": deepcopy(new_structure[cluster_num][level - 1]["samples"])
            }

        # Extending the samples list.
        new_structure[cluster_num][level]["samples"].extend(samples)

    return new_structure


def get_sample_info(arr, distance_matrix):
    """
    Initialize clusters based on number of individual sample merges.

    Parameters:
    - arr: sorted linkage matrix

    Returns:
    - clusters: A dictionary containing the clusters
    """
    # Determining maximum number of levels and clusters
    max_clusters = 1
    max_levels = 1
    clusters = {}
    for i in range(len(arr)):
        level = 1
        cluster_num = max_clusters
        distance = 0
        count = 0
        sample_added = []
        if arr[i, 0] in clusters:
            cluster_num = min([cluster_num, clusters[arr[i, 0]]["cluster_num"]])
            left_level = max([level, clusters[arr[i, 0]]["level"] + 1])
            distance += clusters[arr[i, 0]]["total_dist"]
            count += clusters[arr[i, 0]]["count"]
        else:
            sample_added.append(int(arr[i, 0]))

        if arr[i, 1] in clusters:
            cluster_num = min([cluster_num, clusters[arr[i, 1]]["cluster_num"]])
            right_level = max([level, clusters[arr[i, 1]]["level"] + 1])
            distance += clusters[arr[i, 1]]["total_dist"]
            count += clusters[arr[i, 1]]["count"]
        else:
            sample_added.append(int(arr[i, 1]))

        if arr[i, 0] in clusters and arr[i, 1] in clusters:
            if cluster_num == clusters[arr[i, 0]]["cluster_num"]:
                level = left_level
            elif cluster_num == clusters[arr[i, 1]]["cluster_num"]:
                level = right_level
        elif arr[i, 0] in clusters:
            level = left_level
        elif arr[i, 1] in clusters:
            level = right_level

        count += 1
        distance += arr[i, 2]

        clusters[arr[i, -1]] = {
            "cluster_num": cluster_num,
            "level": level,
            "total_dist": distance,
            "count": count,
            "avg_dist": distance / count,
            "samples_added": sample_added,
            "sample_dist": arr[i, 2],
        }

        if cluster_num == max_clusters and i < len(arr) - 1:
            max_clusters += 1

        if level > max_levels:
            max_levels = level

    # Reorganizing the clusters dictionary
    sample_clusters = reorganize_clusters(clusters)

    # Creating the cluster tracking dictionary
    sample_tracking = {
        i: {
            "cluster": np.zeros(max_levels),
            "distance": np.full((max_levels, max_clusters), np.inf),
            "duplicate": "",
            "outlier": "",
        }
        for i in range(len(arr) + 1)
    }

    for _, values in clusters.items():
        if values["samples_added"]:
            level = values["level"] - 1
            cluster = values["cluster_num"] - 1
            for sample in values["samples_added"]:
                sample_tracking[sample]["cluster"][level] = values["cluster_num"]
                sample_tracking[sample]["distance"][level, cluster] = values[
                    "sample_dist"
                ]
                sample_tracking[sample]["distance"][level, :] = get_distance(
                    cluster,
                    level,
                    sample,
                    sample_tracking[sample]["distance"][level, :],
                    distance_matrix,
                    sample_clusters,
                )
                sample_tracking[sample]["duplicate"] = get_duplicate(
                    arr[:, 2],
                    values["sample_dist"],
                )
                sample_tracking[sample]["outlier"] = get_outlier(
                    level,
                    values["sample_dist"],
                    sample_tracking[sample]["distance"],
                )

    return sample_tracking


def sort_linkage(Z):
    """
    Sort the linkage matrix Z in reverse order by distance and
    then by cluster size (new_size).

    Parameters:
    - arr: linkage matrix

    Returns:
    - arr: Sorted linkage matrix
    """
    # Adjusting linkage matrix to accomodate renumbering
    arr = np.zeros((Z.shape[0], Z.shape[1] + 1))
    arr[:, :-1] = Z.copy()
    arr[:, -1] = np.arange(Z.shape[0] + 1, 2 * Z.shape[0] + 1)

    # Sort by decreasing distance, then by increasing new_size
    # arr = arr[arr[:, 2].argsort()[::-1]]
    # arr = arr[arr[:, -2].argsort(kind="stable")]

    return arr


def hierarchical_clustering_with_tracking(data):
    """
    Perform hierarchical clustering with detailed tracking of each cluster.

    Parameters:
    - data: numpy.ndarray, shape (n_samples, n_features)

    Returns:
    - cluster_dict: dict, detailed tracking of each cluster
    """
    # Compute pairwise distances and perform hierarchical clustering
    distance_matrix = pdist(data, metric="euclidean")
    # distance_matrix = square_to_condensed(L2_distance_matrix(data))
    Z = linkage(distance_matrix, method="single")

    # Sort the linkage matrix
    Zsort = sort_linkage(Z)

    # Get the information for each sample and the optimum # of clusters
    sample_dict = get_sample_info(Zsort, distance_matrix)

    # Get the number of clusters
    # num_clusters = get_clusters(sample_dict)

    return sample_dict


# Getting the mnist dataset and prepping for testing
rng = np.random.default_rng(33)

origin_folder = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
path = _get_file(
    "mnist.npz",
    origin=origin_folder + "mnist.npz",
    file_hash=("731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"),
)

with np.load(path, allow_pickle=True) as fp:
    images, labels = fp["x_train"][:100], fp["y_train"][:100]

dup_images = deepcopy(images[:8])
dup_images[:, :25, :25] = images[:8, 3:, 3:]
dup_images[:, 25:, 25:] = images[:8, :3, :3]

test_imgs = np.concatenate([images, dup_images])
test_imgs /= 255

rng.shuffle(test_imgs)

# Example usage
cluster_tracking = hierarchical_clustering_with_tracking(test_imgs)
# print(num_clusters)

for i, idd in enumerate(cluster_tracking.keys()):
    if i % 10 == 0:
        print(cluster_tracking[idd])

print(cluster_tracking[idd])
