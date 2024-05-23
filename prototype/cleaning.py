from copy import deepcopy

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform


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


def get_distance(cluster, level, sample, distance_array, distance_matrix, sample_clusters):
    # Convert the condensed distance matrix to a square form
    square_distance_matrix = squareform(distance_matrix)

    # For each cluster, check if it is active at the current level
    for cluster_num, level_info in sample_clusters.items():
        if cluster_num != cluster and level in level_info:
            samples = level_info[level]["samples"]

            for other_sample in samples:
                if square_distance_matrix[sample, other_sample] < distance_array[cluster]:
                    distance_array[cluster] = square_distance_matrix[sample, other_sample]

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
            new_structure[cluster_num][level] = {"samples": deepcopy(new_structure[cluster_num][level - 1]["samples"])}

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
            level = max([level, clusters[arr[i, 0]]["level"] + 1])
            distance += clusters[arr[i, 0]]["total_dist"]
            count += clusters[arr[i, 0]]["count"]
        else:
            sample_added.append(arr[i, 0])

        if arr[i, 1] in clusters:
            cluster_num = min([cluster_num, clusters[arr[i, 1]]["cluster_num"]])
            level = max([level, clusters[arr[i, 1]]["level"] + 1])
            distance += clusters[arr[i, 1]]["total_dist"]
            count += clusters[arr[i, 1]]["count"]
        else:
            sample_added.append(arr[i, 1])

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
            level = values["level"]
            cluster = values["cluster_num"]
            for sample in values["samples_added"]:
                sample_tracking[sample]["cluster"][level] = values["cluster_num"]
                sample_tracking[sample]["distance"][level, cluster] = values["sample_dist"]
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


# def get_clusters(samples):
#     for _ in range(len(samples)):
#         break


def sort_linkage(Z):
    """
    Sort the linkage matrix Z in reverse order by distance and
    then by cluster size (new_size).

    Parameters:
    - arr: linkage matrix

    Returns:
    - arr: Sorted linkage matrix
    """
    # Adjusting linkage matrix to accommodate renumbering
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
    Z = linkage(distance_matrix, method="single")

    # Sort the linkage matrix
    Zsort = sort_linkage(Z)

    # Get the information for each sample and the optimum # of clusters
    sample_dict = get_sample_info(Zsort, distance_matrix)

    # Get the number of clusters
    # num_clusters = get_clusters(sample_dict)

    return sample_dict
