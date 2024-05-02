from typing import Any, Dict

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform


def get_condensed_distance_array(data):
    """
    Calculates the condensed euclidean distance array of the input data

    Parameters
    ----------
    data: np.ndarray
        Array to find the pairwise euclidean distances between all points

    Returns
    -------
    np.ndarray
        A 1-D array of the distance matrix
    """
    return pdist(data, metric="euclidean")


def get_square_distance_matrix(condensed_distance_array):
    return squareform(condensed_distance_array)


def get_linkage_arr(condensed_distance_array):
    return linkage(condensed_distance_array, method="single")


def extend_linkage(link_arr):
    """
    Adds a column to the linkage matrix Z that tracks the new id assigned
    to each row

    Parameters
    ----------
    Z
        linkage matrix

    Returns
    -------
    arr
        linkage matrix with adjusted shape, new shape (Z.shape[0], Z.shape[1]+1)
    """
    # Adjusting linkage matrix to accommodate renumbering
    rows, cols = link_arr.shape
    arr = np.zeros((rows, cols + 1))
    arr[:, :-1] = link_arr
    arr[:, -1] = np.arange(rows + 1, 2 * rows + 1)

    return arr


def get_extended_linkage(condensed_distance_array):
    link_arr = get_linkage_arr(condensed_distance_array)
    return extend_linkage(link_arr)


# Can convert to generic type for scalar values
def clamp(x: int, minimum: int, maximum: int):
    """Clamps a value between the minimum and maximum

    Equivalent to
    ```
    minimum if x < minimum else maximum if x > maximum else x
    ```
    """
    return max(minimum, min(x, maximum))


class Clusterer2:
    def __init__(self, dataset: np.ndarray):
        """ """
        # This is done to update the state rather than instantiate a new class when new data is passed in
        self._on_init(dataset)

    def _on_init(self, dataset: np.ndarray):
        self._data: np.ndarray = dataset
        self.num_samples = len(dataset)
        self.darr: np.ndarray = get_condensed_distance_array(dataset)
        self.sqdmat: np.ndarray = get_square_distance_matrix(self.darr)
        self.larr: np.ndarray = get_extended_linkage(self.darr)
        self.max_clusters: int = np.count_nonzero(self.larr[:, 3] == 2)
        self.last_merge_level: int = 1

        min_num = int(self.num_samples * 0.05)
        self.min_num_samples_per_cluster = clamp(min_num, 2, 100)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, x: np.ndarray):
        self._on_init(x)

    def create_clusters(self) -> Dict[int, Any]:
        """Generates clusters based on linkage matrix

        Returns
        -------
        dict[str, Any]
            Cluster information
        """
        cluster_num = 0
        cluster_tracking = 0
        clusters = {}  # Dictionary to store clusters
        tracking = {}  # Dictionary to associate new cluster ids with actual clusters

        # Walking through the linkage array to generate clusters
        for arr_i in self.larr:
            level = 0
            merged = False

            arr_0 = int(arr_i[0])  # Grabbing the left id
            arr_1 = int(arr_i[1])  # Grabbing the right id
            dist = arr_i[2]
            sample_dist = np.array([dist], dtype=np.float16)

            # Linkage matrix first column id
            left_id = tracking.get(arr_0)  # Determining if the id is already associated with a cluster
            if left_id:
                left_cluster = left_id[1]
                left_level = left_id[0] + 1
                left_sample = clusters[left_id[0]][left_cluster]["samples"]
                sample_dist = np.concatenate([clusters[left_id[0]][left_cluster]["sample_dist"], sample_dist])
            # Linkage matrix second column id
            right_id = tracking.get(arr_1)  # Determining if the id is already associated with a cluster
            if right_id:
                right_cluster = right_id[1]
                right_level = right_id[0] + 1
                right_sample = clusters[right_id[0]][right_cluster]["samples"]
                sample_dist = np.concatenate([clusters[right_id[0]][right_cluster]["sample_dist"], sample_dist])

            # Aggregate samples, determine cluster number, and get the level
            if left_id and right_id:
                if clusters[left_id[0]][left_cluster]["count"] > clusters[right_id[0]][right_cluster]["count"]:
                    samples = np.concatenate([left_sample, right_sample])
                else:
                    samples = np.concatenate([right_sample, left_sample])
                cluster_num = min([left_cluster, right_cluster])
                merged = max([left_cluster, right_cluster])
                level = max([left_level, right_level])
                # Only tracking the levels in which clusters merge for the cluster distance matrix
                self.last_merge_level = max(self.last_merge_level, level + 1)
            elif left_id:
                samples = np.concatenate([left_sample, [arr_1]])
                cluster_num = left_cluster
                level = left_level
            elif right_id:
                samples = np.concatenate([right_sample, [arr_0]])
                cluster_num = right_cluster
                level = right_level
            else:
                samples = np.array([arr_0, arr_1], dtype=np.int32)
                cluster_num = cluster_tracking

            # Calculate distances
            dist_avg = np.mean(sample_dist)
            dist_std = np.std(sample_dist) if sample_dist.shape[0] > 1 else 1e-5

            out1 = dist_avg + dist_std
            out2 = out1 + dist_std

            # Initialize the structure if not present
            if level not in clusters:
                clusters[level] = {}

            clusters[level][cluster_num] = {
                "cluster_merged": merged,
                "count": samples.shape[0],
                "avg_dist": dist_avg,
                "dist_std": dist_std,
                "samples": samples,
                "sample_dist": sample_dist,
                "outside_1-std": dist > out1,
                "outside_2-std": dist > out2,
            }

            tracking[int(arr_i[-1])] = (level, cluster_num)  # Associates the new linkage id with the correct cluster

            # If no left or right id, increment tracker to ensure new clusters get unique ids
            if not left_id and not right_id:
                cluster_tracking += 1
            # Update clusters to include previously skipped levels
            if left_id and left_id[0] + 1 != level:
                clusters = self.fill_level(left_id, level, clusters)
            if right_id and right_id[0] + 1 != level:
                clusters = self.fill_level(right_id, level, clusters)

        return clusters

    def fill_level(self, cluster_id, level, clusters):
        new_level, cluster = cluster_id[0] + 1, cluster_id[1]
        cluster_info = {
            "cluster_merged": False,
            "count": clusters[new_level - 1][cluster]["count"],
            "avg_dist": clusters[new_level - 1][cluster]["avg_dist"],
            "dist_std": clusters[new_level - 1][cluster]["dist_std"],
            "samples": clusters[new_level - 1][cluster]["samples"],
            "sample_dist": clusters[new_level - 1][cluster]["sample_dist"],
            "outside_1-std": False,
            "outside_2-std": False,
        }
        # Sets each level's cluster info if it does not exist
        for level_id in range(level - 1, new_level - 2, -1):
            clusters[level_id].setdefault(cluster, cluster_info)

        return clusters
        # Only tracking the levels in which clusters merge for the cluster distance matrix

    def get_cluster_distances(self, clusters):
        # this is the cluster distance matrix
        cluster_matrix = np.full((self.last_merge_level, self.max_clusters, self.max_clusters), -1.0, dtype=np.float32)

        for level, cluster_set in clusters.items():
            if level < self.last_merge_level:
                cluster_ids = sorted(cluster_set.keys())
                for i, cluster_id in enumerate(cluster_ids):
                    cluster_matrix[level, cluster_id, cluster_id] = clusters[level][cluster_id]["avg_dist"]
                    for int_id in range(i + 1, len(cluster_ids)):
                        compare_id = cluster_ids[int_id]
                        sample_a = clusters[level][cluster_id]["samples"]
                        sample_b = clusters[level][compare_id]["samples"]
                        min_mat = self.sqdmat[np.ix_(sample_a, sample_b)].min()
                        cluster_matrix[level, cluster_id, compare_id] = min_mat
                        cluster_matrix[level, compare_id, cluster_id] = min_mat

        return cluster_matrix

    def get_merge_levels(self, clusters):
        """
        Runs through the clusters dictionary determining when clusters merge,
        and how close are those clusters when they merge.

        Parameters
        ----------
        clusters:
            A dictionary containing the original clusters information.

        Returns
        -------
        merge_clusters:
            A dictionary with each clusters merge history
        """

        merge_clusters = {"merge": {}, "likely_merge": {}, "no_merge": {}}

        for level, cluster_set in clusters.items():
            for cluster_id, cluster_info in cluster_set.items():
                merged = cluster_info["cluster_merged"]
                if not merged:
                    continue
                # Extract necessary information
                num_samples = len(cluster_info["samples"])
                out1 = cluster_info["outside_1-std"]
                out2 = cluster_info["outside_2-std"]

                value = [merged]
                if out2:
                    if num_samples < self.min_num_samples_per_cluster:
                        merge_key = "likely_merge"
                        value = value.append("low")
                    else:
                        merge_key = "no_merge"
                elif out1 and num_samples >= self.min_num_samples_per_cluster:
                    merge_key = "likely_merge"
                else:
                    merge_key = "merge"

                if cluster_id not in merge_clusters[merge_key]:
                    merge_clusters[merge_key][cluster_id] = {}
                if level not in merge_clusters[merge_key][cluster_id]:
                    merge_clusters[merge_key][cluster_id][level] = value

        return merge_clusters

    def cluster_merging(self, cluster_merges, cluster_matrix):
        intra_max = []
        merge_mean = []
        merge_list = []
        additional_check = []
        # Process each merge type
        for merge_type, merge_clusters in cluster_merges.items():
            for outer_cluster, inner_clusters in merge_clusters.items():
                for level, cluster_list in inner_clusters.items():
                    # Determine if there is a small far merge
                    additional_check.append(len(cluster_list) == 2)

                    inner_cluster = cluster_list[0]

                    # Get the slice of the distance matrix up to the level before merging
                    distances = cluster_matrix[:level, outer_cluster, inner_cluster]
                    # print(f"Negative check for {outer_cluster}-{inner_cluster} at {level} : {np.any(distances<0)}")
                    # if np.any(distances < 0):
                    #     print(distances)
                    intra_distance = cluster_matrix[:, outer_cluster, outer_cluster]
                    mask = intra_distance >= 0
                    intra_filtered = intra_distance[mask]
                    intra_max.append(np.max(intra_filtered))

                    # Grabbing the corresponding desired values
                    if merge_type == "merge":
                        merge_mean.append(np.max(distances))
                    else:
                        merge_mean.append(np.mean(distances))

                    merge_list.append([level, outer_cluster, inner_cluster])

        return merge_list, merge_mean, intra_max, additional_check

    def get_desired_merge(self, merge_mean, intra_max, additional_check):
        intra_max = np.unique(intra_max)
        intra_value = np.log(intra_max)
        intra_value = intra_value.mean() + 2 * intra_value.std()
        merge_value = np.log(merge_mean)
        desired_merge = merge_value < intra_value

        check = merge_value[~desired_merge]
        check = np.abs((check - intra_value) / intra_value)
        mask = check < 1
        good = check[mask].mean() + check[mask].std()
        merge = check < good
        return desired_merge, merge

    def run(self):
        sample_info = self.create_clusters()

        if self.max_clusters > 1:
            cluster_matrix = self.get_cluster_distances(sample_info)
            merge_levels = self.get_merge_levels(sample_info)
            merge_list = self.generate_merge_list(merge_levels, cluster_matrix)
            last_merge_levels = self.get_last_merge_levels(merge_list)
        else:
            last_merge_levels = {0: int(self.num_samples * 0.1)}
