from copy import deepcopy
from typing import Any, Dict, Tuple

import numpy as np
import sklearn.datasets as dsets
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform


def get_distance_matrix(data):
    return pdist(data, metric="euclidean")


def get_linkage_arr(distance_matrix):
    return linkage(distance_matrix, method="single")


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


def get_extended_linkage(distance_matrix):
    link_arr = get_linkage_arr(distance_matrix)
    return extend_linkage(link_arr)


class Clusterer:
    def __init__(self, dataset: np.ndarray):
        self._on_init(dataset)

    def _on_init(self, x):
        self._data: np.ndarray = x
        self.dmat: np.ndarray = get_distance_matrix(x)
        self.larr: np.ndarray = get_extended_linkage(self.dmat)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, x: np.ndarray):
        self._on_init(x)

    def create_clusters(self) -> Tuple[Dict[str, Any], int, int]:
        """Generates clusters based on linkage matrix

        Returns
        -------
        dict[str, Any]
            Cluster information
        int
            Max level count
        int
            Max cluster count
        """
        max_clusters = 1
        max_levels = 1
        clusters = {}

        for i, arr_i in enumerate(self.larr):
            level = 1
            cluster_num = max_clusters

            left_count = 0
            right_count = 0
            merged = False

            arr_0 = int(arr_i[0])
            arr_1 = int(arr_i[1])
            dist = arr_i[2]

            new_sample = []
            sample_dist = np.array([dist], dtype=np.float16)

            # Cluster left
            cluster_0 = clusters.get(arr_0)
            if cluster_0 is None:
                new_sample.append(arr_0)
            else:
                left_cluster = cluster_0["cluster_num"]
                level = max([level, cluster_0["level"] + 1])
                left_count = cluster_0["count"]
                left_sample = cluster_0["samples"]
                sample_dist = np.concatenate([cluster_0["sample_dist"], sample_dist])

            # Cluster right
            cluster_1 = clusters.get(arr_1)
            if cluster_1 is None:
                new_sample.append(arr_1)
            else:
                right_cluster = cluster_1["cluster_num"]
                level = max([level, cluster_1["level"] + 1])
                right_count = cluster_1["count"]
                right_sample = cluster_1["samples"]
                sample_dist = np.concatenate([cluster_1["sample_dist"], sample_dist])

            # Aggregate samples
            if cluster_0 and cluster_1:
                if left_count > right_count:
                    samples = np.concatenate([left_sample, right_sample])
                else:
                    samples = np.concatenate([right_sample, left_sample])
                cluster_num = min([left_cluster, right_cluster])
                merged = max([left_cluster, right_cluster])
            elif cluster_0:
                samples = np.concatenate([left_sample, new_sample])
                cluster_num = left_cluster
            elif cluster_1:
                samples = np.concatenate([right_sample, new_sample])
                cluster_num = right_cluster
            else:
                samples = np.array(new_sample, dtype=np.int32)

            dist_avg = np.mean(sample_dist)
            dist_std = np.std(sample_dist) if sample_dist.shape[0] > 1 else 0

            clusters[int(arr_i[-1])] = {
                "cluster_num": cluster_num,
                "level": level,
                "count": samples.shape[0],
                "avg_dist": dist_avg,
                "dist_std": dist_std,
                "samples": samples,  # adjusted this to keep tract of all samples in the list
                "sample_dist": sample_dist,
                "cluster_merged": merged,
            }

            # if cluster_num == max_clusters:
            #     max_clusters += 1
            max_clusters = max(max_clusters, cluster_num)
            max_levels = max(max_levels, level)

        # We don't care about the last level where everything is in a single cluster
        return clusters, max_levels, max_clusters

    def reorganize_clusters(self, clusters, min_num_samples_per_cluster=7):
        """
        Reorganize the clusters dictionary to be nested by level, then by cluster_num,
        and include avg_dist, sample_dist, and samples within each level.

        Parameters
        ----------
        clusters:
            A dictionary containing the original clusters information.

        Returns
        -------
        new_structure:
            A dictionary reorganized by level then by cluster_num, with details.
        merge_clusters:
            A dictionary with each clusters merge history
        """
        new_structure = {}
        merge_clusters = {"merge": {}, "likely_merge": {}, "no_merge": {}}
        outliers = []
        possible_outliers = []

        for _, info in clusters.items():
            # Extract necessary information
            cluster_num = info["cluster_num"]
            level = info["level"]
            samples = info["samples"]
            avg_dist = info["avg_dist"]
            dist_std = info["dist_std"]
            samp_dist = info["sample_dist"][-1]
            merged = info["cluster_merged"]

            out1 = avg_dist + dist_std
            out2 = out1 + dist_std

            # Initialize the structure if not present
            if level not in new_structure:
                new_structure[level] = {}

            # If cluster num already processed, skip
            if cluster_num in new_structure[level]:
                continue

            new_structure[level][cluster_num] = {
                "samples": samples,
                "avg_dist": avg_dist,
                "dist_std": dist_std,
                "added_samp_dist": samp_dist,
                "outside_1-std": samp_dist > out1,
                "outside_2-std": samp_dist > out2,
                "contains_clusters": [],
            }

            if merged:
                for i in range(2, level):
                    if merged not in new_structure[i]:
                        new_structure[i][merged] = {
                            "samples": new_structure[i - 1][merged]["samples"],
                            "avg_dist": new_structure[i - 1][merged]["avg_dist"],
                            "dist_std": new_structure[i - 1][merged]["dist_std"],
                            "contains_clusters": new_structure[i - 1][merged]["contains_clusters"],
                        }
                    if cluster_num not in new_structure[i]:
                        new_structure[i][cluster_num] = {
                            "samples": new_structure[i - 1][cluster_num]["samples"],
                            "avg_dist": new_structure[i - 1][cluster_num]["avg_dist"],
                            "dist_std": new_structure[i - 1][cluster_num]["dist_std"],
                            "contains_clusters": new_structure[i - 1][cluster_num]["contains_clusters"],
                        }

                if samp_dist > out2:
                    if len(samples) < min_num_samples_per_cluster:
                        if cluster_num not in merge_clusters["likely_merge"]:
                            merge_clusters["likely_merge"][cluster_num] = {level: (merged, "low sample count")}
                        if level not in merge_clusters["likely_merge"][cluster_num]:
                            merge_clusters["likely_merge"][cluster_num][level] = (
                                merged,
                                "low sample count",
                            )
                    else:
                        if cluster_num not in merge_clusters["no_merge"]:
                            merge_clusters["no_merge"][cluster_num] = {level: merged}
                        if level not in merge_clusters["no_merge"][cluster_num]:
                            merge_clusters["no_merge"][cluster_num][level] = merged
                elif samp_dist > out1 and len(samples) >= min_num_samples_per_cluster:
                    if cluster_num not in merge_clusters["likely_merge"]:
                        merge_clusters["likely_merge"][cluster_num] = {level: merged}
                    if level not in merge_clusters["likely_merge"][cluster_num]:
                        merge_clusters["likely_merge"][cluster_num][level] = merged
                else:
                    if cluster_num not in merge_clusters["merge"]:
                        merge_clusters["merge"][cluster_num] = {level: merged}
                    if level not in merge_clusters["merge"][cluster_num]:
                        merge_clusters["merge"][cluster_num][level] = merged

            else:
                if samp_dist > out2:
                    outliers.append((samples[-1], cluster_num, level))
                elif (
                    samp_dist > out1
                    and len(samples) >= min_num_samples_per_cluster
                    and cluster_num in merge_clusters["likely_merge"]
                ):
                    check = sorted(merge_clusters["likely_merge"][cluster_num].keys())
                    if level > check[0]:
                        possible_outliers.append((samples[-1], cluster_num, level))

            if level != 1:
                new_structure[level][cluster_num]["contains_clusters"] = deepcopy(
                    new_structure[level - 1][cluster_num]["contains_clusters"]
                )
                if merged:
                    prev_cluster_mergings = deepcopy(new_structure[level - 1][merged]["contains_clusters"])
                    if prev_cluster_mergings:
                        new_structure[level][cluster_num]["contains_clusters"].extend(prev_cluster_mergings)
                    new_structure[level][cluster_num]["contains_clusters"].append(merged)

        return new_structure, merge_clusters, outliers, possible_outliers

    def get_cluster_distances(self, clusters, max_clusters, max_levels, square_distance_matrix):
        # this is the cluster distance matrix without the final level and cluster
        cluster_matrix = np.full((max_levels - 1, max_clusters - 1, max_clusters - 1), -1.0, dtype=np.float32)

        for level, cluster_set in clusters.items():
            #     if level <= max_levels:
            #         # print(f"\t\tLevel {level-1}")
            cluster_ids = sorted(cluster_set.keys())
            # print("List of ids: ", cluster_ids)
            for i, cluster_id in enumerate(cluster_ids):
                cluster_id = cluster_id
                # print("Selected cluster ", cluster_id)
                cluster_matrix[level - 1, cluster_id - 1, cluster_id - 1] = clusters[level][cluster_id]["avg_dist"]
                for int_id in range(i + 1, len(cluster_ids)):
                    compare_id = cluster_ids[int_id]
                    # print(f"\tComparing cluster {compare_id}")
                    sample_a = clusters[level][cluster_id]["samples"]
                    sample_b = clusters[level][compare_id]["samples"]
                    # print(f"\t\tSamples A: {sample_a}, Samples B: {sample_b}")
                    min_mat = square_distance_matrix[np.ix_(sample_a, sample_b)].min()
                    cluster_matrix[level - 1, cluster_id - 1, compare_id - 1] = min_mat
                    cluster_matrix[level - 1, compare_id - 1, cluster_id - 1] = min_mat

        return cluster_matrix

    def merge_clusters(self, cluster_merges, cluster_matrix):
        intra_max = []
        merge_mean = []
        merge_list = []
        # Process each merge type
        for merge_type, merge_clusters in cluster_merges.items():
            for outer_cluster, inner_clusters in merge_clusters.items():
                for level, inner_cluster in inner_clusters.items():
                    # Convert to zero-based index
                    outer_idx = outer_cluster - 1
                    inner_idx = inner_cluster - 1
                    level_idx = level - 1

                    # Get the slice of the distance matrix up to the level before merging
                    distances = cluster_matrix[:level_idx, outer_idx, inner_idx]
                    intra_distance = cluster_matrix[:, outer_idx, outer_idx]
                    mask = intra_distance >= 0
                    intra_filtered = intra_distance[mask]
                    intra_max.append(np.max(intra_filtered))

                    # Grabbing the corresponding desired values
                    if merge_type == "merge":
                        merge_mean.append(np.max(distances))
                    else:
                        merge_mean.append(np.mean(distances))

                    merge_list.append([level, outer_cluster, inner_cluster])

        return merge_list, merge_mean, intra_max

    def get_desired_merge(self, merge_mean, intra_max):
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

    def get_last_merge_levels(self, merge_list):
        last_good_merge_levels = {}
        for entry in merge_list:
            level, outer_cluster, inner_cluster, status = entry
            if status == "no-merge":
                if outer_cluster not in last_good_merge_levels:
                    last_good_merge_levels[outer_cluster] = 1
                if inner_cluster not in last_good_merge_levels:
                    last_good_merge_levels[inner_cluster] = 1
                if last_good_merge_levels[outer_cluster] > level:
                    last_good_merge_levels[outer_cluster] = level - 1
            else:
                if outer_cluster in last_good_merge_levels:
                    last_good_merge_levels[outer_cluster] = max(last_good_merge_levels[outer_cluster], level)
        return last_good_merge_levels

    def generate_merge_list(self, cluster_merges, cluster_matrix):
        merge_list, merge_mean, intra_max = self.merge_clusters(cluster_merges, cluster_matrix)
        desired_merge, merge = self.get_desired_merge(merge_mean, intra_max)

        j = 0
        for i, select in enumerate(desired_merge):
            if select:
                merge_list[i].append("merge")
            else:
                if merge[j]:
                    merge_list[i].append("merge")
                else:
                    merge_list[i].append("no-merge")
                j += 1

        merge_list = sorted(merge_list, reverse=True)
        return merge_list

    def fill_dedup_std_list(self, last_merge_levels):
        pass

    def find_duplicates(self, square_distance_matrix, dedup_std_list):
        diag_mask = np.ones(square_distance_matrix.shape, dtype=bool)
        np.fill_diagonal(diag_mask, 0)  # this needs to change to be a mask of the diagonal
        diag_mask = np.triu(diag_mask)

        exact_mask = square_distance_matrix < (np.mean(dedup_std_list) / 100)
        exact_indices = np.nonzero(exact_mask & diag_mask)
        exact_dedup = list(zip(exact_indices[0], exact_indices[1]))

        possible_mask = square_distance_matrix < np.mean(dedup_std_list)
        possible_indices = np.nonzero(possible_mask & diag_mask & ~exact_mask)
        possible_dedup = list(zip(possible_indices[0], possible_indices[1]))

        return exact_dedup, possible_dedup

    def filter_outliers(self, outliers, last_merge_levels):
        # Outliers: List[(sample number, outer_cluster, level), (...)]
        # last_merge_levels: {cluster: level}
        filtered_outliers = []
        for outlier in outliers:
            sample_number, outer_cluster, level = outlier
            if outer_cluster in last_merge_levels and level >= last_merge_levels[outer_cluster] + 1:
                filtered_outliers.append(sample_number)
        return filtered_outliers

    def run(self):
        sample_info, max_levels, max_clusters = self.create_clusters()
        clusters_per_level, merge_groups, outliers, potential_outliers = self.reorganize_clusters(sample_info)
        print(f"Clusters per level: {clusters_per_level}")
        square_distance_matrix = squareform(self.dmat)
        cluster_matrix = self.get_cluster_distances(
            clusters_per_level, max_clusters, max_levels, square_distance_matrix
        )
        print("Cluster distances:", cluster_matrix)
        merge_list = self.generate_merge_list(merge_groups, cluster_matrix)
        last_merge_levels = self.get_last_merge_levels(merge_list)

        outliers = self.filter_outliers(outliers, last_merge_levels)
        potential_outliers = self.filter_outliers(potential_outliers, last_merge_levels)

        dedup_std = []
        min_num_samples_per_cluster = 7
        for cluster, level in last_merge_levels.items():
            level_cluster = clusters_per_level[level][cluster]
            samples = level_cluster["samples"]
            if samples.shape[0] < min_num_samples_per_cluster:
                outliers.extend(samples.tolist())
            else:
                dedup_std.append(level_cluster["dist_std"])

        duplicates, near_duplicates = self.find_duplicates(square_distance_matrix, dedup_std)

        ret = {
            "outliers": outliers,
            "potential_outliers": potential_outliers,
            "duplicates": duplicates,
            "near_duplicates": near_duplicates,
        }

        return ret


def test_outliers(x):
    assert x == [21, 6, 4, 71, 38, 11]
    print("Passed")


def test_potential_outliers(x):
    assert x == [42, 48, 9, 1, 43]
    print("Passed")


def test_duplicates(x):
    assert x == [(24, 79), (58, 63)]
    print("Passed")


def test_near_duplicates(x):
    assert x == [
        (8, 27),
        (10, 65),
        (16, 99),
        (19, 64),
        (22, 87),
        (27, 29),
        (33, 76),
        (39, 55),
        (40, 72),
        (41, 62),
        (80, 81),
        (80, 93),
        (81, 93),
        (87, 95),
    ]
    print("Passed")


def run_tests(x):
    test_outliers(x["outliers"])
    test_potential_outliers(x["potential_outliers"])
    test_duplicates(x["duplicates"])
    test_near_duplicates(x["near_duplicates"])


if __name__ == "__main__":
    plot_kwds = {"alpha": 0.5, "s": 50, "linewidths": 0}

    # moons, _ = dsets.make_moons(n_samples=50, noise=0.1)
    blobs, _ = dsets.make_blobs(  # type: ignore
        n_samples=100,
        centers=[(-1.5, 1.8), (-1, 3), (0.8, 2.1), (2.8, 1.5), (2.5, 3.5)],  # type: ignore
        cluster_std=0.3,
        random_state=33,
    )
    # test_data = np.vstack([moons, blobs])
    test_data = blobs
    test_data[79] = test_data[24]
    test_data[63] = test_data[58] + 1e-5

    clusterer = Clusterer(test_data)
    x = clusterer.run()
    for k, v in x.items():
        print(f"{k}: {v}")

    run_tests(x)
