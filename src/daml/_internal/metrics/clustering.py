from typing import Dict, Iterable, List, NamedTuple, Tuple, Union, cast

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform


def extend_linkage(link_arr: np.ndarray) -> np.ndarray:
    """
    Adds a column to the linkage matrix link_arr that tracks the new id assigned
    to each row

    Parameters
    ----------
    link_arr : np.ndarray
        linkage matrix

    Returns
    -------
    np.ndarray
        linkage matrix with adjusted shape, new shape (link_arr.shape[0], link_arr.shape[1]+1)
    """
    # Adjusting linkage matrix to accommodate renumbering
    rows, cols = link_arr.shape
    arr = np.zeros((rows, cols + 1))
    arr[:, :-1] = link_arr
    arr[:, -1] = np.arange(rows + 1, 2 * rows + 1)

    return arr


class Cluster:
    __slots__ = "merged", "samples", "sample_dist", "is_copy", "count", "dist_avg", "dist_std", "out1", "out2"

    def __init__(self, merged: int, samples: np.ndarray, sample_dist: Union[float, np.ndarray], is_copy: bool = False):
        self.merged = merged
        self.samples = np.array(samples, dtype=np.int32)
        self.sample_dist = np.array([sample_dist] if np.isscalar(sample_dist) else sample_dist)
        self.is_copy = is_copy

        dist = float(self.sample_dist[-1])

        self.count = len(self.samples)
        if is_copy:
            self.dist_avg = 0.0
            self.dist_std = 0.0
            self.out1 = False
            self.out2 = False
        else:
            self.dist_avg = float(np.mean(self.sample_dist))
            self.dist_std = float(np.std(self.sample_dist)) if len(self.sample_dist) > 1 else 1e-5
            out1 = self.dist_avg + self.dist_std
            out2 = out1 + self.dist_std
            self.out1 = dist > out1
            self.out2 = dist > out2

    def copy(self) -> "Cluster":
        return Cluster(False, self.samples, self.sample_dist, True)

    def __repr__(self) -> str:
        _params = {
            "merged": self.merged,
            "samples": self.samples,
            "sample_dist": self.sample_dist,
            "is_copy": self.is_copy,
        }
        return f"{self.__class__.__name__}(**{repr(_params)})"


class ClusterPosition(NamedTuple):
    """Keeps track of a cluster's level and ID"""

    level: int
    cid: int


class ClusterMergeEntry:
    __slots__ = "level", "outer_cluster", "inner_cluster", "status"

    def __init__(self, level: int, outer_cluster: int, inner_cluster: int, status: int):
        self.level = level
        self.outer_cluster = outer_cluster
        self.inner_cluster = inner_cluster
        self.status = status

    def __lt__(self, value: "ClusterMergeEntry") -> bool:
        return self.level.__lt__(value.level)

    def __gt__(self, value: "ClusterMergeEntry") -> bool:
        return self.level.__gt__(value.level)


class Clusterer:
    """
    Uses hierarchical clustering to flag dataset properties of interest like outliers and duplicates

    Parameters
    ----------
    dataset : np.ndarray
        An array of images or image embeddings to perform clustering
    """

    def __init__(self, dataset: np.ndarray):
        # Allows an update to dataset to reset the state rather than instantiate a new class
        self._on_init(dataset)

    def _on_init(self, dataset: np.ndarray):
        self._validate_data(dataset)
        self._data: np.ndarray = dataset
        self._num_samples = len(dataset)

        self._darr: np.ndarray = pdist(dataset, metric="euclidean")
        self._sqdmat: np.ndarray = squareform(self._darr)
        self._larr: np.ndarray = extend_linkage(linkage(self._darr))
        self._max_clusters: int = np.count_nonzero(self._larr[:, 3] == 2)
        self._max_level: int = 1

        min_num = int(self._num_samples * 0.05)
        self._min_num_samples_per_cluster = min(max(2, min_num), 100)

        self._clusters: Dict[int, Dict[int, Cluster]] = {}  # Dictionary to store clusters
        self._last_good_merge_levels: Dict = {}

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, x: np.ndarray):
        self._on_init(x)

    @property
    def clusters(self) -> Dict[int, Dict[int, Cluster]]:
        if not self._clusters:
            self._clusters = self._create_clusters()
        return self._clusters

    @property
    def last_good_merge_levels(self) -> Dict[int, int]:
        if not self._last_good_merge_levels:
            self._last_good_merge_levels = self._get_last_merge_levels()
        return self._last_good_merge_levels

    @classmethod
    def _validate_data(cls, x: np.ndarray):
        """Checks that the data has the correct size, shape, and format"""
        if not isinstance(x, np.ndarray):
            raise TypeError(f"Data should be of type np.ndarray; got {type(x)}")

        if x.ndim != 2:
            raise ValueError(
                f"Data should only have 2 dimensions; got {x.ndim}. Data should be flattened before being input"
            )
        samples, features = x.shape  # Due to above check, we know shape has a length of 2
        if samples < 2:
            raise ValueError(f"Data should have at least 2 samples; got {samples}")
        if features < 1:
            raise ValueError(f"Samples should have at least 1 feature; got {features}")

    def _create_clusters(self) -> Dict[int, Dict[int, Cluster]]:
        """Generates clusters based on linkage matrix"""
        next_cluster_id = 0
        cluster_map: Dict[int, ClusterPosition] = {}  # Dictionary to associate new cluster ids with actual clusters
        clusters: Dict[int, Dict[int, Cluster]] = {}

        # Walking through the linkage array to generate clusters
        for arr_i in self._larr:
            left_id = int(arr_i[0])
            right_id = int(arr_i[1])
            sample_dist = np.array([arr_i[2]], dtype=np.float16)
            merged = False

            # Determine if the id is already associated with a cluster
            left = cluster_map.get(left_id)
            right = cluster_map.get(right_id)

            if left and right:
                merged = max([left.cid, right.cid])
                lc = clusters[left.level][left.cid]
                rc = clusters[right.level][right.cid]
                left_first = len(lc.samples) >= len(rc.samples)
                samples = np.concatenate([lc.samples, rc.samples] if left_first else [rc.samples, lc.samples])
                sample_dist = np.concatenate([rc.sample_dist, lc.sample_dist, sample_dist])
                level, cid = max(left.level, right.level) + 1, min(left.cid, right.cid)

                # Only tracking the levels in which clusters merge for the cluster distance matrix
                self._max_level = max(self._max_level, left.level, right.level)
                # Update clusters to include previously skipped levels
                clusters = self._fill_levels(clusters, left, right)
            elif left or right:
                child, other_id = cast(Tuple[ClusterPosition, int], (left, right_id) if left else (right, left_id))
                cc = clusters[child.level][child.cid]
                samples = np.concatenate([cc.samples, [other_id]])
                sample_dist = np.concatenate([cc.sample_dist, sample_dist])
                level, cid = child.level + 1, child.cid
            else:
                samples = np.array([left_id, right_id], dtype=np.int32)
                level, cid = 0, next_cluster_id
                next_cluster_id += 1

            # Set the cluster and associate the linkage id with the cluster
            if level not in clusters:
                clusters[level] = {}

            clusters[level][cid] = Cluster(merged, samples, sample_dist)
            cluster_map[int(arr_i[-1])] = ClusterPosition(level, cid)

        return clusters

    def _fill_levels(
        self, clusters: Dict[int, Dict[int, Cluster]], left: ClusterPosition, right: ClusterPosition
    ) -> Dict[int, Dict[int, Cluster]]:
        # Sets each level's cluster info if it does not exist
        if left.level != right.level:
            (level, cid), max_level = (left, right[0]) if left[0] < right[0] else (right, left[0])
            cluster = clusters[level][cid].copy()
            for level_id in range(max_level, level, -1):
                clusters[level_id].setdefault(cid, cluster)
        return clusters

    def _get_cluster_distances(self) -> np.ndarray:
        """Calculates the minimum distances between clusters are each level"""
        # Must call self.clusters before creating matrix because
        # self._max_level and self._max_clusters are calculated by self.clusters property
        clusters = self.clusters
        # Cluster distance matrix
        cluster_matrix = np.full((self._max_level, self._max_clusters, self._max_clusters), -1.0, dtype=np.float32)

        for level, cluster_set in clusters.items():
            if level < self._max_level:
                cluster_ids = sorted(cluster_set.keys())
                for i, cluster_id in enumerate(cluster_ids):
                    cluster_matrix[level, cluster_id, cluster_id] = clusters[level][cluster_id].dist_avg
                    for int_id in range(i + 1, len(cluster_ids)):
                        compare_id = cluster_ids[int_id]
                        sample_a = clusters[level][cluster_id].samples
                        sample_b = clusters[level][compare_id].samples
                        min_mat = self._sqdmat[np.ix_(sample_a, sample_b)].min()
                        cluster_matrix[level, cluster_id, compare_id] = min_mat
                        cluster_matrix[level, compare_id, cluster_id] = min_mat

        return cluster_matrix

    def _calc_merge_indices(self, merge_mean: List[np.ndarray], intra_max: List[float]) -> np.ndarray:
        """
        Determine what clusters should be merged and return their indices
        """
        intra_max_uniques = np.unique(intra_max)
        intra_log_values = np.log(intra_max_uniques)
        two_std_all = intra_log_values.mean() + 2 * intra_log_values.std()
        merge_value = np.log(merge_mean)
        # Mask of indices we know we want to merge
        desired_merge = merge_value < two_std_all

        # List[Values] for indices we might want to merge
        check = merge_value[~desired_merge]
        # Check distance from value to 2 stds of all values
        check = np.abs((check - two_std_all) / two_std_all)
        # Mask List[Values < 1]
        mask = check < 1
        one_std_check = check[mask].mean() + check[mask].std()
        # Mask of indices that should also be merged
        mask2_vals = np.abs((merge_value - two_std_all) / two_std_all)
        mask2 = mask2_vals < one_std_check
        return np.logical_or(desired_merge, mask2)

    def _generate_merge_list(self, cluster_matrix: np.ndarray) -> List[ClusterMergeEntry]:
        """
        Runs through the clusters dictionary determining when clusters merge,
        and how close are those clusters when they merge.

        Parameters
        ----------
        cluster_matrix:
            The distance matrix for all clusters to all others

        Returns
        -------
        List[ClusterMergeEntry]:
            A list with each cluster's merge history
        """
        intra_max = []
        merge_mean = []
        merge_list: List[ClusterMergeEntry] = []

        for level, cluster_set in self.clusters.items():
            for outer_cluster, cluster in cluster_set.items():
                inner_cluster = cluster.merged
                if not inner_cluster:
                    continue
                # Extract necessary information
                num_samples = len(cluster.samples)
                out1 = cluster.out1
                out2 = cluster.out2

                # If outside 2-std or 1-std and larger than a minimum sized cluster, take the mean distance, else max
                aggregate_func = (
                    np.mean if out2 or (out1 and num_samples >= self._min_num_samples_per_cluster) else np.max
                )

                distances = cluster_matrix[:level, outer_cluster, inner_cluster]
                intra_distance = cluster_matrix[:, outer_cluster, outer_cluster]
                positive_mask = intra_distance >= 0
                intra_filtered = intra_distance[positive_mask]

                # TODO: Append now, take max over axis later?
                intra_max.append(np.max(intra_filtered))
                # Calculate the corresponding distance stats
                distance_stats_arr = aggregate_func(distances)
                merge_mean.append(distance_stats_arr)
                merge_list.append(ClusterMergeEntry(level, outer_cluster, inner_cluster, 0))

        all_merge_indices = self._calc_merge_indices(merge_mean=merge_mean, intra_max=intra_max)

        for i, is_mergeable in enumerate(all_merge_indices):
            merge_list[i].status = is_mergeable

        merge_list = sorted(merge_list, reverse=True)

        return merge_list

    def _get_last_merge_levels(self) -> Dict[int, int]:
        """
        Creates a dictionary for important cluster ids mapped to their last good merge level

        Returns
        -------
        Dict[int, int]
            A mapping of a cluster id to its last good merge level
        """
        last_merge_levels: Dict[int, int] = {}

        if self._max_clusters <= 1:
            last_merge_levels = {0: int(self._num_samples * 0.1)}
        else:
            cluster_matrix = self._get_cluster_distances()
            merge_list = self._generate_merge_list(cluster_matrix)
            for entry in merge_list:
                if not entry.status:
                    if entry.outer_cluster not in last_merge_levels:
                        last_merge_levels[entry.outer_cluster] = 0
                    if entry.inner_cluster not in last_merge_levels:
                        last_merge_levels[entry.inner_cluster] = 0
                    if last_merge_levels[entry.outer_cluster] > entry.level:
                        last_merge_levels[entry.outer_cluster] = entry.level - 1
                else:
                    if entry.outer_cluster in last_merge_levels:
                        last_merge_levels[entry.outer_cluster] = max(
                            last_merge_levels[entry.outer_cluster], entry.level
                        )

        return last_merge_levels

    def find_outliers(self, last_merge_levels: Dict[int, int]) -> Tuple[List[int], List[int]]:
        """
        Retrieves outliers based on when the sample was added to the cluster
        and how far it was from the cluster when it was added

        Parameters
        ----------
        last_merge_levels : Dict[int, int]
            A mapping of a cluster id to its last good merge level

        Returns
        -------
        Tuple[List[int], List[int]]
            The outliers and possible outliers as sorted lists of indices
        """
        outliers = set()
        possible_outliers = set()

        for level, cluster_set in self.clusters.items():
            for cluster_id, cluster in cluster_set.items():
                if not cluster.merged and cluster_id in last_merge_levels and level > last_merge_levels[cluster_id]:
                    if cluster.out2:
                        outliers.add(cluster.samples[-1])
                    elif cluster.out1 and len(cluster.samples) >= self._min_num_samples_per_cluster:
                        possible_outliers.add(cluster.samples[-1])
                    elif len(cluster.samples) < self._min_num_samples_per_cluster:
                        outliers.update(cluster.samples.tolist())

        return sorted(outliers), sorted(possible_outliers)

    def _sorted_union_find(self, index_groups: Iterable[Iterable[int]]) -> List[List[int]]:
        """Merges and sorts groups of indices that share any common index"""
        groups: List[List[int]] = []
        for indices in zip(*index_groups):
            indices = set(indices)
            temp = []
            for group in groups:
                if not set(group).isdisjoint(indices):
                    indices.update(group)
                else:
                    temp.append(group)
            temp.append(sorted(indices))
            groups = temp
        return sorted(groups)

    def find_duplicates(self, last_merge_levels: Dict[int, int]) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Finds duplicate and near duplicate data based on the last good merge levels when building the cluster

        Parameters
        ----------
        last_merge_levels : Dict[int, int]
            A mapping of a cluster id to its last good merge level

        Returns
        -------
        Tuple[List[List[int]], List[List[int]]]
            The exact duplicates and near duplicates as lists of related indices
        """

        duplicates_std = []
        for cluster_id, level in last_merge_levels.items():
            samples = self.clusters[level][cluster_id].samples
            if len(samples) >= self._min_num_samples_per_cluster:
                duplicates_std.append(self.clusters[level][cluster_id].dist_std)
        diag_mask = np.ones_like(self._sqdmat, dtype=bool)
        np.fill_diagonal(diag_mask, 0)
        diag_mask = np.triu(diag_mask)

        exact_mask = self._sqdmat <= (np.mean(duplicates_std) / 100)
        exact_indices = np.nonzero(exact_mask & diag_mask)
        exact_dupes = self._sorted_union_find(exact_indices)

        near_mask = self._sqdmat <= np.mean(duplicates_std)
        near_indices = np.nonzero(near_mask & diag_mask & ~exact_mask)
        near_dupes = self._sorted_union_find(near_indices)

        return exact_dupes, near_dupes

    def evaluate(self):
        """Finds and flags indices of the data for outliers and duplicates

        Returns
        -------

        Dict[str, Union[List[int]], List[List[int]]]
            Dictionary containing list of outliers, potential outliers, duplicates, and near duplicates in keys
            "outliers", "potential_outliers", "duplicates", "near_duplicates" respectively
        """

        outliers, potential_outliers = self.find_outliers(self.last_good_merge_levels)
        duplicates, near_duplicates = self.find_duplicates(self.last_good_merge_levels)

        ret = {
            "outliers": outliers,
            "potential_outliers": potential_outliers,
            "duplicates": duplicates,
            "near_duplicates": near_duplicates,
        }

        return ret
