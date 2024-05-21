import numpy as np
import numpy.testing as npt
import pytest
import sklearn.datasets as dsets

from daml._internal.metrics.clustering import (
    Cluster,
    Clusterer,
    ClusterPosition,
    extend_linkage,
)


def union_find(lis):
    lis = map(set, lis)
    unions = []
    for item in lis:
        temp = []
        for s in unions:
            if not s.isdisjoint(item):
                item = s.union(item)
            else:
                temp.append(s)
        temp.append(item)
        unions = temp
    return unions


def get_functional_data():
    blobs, _ = dsets.make_blobs(  # type: ignore
        n_samples=100,
        centers=[(-1.5, 1.8), (-1, 3), (0.8, 2.1), (2.8, 1.5), (2.5, 3.5)],  # type: ignore
        cluster_std=0.3,
        random_state=33,
    )
    # test_data = np.vstack([moons, blobs])
    functional_data = blobs
    functional_data[79] = functional_data[24]
    functional_data[63] = functional_data[58] + 1e-5

    return functional_data


def get_duplicate_data():
    x = np.ones(shape=(6, 2))
    return x


def get_outlier_data():
    blobs, _ = dsets.make_blobs(  # type: ignore
        n_samples=10,
        centers=[(-1.5, 1.8), (-1, 3), (0.8, 2.1), (2.8, 1.5), (2.5, 3.5)],  # type: ignore
        cluster_std=0.3,
        random_state=33,
    )
    # test_data = np.vstack([moons, blobs])
    test_data = blobs
    rand_indices = np.random.randint(0, 10, size=2)
    test_data[rand_indices] *= 3.0

    return test_data


@pytest.fixture
def functional_data():
    return get_functional_data()


@pytest.fixture
def duplicate_data():
    return get_duplicate_data()


@pytest.fixture
def outlier_data():
    return get_outlier_data()


class TestMatrixOps:
    @pytest.mark.parametrize(
        "shape",
        (
            (2, 0),  # No features
            (2, 1),  # Minimum size
            (4, 4),  # Square small
            (100, 2),  # High sample, low features
            (2, 100),  # Low samples, high features
            (100, 100),  # Square large
        ),
    )
    def test_matrices(self, shape):
        """Test sample size (rows), feature size (cols) and non-uniform shapes can create matrix"""
        rows, cols = shape
        rand_arr = np.random.random(size=(rows, cols))
        dup_arr = np.ones(shape=(rows, cols))

        test_sets = (rand_arr, dup_arr)

        for test_set in test_sets:
            c = Clusterer(test_set)

            # Distance matrix
            assert not any(np.isnan(c._darr))  # Should contain no NaN
            assert all(c._darr >= 0)  # Distances are always positive or 0 (same data point)
            assert len(c._darr) == rows * (rows - 1) / 2  #  condensed form of matrix

            # Square distance matrix
            assert len(set(c._sqdmat.shape)) == 1  # All dims are equal size
            assert np.all(c._sqdmat == c._sqdmat.T)  # Matrix is symmetrical

            # Extend function
            arr = np.ones(shape=(rows, cols))
            ext_matrix = extend_linkage(arr)
            assert ext_matrix.shape == (rows, cols + 1)  # Adds a column to the right
            # New column contains new max_row number of ids starting from max row + 1
            # i.e. [1, 2, 3] -> [[1, 2, 3], [4, 5, 6]]
            npt.assert_array_equal(ext_matrix[:, -1], np.arange(rows + 1, 2 * rows + 1))

            # Linkage arr
            assert not np.any(np.isnan(c._larr))  # Should contain no NaN

    @pytest.mark.parametrize(
        "shape",
        (
            (0, 0),  # Empty array
            (0, 1),  # No samples
            (1, 1),  # Minimum logical size
            (2, 2, 2),  # Invalid shape
            (10),  # Invalid shape
            (10,),  # Invalid shape
        ),
    )
    def test_matrices_invalid(self, shape):
        if isinstance(shape, int) or len(shape) == 1:
            rows, cols = shape, 0
        else:
            rows, cols = shape[0], shape[1]

        test_set = np.ones(shape=shape)
        with pytest.raises(ValueError):
            c = Clusterer(test_set)
            # Distance matrix
            assert not any(np.isnan(c._darr))  # Should contain no NaN
            assert all(c._darr >= 0)  # Distances are always positive or 0 (same data point)
            assert len(c._darr) == rows * (rows - 1) / 2  #  condensed form of matrix

            # Square distance matrix
            assert len(set(c._sqdmat.shape)) == 1  # All dims are equal size
            assert np.all(c._sqdmat == c._sqdmat.T)  # Matrix is symmetrical

            # Linkage arr
            assert not np.any(np.isnan(c._larr))  # Should contain no NaN

            # Extend function
            arr = np.ones(shape=(rows, cols))
            ext_matrix = extend_linkage(arr)

            assert ext_matrix.shape == (rows, cols + 1)  # Adds a column to the right
            # New column contains new max_row number of ids starting from max row + 1
            # i.e. [1, 2, 3] -> [[1, 2, 3], [4, 5, 6]]
            npt.assert_array_equal(ext_matrix[:, -1], np.arange(rows + 1, 2 * rows + 1))


class TestClusterer:
    def test_on_init(self, functional_data):
        """Tests that the init correctly sets the distance matrix and linkage array"""
        cl = Clusterer(functional_data)
        assert cl._num_samples is not None
        assert cl._darr is not None
        assert cl._sqdmat is not None
        assert cl._larr is not None
        assert cl._max_clusters is not None
        assert cl._min_num_samples_per_cluster is not None
        assert cl._clusters == {}

    @pytest.mark.parametrize(
        "shape",
        (
            (0, 0),  # Empty array
            (0, 1),  # No samples
            (1, 1),  # Minimum logical size
            (2, 2, 2),  # Invalid shape
            (10),  # Invalid shape
            (10,),  # Invalid shape
        ),
    )
    def test_on_init_fail(self, shape):
        dataset = np.ones(shape=shape)
        with pytest.raises(ValueError):
            Clusterer(dataset)

    def test_reset_results_on_new_data(self, functional_data, duplicate_data):
        """
        When new data is given to clusterer,
        recalculate the distance matrix and linkage arr
        """
        cl = Clusterer(functional_data)
        npt.assert_array_equal(cl._data, functional_data)

        cl.data = duplicate_data
        npt.assert_array_equal(cl._data, duplicate_data)

    def test_data_getter(self, functional_data):
        npt.assert_array_equal(Clusterer(functional_data).data, functional_data)

    @pytest.mark.parametrize("data_func", [get_functional_data, get_duplicate_data, get_outlier_data])
    def test_create_clusters(self, data_func):
        """
        Tests to confirm:
        1. All keys are present in outer and inner dicts
        2. Max level and max clusters are correct
        3. Distances are all positive
        4. All samples have a cluster
        """
        dataset = data_func()
        clusterer = Clusterer(dataset)
        clusterer.create_clusters()

        # Max level and max clusters are empirically correct
        n = len(dataset)
        assert clusterer._max_level <= n  # Max levels must be less than samples
        assert clusterer._max_clusters <= n // 2  # Minimum 2 samples for a valid cluster
        assert clusterer._max_level >= 1
        assert clusterer._max_clusters >= 1

        result_all_levels = {}
        result_all_clusters = {}

        all_samples = set()

        # Collect all samples in results to confirm they have been added to the dict
        for cluster_id_dict in clusterer._clusters.values():
            for cluster in cluster_id_dict.values():
                # All distances must be positive
                assert cluster.dist_avg >= 0
                assert cluster.dist_std >= 0
                assert np.all(cluster.sample_dist >= 0)

                samples = cluster.samples
                all_samples.update(set(samples))
            # Max of all clusters, checking at each level
            result_all_clusters.update(cluster_id_dict)
        result_all_levels = set(clusterer._clusters)

        result_max_cluster = max(result_all_clusters)
        result_max_level = max(result_all_levels)

        # Quick check that no levels or cluster_ids are skipped, both are 0-indexed
        assert len(result_all_levels) == result_max_level + 1
        assert len(result_all_clusters) == result_max_cluster + 1

        # Confirm that over all results, last level has all samples
        last_level_results = clusterer._clusters.get(result_max_level)
        assert last_level_results is not None
        # Should only contain one cluster
        last_level_cluster = list(last_level_results.values())
        assert len(last_level_cluster) == 1
        # Confirm final cluster contains all samples, and is the same as all samples found in results
        last_cluster_info = last_level_cluster[0]
        assert last_cluster_info.count == len(all_samples)
        assert sorted(last_cluster_info.samples) == sorted(all_samples)

        # result_max_cluster is 0-indexed, so adjust for total number
        assert clusterer._max_clusters == result_max_cluster + 1

        assert len(all_samples) == n


class TestClustererNoInit:
    clusterer = Clusterer(np.zeros((3, 1)))

    def test_fill_level(self):
        dummy_data = Cluster(False, np.ndarray([0]), 0.0, True)
        x = {
            0: {
                0: dummy_data,
                1: dummy_data,
            },
            1: {
                0: dummy_data,
            },
            2: {
                0: dummy_data,
            },
            3: {
                0: dummy_data,
            },
        }

        self.clusterer._clusters = x

        # Fill 0,1 up to 2,1
        self.clusterer._fill_levels(ClusterPosition(0, 1), ClusterPosition(2, 1))

        # Confirm cluster info has been placed into levels up to merge level (3)
        # assert cluster_1 in [0, 1, 2, 3)
        assert [1 in self.clusterer.clusters[i] for i in (0, 1, 2)]
        assert 1 not in self.clusterer.clusters[3]

    def test_get_cluster_distances(self):
        pass

    def test_calc_merge_indices(self):
        x = [np.array([1, 1.1, 1.2, 1.3, 5])]
        m = [5.0]
        merge_indices = self.clusterer._calc_merge_indices(x, m)

        assert len(merge_indices) == len(x)
        npt.assert_equal(merge_indices, np.array([[True, True, True, True, False]]))

    def test_calc_merge_indices_multidim(self):
        x = [np.array([1, 1.1, 1.2, 1.3, 5]), np.array([1, 1.1, 1.2, 1.3, 5]), np.array([1, 1.1, 1.2, 1.3, 5])]
        m = [5.0, 5.0, 5.0]
        merge_indices = self.clusterer._calc_merge_indices(x, m)

        assert len(merge_indices) == len(x)
        npt.assert_equal(
            merge_indices,
            np.array(
                [[True, True, True, True, False], [True, True, True, True, False], [True, True, True, True, False]]
            ),
        )

    def test_generate_merge_list(self):
        pass

    def test_get_last_merge_levels(self):
        pass

    def test_find_outliers(self):
        pass

    def test_calc_duplicate_std(self):
        pass

    def test_find_duplicates(self):
        pass


class TestClustererFunctional:
    """Tests individual dataset results for pseudo-functional, duplicate, and outlier data"""

    def test_run_results(self, functional_data):
        clusterer = Clusterer(functional_data)
        results = clusterer.run()

        assert results["outliers"] == [4, 6, 11, 21, 38, 71]
        assert results["potential_outliers"] == [1, 9, 42, 43, 48]
        assert results["duplicates"] == [[24, 79], [58, 63]]
        assert results["near_duplicates"] == [
            [8, 27, 29],
            [10, 65],
            [16, 99],
            [19, 64],
            [22, 87, 95],
            [33, 76],
            [39, 55],
            [40, 72],
            [41, 62],
            [80, 81, 93],
        ]

        # Sorts related tuples into sets
        setlist = union_find(results["near_duplicates"])
        # Calculates the length of all items in bins
        sum_lens = sum([len(s) for s in setlist])
        # Calculates all unique values in near_duplicates list
        unique_counts = {x for xs in results["near_duplicates"] for x in xs}
        # The length of all sets is the same as the unique values
        assert sum_lens == len(unique_counts)

    def test_duplicate_images(self, duplicate_data):
        cl = Clusterer(duplicate_data)

        results = cl.run()
        duplicates = results["duplicates"]
        near_duplicates = results["near_duplicates"]

        setlist = union_find(duplicates)
        # Only 1 set (all dupes) in list of sets
        assert len(setlist[0]) == len(duplicate_data)
        assert near_duplicates == []
