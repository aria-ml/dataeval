from unittest.mock import MagicMock

import numpy as np
import numpy.testing as npt
import pytest
import sklearn.datasets as dsets

from dataeval.detectors.linters.clusterer import (
    Clusterer,
    _Cluster,
    _ClusterMergeEntry,
    _ClusterPosition,
    _Clusters,
    _extend_linkage,
)


def get_blobs(std=0.3):
    blobs, _ = dsets.make_blobs(  # type: ignore
        n_samples=100,
        centers=[(-1.5, 1.8), (-1, 3), (0.8, 2.1), (2.8, 1.5), (2.5, 3.5)],  # type: ignore
        cluster_std=std,
        random_state=33,
    )
    return blobs


@pytest.fixture(scope="module")
def functional_data():
    functional_data = get_blobs()
    functional_data[79] = functional_data[24]
    functional_data[63] = functional_data[58] + 1e-5

    return functional_data


@pytest.fixture(scope="module")
def duplicate_data():
    x = np.ones(shape=(6, 2))
    return x


@pytest.fixture(scope="module")
def outlier_data():
    test_data = get_blobs()
    rng = np.random.default_rng(33)
    rints = rng.integers(0, 100, 2)
    test_data[rints] *= 10.0

    return test_data


@pytest.mark.required
class TestMatrixOps:
    @pytest.mark.parametrize(
        "shape",
        (
            (2, 1),  # Minimum size
            (4, 4),  # Square small
            (100, 2),  # High sample, low features
            (2, 100),  # Low samples, high features
            (100, 100),  # Square large
        ),
    )
    def test_matrices(self, shape):
        """Sample size (rows), feature size (cols) and non-uniform shapes can create matrix"""
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
            ext_matrix = _extend_linkage(arr)
            assert ext_matrix.shape == (rows, cols + 1)  # Adds a column to the right
            # New column contains new max_row number of ids starting from max row + 1
            # i.e. [1, 2, 3] -> [[1, 2, 3], [4, 5, 6]]
            npt.assert_array_equal(ext_matrix[:, -1], np.arange(rows + 1, 2 * rows + 1))

            # Linkage arr
            assert not np.any(np.isnan(c._larr))  # Should contain no NaN


@pytest.mark.required
class TestCluster:
    def test_init_not_copy(self):
        """Variables are calculated correctly when not copying"""
        c1 = _Cluster(merged=False, samples=[0, 1], sample_dist=[1, 4], is_copy=False)  # type: ignore

        assert isinstance(c1.samples, np.ndarray)
        assert c1.samples.dtype == np.int32

        assert isinstance(c1.sample_dist, np.ndarray)
        npt.assert_array_equal(c1.sample_dist, np.array([1, 4]))

        assert c1.dist_avg == 2.5
        assert c1.dist_std == 1.5
        assert not c1.out1
        assert not c1.out2

    def test_init_dist_is_scalar(self):
        """Special case where there is only one sample"""
        c1 = _Cluster(merged=False, samples=[0], sample_dist=0, is_copy=False)  # type: ignore

        assert isinstance(c1.samples, np.ndarray)
        assert c1.samples.dtype == np.int32

        assert isinstance(c1.sample_dist, np.ndarray)
        npt.assert_array_equal(c1.sample_dist, np.array([0]))

        assert c1.dist_avg == 0
        assert c1.dist_std == 1e-5
        assert not c1.out1
        assert not c1.out2

    def test_init_is_copy(self):
        """Default values are set for copied clusters"""
        # A copy ignores scalar sample_dist, and sets dist_std to 0 instead of 1e-5
        c1 = _Cluster(merged=False, samples=[0], sample_dist=0, is_copy=True)  # type: ignore

        assert isinstance(c1.samples, np.ndarray)
        assert c1.samples.dtype == np.int32

        assert isinstance(c1.sample_dist, np.ndarray)
        assert c1.sample_dist == np.array([0])

        assert c1.dist_avg == 0
        assert c1.dist_std == 0
        assert not c1.out1
        assert not c1.out2

    def test_cluster_copy(self):
        """A copied cluster retains parent's samples, but not other status"""
        c1 = _Cluster(merged=True, samples=[0, 1], sample_dist=[1, 4], is_copy=False)  # type: ignore

        c2 = c1.copy()

        assert c1.merged != c2.merged
        assert not c2.merged

        npt.assert_array_equal(c1.samples, c2.samples)
        npt.assert_array_equal(c1.sample_dist, c2.sample_dist)

        assert c1.dist_avg != c2.dist_avg  # 2.5 != 0.0
        assert c1.dist_std != c2.dist_std  # 1.5 != 0.0

        assert not c2.out1
        assert not c2.out2

    def test_cluster_repr(self):
        c1 = _Cluster(merged=False, samples=[0, 1], sample_dist=[1, 4], is_copy=False)  # type: ignore

        assert (
            c1.__repr__()
            == "_Cluster(**{'merged': False, 'samples': array([0, 1], dtype=int32), 'sample_dist': array([1, 4]), 'is_copy': False})"  # noqa: E501
        )

        # Numpy repr assumes direct numpy namespace imports
        from numpy import array, int32

        assert _Cluster(
            **{"merged": False, "samples": array([0, 1], dtype=int32), "sample_dist": array([1, 4]), "is_copy": False}
        )

        del array, int32  # Remove imports


@pytest.mark.required
class TestClusterPosition:
    def test_get_by_name(self):
        cp = _ClusterPosition(1, 1)
        assert cp.level
        assert cp.cid


@pytest.mark.required
class TestClusterMergeEntry:
    def test_arithmetic_ops(self):
        cme = _ClusterMergeEntry(1, 1, 1, True)

        cme_less = _ClusterMergeEntry(0, 10, 10, False)
        cme_more = _ClusterMergeEntry(10, 0, 0, False)

        assert cme_less < cme < cme_more
        assert cme_more > cme > cme_less


@pytest.mark.required
class TestClustererValidate:
    @pytest.mark.parametrize(
        "data, error, error_msg",
        [
            ([], TypeError, "Data should be of type NDArray; got <class 'list'>"),
            (
                np.array([1, 2, 3]),
                ValueError,
                "Data should only have 2 dimensions; got 1. Data should be flattened before being input",
            ),
            (
                np.array([[[0]]]),
                ValueError,
                "Data should only have 2 dimensions; got 3. Data should be flattened before being input",
            ),
            (np.array([[1]]), ValueError, "Data should have at least 2 samples; got 1"),  # samples < 2
            (np.array([[], []]), ValueError, "Samples should have at least 1 feature; got 0"),  # features < 1
        ],
    )
    def test_invalid(self, data, error, error_msg):
        with pytest.raises(error) as e:
            Clusterer._validate_data(data)
        assert e.value.args[0] == error_msg

    def test_valid(self):
        data = np.ones((2, 1, 2, 3, 4))
        Clusterer._validate_data(data.reshape((2, -1)))


@pytest.mark.required
class TestClustererInit:
    def test_on_init(self, functional_data):
        """Tests that the init correctly sets the distance matrix and linkage array"""
        cl = Clusterer(functional_data)
        assert cl._num_samples == 100
        assert cl._darr.shape == (4950,)
        assert cl._sqdmat.shape == (100, 100)
        assert cl._larr.shape == (99, 5)
        assert cl._max_clusters == 29
        assert cl._min_num_samples_per_cluster == 5
        assert cl._clusters is None
        assert cl._last_good_merge_levels is None

    @pytest.mark.parametrize(
        "shape",
        (
            (0, 0),  # Empty array
            (0, 1),  # No samples
            (1, 1),  # Minimum logical size
        ),
    )
    def test_on_init_fail(self, shape):
        """Invalid shapes and dims raise ValueError"""
        dataset = np.ones(shape=shape)
        with pytest.raises(ValueError):
            Clusterer(dataset)

    @pytest.mark.parametrize(
        "shape",
        [
            (100, 10),
            (10, 10, 10),
            (100, 2),
            (10),  # Invalid shape
            (10,),  # Invalid shape
        ],
    )
    def test_on_init_good_dims(self, shape):
        Clusterer(np.ones(shape=shape))

    def test_reset_results_on_new_data(self, functional_data, duplicate_data):
        """The distance matrix and linkage arr are recalculated when new data is given to clusterer"""
        cl = Clusterer(functional_data)
        npt.assert_array_equal(cl._data, functional_data)

        cl.data = duplicate_data
        npt.assert_array_equal(cl._data, duplicate_data)

    def test_data_getter(self, functional_data):
        """The underlying data variable is properly set and retrieved after initialization"""
        npt.assert_array_equal(Clusterer(functional_data).data, functional_data)


@pytest.mark.required
class TestCreateClusters:
    """Tests all functions related to and including the `create_clusters` method"""

    clusterer = Clusterer(np.zeros((3, 1)))

    def test_fill_level(self):
        """Merged clusters fill levels with missing cluster data"""
        dummy_data = _Cluster(False, np.ndarray([0]), 0.0, True)
        x = _Clusters(
            {
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
        )

        # Fill 0,1 up to 2,1
        filled_clusters = self.clusterer._fill_levels(x, _ClusterPosition(0, 1), _ClusterPosition(2, 1))

        # Confirm cluster info has been placed into levels up to merge level (3)
        # assert cluster_1 in [0, 1, 2, 3)
        assert [1 in filled_clusters[i] for i in (0, 1, 2)]
        assert 1 not in filled_clusters[3]

    @pytest.mark.parametrize("data_func", ["functional_data", "duplicate_data", "outlier_data"])
    def test_create_clusters(self, data_func, request):
        """
        1. All keys are present in outer and inner dicts
        2. Max level and max clusters are correct
        3. Distances are all positive
        4. All samples have a cluster
        """
        dataset = request.getfixturevalue(data_func)
        clusterer = Clusterer(dataset)
        # Calling clusterer.clusters calls _create_clusters if _clusters is empty
        clusters = clusterer.clusters

        # Max level and max clusters are empirically correct
        n = len(dataset)
        assert clusterer._max_clusters >= 1
        assert clusterer._max_clusters <= n // 2  # Minimum 2 samples for a valid cluster
        assert clusters.max_level >= 1
        assert clusters.max_level <= n  # Max levels must be less than samples

        result_all_levels = {}
        result_all_clusters = {}

        all_samples = set()

        # Collect all samples in results to confirm they have been added to the dict
        for cluster_id_dict in clusters.values():
            for cluster in cluster_id_dict.values():
                # All distances must be positive
                assert cluster.dist_avg >= 0
                assert cluster.dist_std >= 0
                assert np.all(cluster.sample_dist >= 0)

                samples = cluster.samples
                all_samples.update(set(samples))
            # Max of all clusters, checking at each level
            result_all_clusters.update(cluster_id_dict)
        result_all_levels = set(clusters)  # type: ignore

        result_max_cluster = max(result_all_clusters)
        result_max_level = max(result_all_levels)

        # Quick check that no levels or cluster_ids are skipped, both are 0-indexed
        assert len(result_all_levels) == result_max_level + 1
        assert len(result_all_clusters) == result_max_cluster + 1

        # Confirm that over all results, last level has all samples
        last_level_results = clusters.get(result_max_level)
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

    def test_skip_create_clusters(self, functional_data):
        c = Clusterer(functional_data)
        x = {0: {0: _Cluster(0, np.array([0]), np.array([0]))}}
        c._clusters = x  # type: ignore

        assert c.clusters == x


@pytest.mark.required
class TestClusterOutliers:
    """Tests all functions related to and including the `find_outliers` method"""

    @pytest.mark.parametrize(
        "cluster, outs, pouts",
        [
            # out2 is True; add last sample to outliers
            (_Cluster(0, np.arange(10), np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 11])), [9], []),
            # out1 is True and len(cluster.samples) >= min_num; add last sample to possible_outliers
            (_Cluster(0, np.arange(5), np.array([1, 1, 1, 1, 6])), [], [4]),
            # len(cluster.samples) < self.min_num; add all samples to outliers
            (_Cluster(0, np.arange(3), np.zeros((3, 1))), [0, 1, 2], []),
        ],
    )
    def test_find_outliers(self, cluster: _Cluster, outs, pouts):
        """Specified outliers are added to lists"""

        x = _Clusters({1: {0: cluster}})
        last_merge_levels = {0: 0}

        c = Clusterer(np.zeros((3, 1)))
        c._clusters = x
        c._min_num_samples_per_cluster = 4

        o, po = c.find_outliers(last_merge_levels=last_merge_levels)

        assert o == outs
        assert po == pouts

    @pytest.mark.parametrize(
        "mid, cid, merge_lvl, cluster",
        [
            (0, 0, 0, _Cluster(1, np.array([0]), 0.0)),  # merged
            (0, 1, 0, _Cluster(0, np.array([0]), 0.0)),  # cluster_id not in last_merge_levels
            (0, 0, 2, _Cluster(0, np.array([0]), 0.0)),  # merge_level > level (1)
            (0, 0, 0, _Cluster(0, np.arange(3), np.zeros((3, 1)))),  # No outliers
        ],
    )
    def test_no_outliers(self, mid, cid, merge_lvl, cluster):
        """No outliers are found"""
        x = _Clusters({1: {cid: cluster}})

        last_merge_levels = {mid: merge_lvl}

        c = Clusterer(np.zeros((3, 1)))
        c._clusters = x
        c._min_num_samples_per_cluster = 2

        o, po = c.find_outliers(last_merge_levels=last_merge_levels)

        assert not o and not po  # Both lists empty

    @pytest.mark.parametrize(
        "indices",
        [
            list(range(4)),
            list(range(4)) * -1,
            np.random.randint(0, 100, size=4),
        ],
    )
    def test_outliers(self, indices):
        """Integration test: `Clusterer` finds outlier data"""
        data = get_blobs(0.1)
        data[indices] *= 10.0
        c = Clusterer(np.array(data))
        results = c.evaluate()
        outliers = results.outliers

        # Only need to check specified outliers in results, but there might be other outliers
        assert all(x in outliers for x in indices)


@pytest.mark.required
class TestClusterDuplicates:
    """Tests all functions related to and including the `find_duplicates` method"""

    def test_group_pairs(self):
        """Group pairs merges overlapping sets"""
        c = Clusterer(np.ones((2, 1)))
        a = [0, 0, 1, 3, 3, 4]
        b = [1, 2, 2, 4, 5, 5]
        pairs = c._sorted_union_find([a, b])
        assert pairs == [[0, 1, 2], [3, 4, 5]]

    def test_group_multi(self):
        clusterer = Clusterer(np.ones((3, 2)))

        a = [0, 0, 5, 5, 9, 11]
        b = [1, 3, 6, 6, 10, 11]
        c = [2, 4, 7, 8, 11, 11]

        groups = clusterer._sorted_union_find([a, b, c])
        assert groups == [[0, 1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11]]

    def test_duplicates(self, duplicate_data):
        """`Clusterer` finds duplicate data during evaluate"""
        cl = Clusterer(duplicate_data)

        results = cl.evaluate()
        duplicates = results.duplicates
        potential_duplicates = results.potential_duplicates

        # Only 1 set (all dupes) in list of sets
        assert len(duplicates[0]) == len(duplicate_data)
        assert potential_duplicates == []

    def test_no_duplicates(self):
        """`Clusterer` finds no :term:`duplicates<Duplicates>` during evaluate"""
        data = np.array([[0, 0], [1, 1], [2, 2]])
        c = Clusterer(data)
        results = c.evaluate()
        assert not len(results.duplicates)
        assert not len(results.potential_duplicates)


@pytest.mark.required
class TestClustererGetLastMergeLevels:
    """Tests all functions related to and including the `get_last_merge_levels` method"""

    clusterer = Clusterer(np.zeros((3, 1)))

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

    def test_get_last_merge_levels_max_clusters_one(self):
        c = Clusterer(np.ones((2, 1)))  # Max clusters == 1
        assert c.last_good_merge_levels == {0: 0}

    def test_get_last_merge_levels(self):
        c = Clusterer(np.ones([3, 2]))
        c._max_clusters = 2

        merge_list = [
            _ClusterMergeEntry(-1, 0, 1, 0),  # level=-1, forces entry.level-1
            _ClusterMergeEntry(1, 0, 2, 1),  # Changes level back to 1
        ]

        c._get_cluster_distances = MagicMock()
        c._generate_merge_list = MagicMock()
        c._generate_merge_list.return_value = merge_list

        x = c._get_last_merge_levels()

        assert c._get_cluster_distances.call_count == 1
        assert c._generate_merge_list.call_count == 1

        assert x.get(0) == 1  # Outer cluster
        assert x.get(1) == 0  # Inner cluster


@pytest.mark.optional
class TestClustererEvaluate:
    """Tests the evaluate function with known dataset and results"""

    def test_evaluate_functional(self, functional_data):
        """The results of evaluate are equivalent to the known outputs"""
        clusterer = Clusterer(functional_data)
        results = clusterer.evaluate()

        assert results.outliers == [4, 6, 11, 21, 38, 71]
        assert results.potential_outliers == [1, 9, 42, 43, 48]
        assert results.duplicates == [[24, 79], [58, 63]]
        assert results.potential_duplicates == [
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
