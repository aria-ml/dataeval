import numpy as np
import numpy.testing as npt
import pytest
import sklearn.datasets as dsets

from daml._prototype.clusterer import (
    Clusterer,
    extend_linkage,
    get_distance_matrix,
    get_extended_linkage,
    get_linkage_arr,
)


@pytest.fixture
def func_data():
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

    return test_data


@pytest.fixture
def dupe_data():
    # rand_inds = np.random.randint(0, 25, size=(5))
    x = np.ones(shape=(25, 2))
    # x[rand_inds] += 1
    return x


@pytest.fixture
def outlier_data():
    blobs, _ = dsets.make_blobs(  # type: ignore
        n_samples=100,
        centers=[(-1.5, 1.8), (-1, 3), (0.8, 2.1), (2.8, 1.5), (2.5, 3.5)],  # type: ignore
        cluster_std=0.3,
        random_state=33,
    )
    # test_data = np.vstack([moons, blobs])
    test_data = blobs
    rand_indices = np.random.randint(0, 100, size=5)
    test_data[rand_indices] *= 3.0

    return test_data


class TestMatrixOps:
    @pytest.mark.parametrize(
        "rows, cols",
        (
            (2, 1),
            (4, 4),
            (100, 2),
            (100, 100),
        ),
    )
    def test_matrices(self, rows, cols):
        # PyTest parameterize doesn't like fixtures, so manually creating
        # random data, duplicate data, and outlier data
        rand_arr = np.random.random(size=(rows, cols))
        dup_arr = np.ones(shape=(rows, cols))

        test_sets = (rand_arr, dup_arr)

        for test_set in test_sets:
            # Distance matrix

            dm = get_distance_matrix(test_set)
            assert not any(np.isnan(dm))
            assert all(dm >= 0)
            assert len(dm) == rows * (rows - 1) / 2  #  condensed form

            # Linkage arr
            Z = get_linkage_arr(dm)
            assert Z.shape == (rows - 1, 4)
            assert not np.any(np.isnan(Z))

            # Extend function
            arr = np.ones(shape=(rows, cols))
            ext_matrix = extend_linkage(arr)

            assert ext_matrix.shape == (rows, cols + 1)
            npt.assert_array_equal(ext_matrix[:, -1], np.arange(rows + 1, 2 * rows + 1))


class TestClusterer:
    def test_init(self, func_data):
        """Tests that the init correctly sets the distance matrix and linkage array"""

        cl = Clusterer(func_data)

        ans_dmat = get_distance_matrix(func_data)
        ans_larr = get_extended_linkage(ans_dmat)

        npt.assert_array_equal(cl.dmat, ans_dmat)
        npt.assert_array_equal(cl.larr, ans_larr)

    def test_reset_results_on_new_data(self, func_data, dupe_data):
        """
        When new data is given to clusterer,
        recalculate the distance matrix and linkage arr
        """
        cl = Clusterer(func_data)
        npt.assert_array_equal(get_distance_matrix(func_data), cl.dmat)
        npt.assert_array_equal(get_extended_linkage(cl.dmat), cl.larr)

        cl.data = dupe_data
        npt.assert_array_equal(cl.data, dupe_data)
        npt.assert_array_equal(get_distance_matrix(dupe_data), cl.dmat)
        npt.assert_array_equal(get_extended_linkage(cl.dmat), cl.larr)

    def test_create_clusters(self, func_data, dupe_data, outlier_data):
        """
        Tests to confirm:
        1. All keys are present in outer and inner dicts
        2. Max level and max clusters are correct
        3. Distances are all positive
        4. All samples have a cluster
        """
        for dataset in (func_data, dupe_data, outlier_data):
            clusterer = Clusterer(dataset)
            result_clusters, max_levels, max_clusters = clusterer.create_clusters()

            # All inner keys are present, and none are added
            inner_key_set = {
                "cluster_num",
                "level",
                "count",
                "avg_dist",
                "dist_std",
                "samples",
                "sample_dist",
                "cluster_merged",
            }
            for value_dict in result_clusters.values():
                assert set(value_dict) == inner_key_set

            # Max level and max clusters are empirically correct
            n = len(dataset)
            assert max_levels <= n  # Max levels must be less than samples
            assert max_clusters <= n // 2  # Minimum 2 samples for a valid cluster
            assert max_levels >= 1
            assert max_clusters >= 1

            result_max_level = -1
            result_max_cluster = -1

            all_samples = set()

            for v in result_clusters.values():
                res_level = v.get("level")
                res_cluster_num = v.get("cluster_num")
                assert res_level
                assert res_cluster_num

                result_max_level = max(result_max_level, res_level)
                result_max_cluster = max(result_max_cluster, res_cluster_num)

                # All distances must be positive
                assert v.get("avg_dist", -1) >= 0
                assert v.get("dist_std", -1) >= 0

                # pseudo assert: will crash on no samples since list is unhashable
                all_samples.update(set(v.get("samples", [])))
                # TODO -> last level should contain all samples
                # -> can just check that result_clusters[sample_size+len(clusterer.larr)]["count"] == sample_size
                # --> to get the last level you need sample_size+len(clusterer.larr) because the inner dict starts
                # --> at sample_size and adds one for each row of clusterer.larr

            # Max levels and clusters were tracked correctly,
            # but we subtracted one doing the return to account for final merge
            assert max_levels == result_max_level
            assert max_clusters == result_max_cluster

            assert len(all_samples) == n

    def test_reorganize(self, func_data, dupe_data, outlier_data):
        # for dataset in (func_data, dupe_data, outlier_data):
        for dataset in (dupe_data,):
            cl = Clusterer(dataset)
            res, max_levels, max_clusters = cl.create_clusters()
            clusters_per_lvl, merge_groups, outliers, potential_outliers = cl.reorganize_clusters(
                res, min_num_samples_per_cluster=2
            )

            for lvl in range(1, max_levels):
                assert lvl in clusters_per_lvl

            cluster_set = set()

            for clusters in merge_groups.values():
                sck = set(clusters.keys())
                cluster_set.update(sck)

            assert max(cluster_set) <= max_clusters

            """
            TODO: determine size of sample, compare with min num samples per cluster value
            we need to come up with a default minimum based on sample size - 5%??
            Need to hard code min_num_samples_per_cluster >=2 and <= some max?? like 50, 100?

            c1 - 51 -> 49
            c2 - ?
            5% -> 5 samples come outliers
            100 samples
            check 5%
            min_num has to be at at least 2
            """

            # Checks that no indices are shared
            assert set(outliers).isdisjoint(set(potential_outliers))


class TestFunctional:
    def test_run_results(self, blobs):
        data = blobs

        clusterer = Clusterer(data)
        results = clusterer.run()

        assert results["outliers"] == [21, 6, 4, 71, 38, 11]
        assert results["potential_outliers"] == [42, 48, 9, 1, 43]
        assert results["duplicates"] == [(24, 79), (58, 63)]
        assert results["near_duplicates"] == [
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

    def test_duplicate_images(self, dupes):
        cl = Clusterer(dupes)
        results = cl.run()

        assert results is None
