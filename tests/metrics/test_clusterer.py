import numpy as np
import pytest
import sklearn.datasets as dsets

from dataeval.metrics.estimators import ClustererOutput, clusterer


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
class TestClusterer:
    """Tests all functions related to and including the `create_clusters` method"""

    clusterer_output = clusterer(np.zeros((3, 1)))

    @pytest.mark.parametrize("data_func", ["functional_data", "duplicate_data", "outlier_data"])
    def test_create_clusters(self, data_func, request):
        """
        1. All keys are present in outer and inner dicts
        2. Max level and max clusters are correct
        3. Distances are all positive
        4. All samples have a cluster
        """
        dataset = request.getfixturevalue(data_func)
        cl = clusterer(dataset)

        # clusters are counting numbers >= -1
        assert (cl.clusters >= -1).all()
        assert np.issubdtype(cl.clusters.dtype, np.integer)

    def test_functional(self, functional_data):
        """The results of evaluate are equivalent to the known outputs"""
        cl = clusterer(functional_data)

        outliers = cl.find_outliers()
        assert len(outliers) == 0

        duplicates, potential_duplicates = cl.find_duplicates()
        assert duplicates == [[24, 79], [58, 63]]
        assert potential_duplicates == [
            [0, 13, 15, 22, 30, 57, 67, 87, 95],
            [3, 79],
            [8, 27, 29],
            [10, 65],
            [16, 99],
            [19, 64],
            [31, 86],
            [33, 76],
            [36, 66],
            [39, 55],
            [40, 72, 96],
            [41, 62],
            [58, 83],
            [78, 91],
            [80, 81, 93],
            [82, 97],
        ]

        assert cl.mst is not None
        assert cl.linkage_tree is not None
        assert cl.condensed_tree is not None

        # fmt: off
        assert cl.clusters.tolist() == [
            1,0,1,1,1,3,3,4,4,0,3,1,4,1,2,1,1,4,3,4,0,3,1,2,1,
            2,3,4,0,4,1,2,4,4,2,3,1,3,2,4,2,2,3,2,3,3,1,4,3,2,
            4,4,4,3,2,4,0,1,0,2,3,3,2,0,4,3,1,1,2,0,4,0,2,3,4,
            4,4,0,0,1,0,0,0,0,2,0,2,1,2,2,3,0,4,0,4,1,2,0,0,1,
        ]
        # fmt: on


@pytest.mark.required
class TestClusterOutliers:
    """Tests all functions related to and including the `find_outliers` method"""

    @pytest.mark.parametrize(
        "clusters, outs",
        [
            (np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, -1]), [9]),
            (np.array([-1, -1, -1]), [0, 1, 2]),
            (np.array([3, 1, 1, 0, -1, 2, 3, 2, -1, 0]), [4, 8]),
            (np.array([0, 2, 4, 1, 0, 4, 3, 3, 2, 2, 1]), []),
        ],
    )
    def test_find_outliers(self, clusters, outs):
        """Specified outliers are added to lists"""

        cl = ClustererOutput(clusters, np.array([]), np.array([]), np.array([]), np.array([]))

        outliers = cl.find_outliers()

        assert outliers.tolist() == outs

    @pytest.mark.parametrize(
        "indices",
        [
            list(range(4)),
            list(range(4)) * -1,
            [12, 44, 78, 91],
        ],
    )
    def test_outliers(self, indices):
        """Integration test: `Clusterer` finds outlier data"""
        data = get_blobs(0.1)
        data[indices] *= 10.0
        cl = clusterer(np.array(data))
        outliers = cl.find_outliers()

        # Only need to check specified outliers in results, but there might be other outliers
        assert all(x in outliers for x in indices)


@pytest.mark.required
class TestClusterDuplicates:
    """Tests all functions related to and including the `find_duplicates` method"""

    def test_duplicates(self, duplicate_data):
        """`Clusterer` finds duplicate data during evaluate"""
        cl = clusterer(duplicate_data)
        duplicates, potential_duplicates = cl.find_duplicates()

        # Only 1 set (all dupes) in list of sets
        assert len(duplicates[0]) == len(duplicate_data)
        assert potential_duplicates == []

    def test_no_duplicates(self):
        """`Clusterer` finds no :term:`duplicates<Duplicates>` during evaluate"""
        data = np.array([[0, 0], [1, 1], [2, 2]])
        cl = clusterer(data)
        duplicates, potential_duplicates = cl.find_duplicates()

        assert not len(duplicates)
        assert not len(potential_duplicates)
