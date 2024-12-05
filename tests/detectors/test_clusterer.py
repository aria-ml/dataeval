import numpy as np
import numpy.testing as npt
import pytest
import sklearn.datasets as dsets

from dataeval.detectors.linters.clusterer import Clusterer


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
            assert not np.any(np.isnan(c._kdistances))  # Should contain no NaN
            assert (c._kdistances >= 0).all()  # Distances are always positive or 0 (same data point)

            # Minimum spanning tree
            assert not np.any(np.isnan(c._mst))  # Should contain no NaN
            print(c._mst)

            # Linkage arr
            assert not np.any(np.isnan(c._linkage_tree))  # Should contain no NaN
            print(c._linkage_tree)


@pytest.mark.required
class TestClustererValidate:
    @pytest.mark.parametrize(
        "data, error, error_msg",
        [
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
        assert cl._kdistances.shape == (100, 25)
        assert cl._mst.shape == (99, 3)
        assert cl._linkage_tree.shape == (99, 4)
        assert cl._condensed_tree is not None

    @pytest.mark.parametrize(
        "shape",
        (
            (0, 0),  # Empty array
            (0, 1),  # No samples
            (1, 0),  # No features
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
        clusters = clusterer.create_clusters()

        # clusters are counting numbers >= -1
        assert (clusters >= -1).all()
        assert np.issubdtype(clusters.dtype, np.integer)


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

        c = Clusterer(np.zeros((3, 1)))
        o = c.find_outliers(clusters=clusters)

        assert o.tolist() == outs

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
        c = Clusterer(np.array(data))
        results = c.evaluate()
        outliers = results.outliers.tolist()

        # Only need to check specified outliers in results, but there might be other outliers
        assert all(x in outliers for x in indices)


@pytest.mark.required
class TestClusterDuplicates:
    """Tests all functions related to and including the `find_duplicates` method"""

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


class TestClustererEvaluate:
    """Tests the evaluate function with known dataset and results"""

    @pytest.mark.parametrize("return_trees", [False, True])
    def test_evaluate_functional(self, functional_data, return_trees):
        """The results of evaluate are equivalent to the known outputs"""
        clusterer = Clusterer(functional_data)
        results = clusterer.evaluate(return_trees=return_trees)

        assert len(results.outliers) == 0
        assert results.duplicates == [[24, 79], [58, 63]]
        assert results.potential_duplicates == [
            [0, 13, 67],
            [10, 65, 73],
            [16, 99],
            [22, 87, 95],
            [26, 53],
            [28, 78, 80, 81, 91, 93],
            [30, 57],
            [40, 72],
            [41, 62],
            [58, 83],
            [63, 82, 97],
            [69, 85],
        ]

        if return_trees:
            assert results.mst is not None
            assert results.linkage_tree is not None
            assert results.condensed_tree is not None

        # fmt: off
        assert results.clusters.tolist() == [
            1,0,1,1,1,3,3,4,4,0,3,1,4,1,2,1,1,4,3,4,0,3,1,2,1,
            2,3,4,0,4,1,2,4,4,2,3,1,3,2,4,2,2,3,2,3,3,1,4,3,2,
            4,4,4,3,2,4,0,1,0,2,3,3,2,0,4,3,1,1,2,0,4,0,2,3,4,
            4,4,0,0,1,0,0,0,0,2,0,2,1,2,2,3,0,4,0,4,1,2,0,0,1,
        ]
        # fmt: on
