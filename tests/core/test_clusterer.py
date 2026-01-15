import numpy as np
import pytest
import sklearn.datasets as dsets


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
        # import on runtime to minimize load times
        from dataeval.core._clusterer import cluster

        rows, cols = shape
        rand_arr = np.random.random(size=(rows, cols))
        dup_arr = np.ones(shape=(rows, cols))

        test_sets = (rand_arr, dup_arr)

        for test_set in test_sets:
            for alg in ["hdbscan", "kmeans"]:
                c = cluster(test_set, algorithm=alg)  # type: ignore

                # Distance matrix
                assert not np.any(np.isnan(c["k_distances"]))  # Should contain no NaN
                assert (c["k_distances"] >= 0).all()  # Distances are always positive or 0 (same data point)

                # Minimum spanning tree
                assert not np.any(np.isnan(c["mst"]))  # Should contain no NaN

                # Linkage arr
                assert not np.any(np.isnan(c["linkage_tree"]))  # Should contain no NaN


@pytest.mark.required
class TestClustererValidate:
    @pytest.mark.parametrize(
        "data, error, error_msg",
        [
            (
                np.array([[[0]]]),
                ValueError,
                "Data should have at least 2 samples; got 1",
            ),
            (np.array([[1]]), ValueError, "Data should have at least 2 samples; got 1"),  # samples < 2
            (np.array([[], []]), ValueError, "Samples should have at least 1 feature; got 0"),  # features < 1
        ],
    )
    def test_invalid(self, data, error, error_msg):
        # import on runtime to minimize load times
        from dataeval.core._clusterer import cluster

        for alg in ["hdbscan", "kmeans"]:
            with pytest.raises(error) as e:
                cluster(data, algorithm=alg)  # type: ignore
            assert e.value.args[0] == error_msg

    def test_valid(self):
        # import on runtime to minimize load times
        from dataeval.core._clusterer import cluster

        data = np.ones((2, 1, 2, 3, 4))
        for alg in ["hdbscan", "kmeans"]:
            cluster(data.reshape((2, -1)), algorithm=alg)  # type: ignore


def get_blobs(std=0.3) -> np.typing.NDArray[np.float64]:
    blobs, _ = dsets.make_blobs(  # type: ignore
        n_samples=100,
        centers=[(-1.5, 1.8), (-1, 3), (0.8, 2.1), (2.8, 1.5), (2.5, 3.5)],  # type: ignore
        cluster_std=std,
        random_state=33,
    )
    return np.asarray(blobs, dtype=np.float64)


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

    @pytest.mark.parametrize("data_func", ["functional_data", "duplicate_data", "outlier_data"])
    def test_create_clusters(self, data_func, request):
        """
        1. All keys are present in outer and inner dicts
        2. Max level and max clusters are correct
        3. Distances are all positive
        4. All samples have a cluster
        """
        from dataeval.core._clusterer import cluster

        dataset = request.getfixturevalue(data_func)
        for alg in ["hdbscan", "kmeans"]:
            cl = cluster(dataset)

            # clusters are counting numbers >= -1
            assert (cl["clusters"] >= -1).all()
            assert np.issubdtype(cl["clusters"].dtype, np.integer)

    def test_hdbscan_functional(self, functional_data):
        """The results of evaluate are equivalent to the known outputs"""
        from dataeval.core._clusterer import cluster

        cl = cluster(functional_data, algorithm="hdbscan")

        assert cl["mst"] is not None
        assert cl["linkage_tree"] is not None

        # fmt: off
        assert cl["clusters"].tolist() == [
            2, 0, 2, 2, 2, 3, 3, 4, 4, 0, 3, 2, 4, 2, 1, 2, 2, 4, 3, 4, 0, 3,
            2, 1, 2, 1, 3, 4, 0, 4, 2, 1, 4, 4, 1, 3, 2, 3, 1, 4, 1, 1, 3, 1,
            3, 3, 2, 4, 3, 1, 4, 4, 4, 3, 1, 4, 0, 2, 0, 1, 3, 3, 1, 0, 4, 3,
            2, 2, 1, 0, 4, 0, 1, 3, 4, 4, 4, 0, 0, 2, 0, 0, 0, 0, 1, 0, 1, 2,
            1, 1, 3, 0, 4, 0, 4, 2, 1, 0, 0, 2
        ]
        # fmt: on

    def test_kmeans_functional(self, functional_data):
        """The results of evaluate are equivalent to the known outputs"""
        from dataeval.core._clusterer import cluster

        cl = cluster(functional_data, algorithm="kmeans", n_clusters=5)

        assert cl["mst"] is not None
        assert cl["linkage_tree"] is not None

        # fmt: off
        assert cl["clusters"].tolist() == [
            3, 2, 3, 3, 3, 4, 4, 1, 1, 2, 4, 3, 1, 3, 0, 3, 3, 1, 4, 1, 2, 4,
            3, 0, 3, 0, 4, 1, 2, 1, 3, 0, 1, 1, 0, 4, 3, 4, 0, 1, 0, 0, 4, 0,
            4, 4, 3, 1, 4, 0, 1, 1, 1, 4, 0, 1, 2, 3, 2, 0, 4, 4, 0, 2, 1, 4,
            3, 3, 0, 2, 4, 2, 0, 4, 1, 1, 1, 2, 2, 3, 2, 2, 2, 2, 0, 2, 0, 3,
            0, 0, 4, 2, 1, 2, 1, 3, 0, 2, 2, 3
        ]
        # fmt: on
