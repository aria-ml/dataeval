import numpy as np
import pytest

from dataeval.utils._clusterer import cluster


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
            c = cluster(test_set)

            # Distance matrix
            assert not np.any(np.isnan(c.k_distances))  # Should contain no NaN
            assert (c.k_distances >= 0).all()  # Distances are always positive or 0 (same data point)

            # Minimum spanning tree
            assert not np.any(np.isnan(c.mst))  # Should contain no NaN
            print(c.mst)

            # Linkage arr
            assert not np.any(np.isnan(c.linkage_tree))  # Should contain no NaN
            print(c.linkage_tree)


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
        with pytest.raises(error) as e:
            cluster(data)
        assert e.value.args[0] == error_msg

    def test_valid(self):
        data = np.ones((2, 1, 2, 3, 4))
        cluster(data.reshape((2, -1)))
