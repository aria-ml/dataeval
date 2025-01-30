from unittest.mock import MagicMock

import numpy as np
import pytest

from dataeval.utils._shared import compute_neighbors, get_classes_counts, minimum_spanning_tree


@pytest.mark.required
class TestSharedUtils:
    def test_class_min(self):
        with pytest.raises(ValueError):
            get_classes_counts(np.ones(1, dtype=np.int_))

    def test_mst(self):
        images = np.ones((10, 3, 3))
        assert minimum_spanning_tree(images).shape == (10, 10)

    def test_compute_neighbors(self):
        images_0 = np.zeros((10, 3, 3))
        images_1 = np.ones((10, 3, 3))

        assert compute_neighbors(images_0, images_1).shape == (10,)

    def test_compute_neighbors_k0(self):
        images_0 = np.zeros((10, 3, 3))
        images_1 = np.ones((10, 3, 3))

        with pytest.raises(ValueError):
            compute_neighbors(images_0, images_1, k=0).shape

    def test_compute_neighbors_k2(self):
        images_0 = np.zeros((10, 3, 3))
        images_1 = np.ones((10, 3, 3))

        assert compute_neighbors(images_0, images_1, k=2).shape == (10, 2)

    def test_compute_neighbors_kdtree(self):
        images_0 = np.zeros((10, 3, 3))
        images_1 = np.ones((10, 3, 3))
        assert compute_neighbors(images_0, images_1, algorithm="kd_tree").shape == (10,)

    def test_compute_neighbors_balltree(self):
        images_0 = np.zeros((10, 3, 3))
        images_1 = np.ones((10, 3, 3))
        assert compute_neighbors(images_0, images_1, algorithm="ball_tree").shape == (10,)

    def test_compute_neighbors_invalid_alg(self):
        """Brute algorithm is valid for sklearn.NearestNeighbor, but is invalid for DataEval"""

        with pytest.raises(ValueError):
            compute_neighbors(MagicMock(), MagicMock(), algorithm="brute")  # type: ignore #
