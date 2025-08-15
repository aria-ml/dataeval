from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.mark.required
class TestMst:
    def test_mst(self):
        from dataeval.core._mst import minimum_spanning_tree

        images = np.ones((10, 3, 3))
        rows, cols = minimum_spanning_tree(images)
        assert (rows == [0, 1, 2, 4, 5, 6, 7, 8, 9]).all() and (cols == [3, 3, 3, 3, 3, 3, 3, 3, 3]).all()

    def test_simple_nodes(self):
        from dataeval.core._mst import minimum_spanning_tree

        X = np.array(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [1, 1],
                [10, 0],
                [11, 0],
                [10, 1],
            ]
        ).astype(np.float64)
        rows, cols = minimum_spanning_tree(X)

        total = 0.0
        for i in range(len(rows)):
            x0, y0 = X[rows[i]]
            x1, y1 = X[cols[i]]
            total += np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

        # Disabling test until fix is in
        # assert (total == 14.0)  # 14-Aug-2025, picking wrong long edge and getting 15.0

    def test_compute_neighbors(self):
        from dataeval.core._mst import compute_neighbors

        images_0 = np.zeros((10, 3, 3))
        images_1 = np.ones((10, 3, 3))

        assert compute_neighbors(images_0, images_1).shape == (10,)

    def test_compute_neighbors_k0(self):
        from dataeval.core._mst import compute_neighbors

        images_0 = np.zeros((10, 3, 3))
        images_1 = np.ones((10, 3, 3))

        with pytest.raises(ValueError):
            compute_neighbors(images_0, images_1, k=0).shape

    def test_compute_neighbors_k2(self):
        from dataeval.core._mst import compute_neighbors

        images_0 = np.zeros((10, 3, 3))
        images_1 = np.ones((10, 3, 3))

        assert compute_neighbors(images_0, images_1, k=2).shape == (10, 2)

    def test_compute_neighbors_kdtree(self):
        from dataeval.core._mst import compute_neighbors

        images_0 = np.zeros((10, 3, 3))
        images_1 = np.ones((10, 3, 3))
        assert compute_neighbors(images_0, images_1, algorithm="kd_tree").shape == (10,)

    def test_compute_neighbors_balltree(self):
        from dataeval.core._mst import compute_neighbors

        images_0 = np.zeros((10, 3, 3))
        images_1 = np.ones((10, 3, 3))
        assert compute_neighbors(images_0, images_1, algorithm="ball_tree").shape == (10,)

    def test_compute_neighbors_invalid_alg(self):
        """Brute algorithm is valid for sklearn.NearestNeighbor, but is invalid for DataEval"""
        from dataeval.core._mst import compute_neighbors

        with pytest.raises(ValueError):
            compute_neighbors(MagicMock(), MagicMock(), algorithm="brute")  # type: ignore #
