from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.mark.required
class TestMst:
    def test_mst(self):
        from dataeval.core._mst import minimum_spanning_tree

        images = np.ones((10, 3, 3))
        mst = minimum_spanning_tree(images)
        assert (mst["source"] == [7, 7, 7, 6, 6, 6, 6, 6, 7]).all()
        assert (mst["target"] == [6, 3, 2, 0, 5, 9, 4, 8, 1]).all()

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
                [15, 0],
            ],
        ).astype(np.float64)
        mst = minimum_spanning_tree(X)

        total = 0.0
        for i in range(len(mst["source"])):
            x0, y0 = X[mst["source"][i]]
            x1, y1 = X[mst["target"][i]]
            total += np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

        assert total == 18.0

    def test_compute_neighbors(self):
        from dataeval.core._mst import compute_neighbors

        images_0 = np.zeros((10, 3, 3))
        images_1 = np.ones((10, 3, 3))

        assert compute_neighbors(images_0, images_1).shape == (10,)

    def test_compute_neighbors_k0(self):
        from dataeval.core._mst import compute_neighbors

        images_0 = np.zeros((10, 3, 3))
        images_1 = np.ones((10, 3, 3))

        with pytest.raises(ValueError, match="k must be >= 1"):
            compute_neighbors(images_0, images_1, k=0)

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
        """Brute algorithm is valid for sklearn.NearestNeighbor, but is invalid for DataEval."""
        from dataeval.core._mst import compute_neighbors

        with pytest.raises(ValueError, match="Algorithm must be"):
            compute_neighbors(MagicMock(), MagicMock(), algorithm="brute")  # type: ignore #

    def test_compare_links_skips_noise(self):
        """Noise points (cluster id -1) must not be flagged as duplicates.

        Regression: prior implementation initialized cluster_grouping with -1,
        colliding with the HDBSCAN noise label. Noise-noise edges were grouped
        with inter-cluster edges and flagged when their distance was below the
        sensitivity-scaled std of that mixed bag.
        """
        from dataeval.core._fast_hdbscan._mst import compare_links_to_cluster_std

        # MST over 7 points: 5 in cluster 0 with one tight pair so std > 0
        # and one short edge below cluster_sensitivity * std, plus 2 noise
        # points close to each other but far from the cluster.
        mst = np.array(
            [
                [0, 1, 0.10],  # intra-cluster 0 (tight — should be flagged)
                [1, 2, 1.00],  # intra-cluster 0
                [2, 3, 1.00],  # intra-cluster 0
                [3, 4, 1.00],  # intra-cluster 0
                [4, 5, 5.00],  # cluster 0 -> noise (inter)
                [5, 6, 0.05],  # noise <-> noise (must NOT be flagged)
            ],
            dtype=np.float32,
        )
        clusters = np.array([0, 0, 0, 0, 0, -1, -1], dtype=np.int64)

        pairs = compare_links_to_cluster_std(mst, clusters, cluster_sensitivity=2.0)

        # No flagged pair may involve a noise point.
        assert pairs.size > 0, "expected intra-cluster duplicates to still be flagged"
        for p, q in pairs:
            assert clusters[p] != -1
            assert clusters[q] != -1, f"noise pair flagged: ({p}, {q})"

    def test_knn_exhaustion_warning(self, caplog):
        """Test that algorithm warns when k-nearest neighbor graph is exhausted."""
        import logging

        from dataeval.core._mst import minimum_spanning_tree

        # Create two distant clusters: insufficient k will trigger warning
        cluster1 = np.random.RandomState(42).uniform(-1, 1, (10, 3))  # 10 points near origin
        cluster2 = np.random.RandomState(123).uniform(-1, 1, (10, 3)) + [50, 0, 0]  # 10 points far away
        X = np.vstack([cluster1, cluster2]).astype(np.float64)

        # Test with k=5 (insufficient: each cluster has 9 other points, so inter-cluster edges at rank 10+)
        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            mst = minimum_spanning_tree(X, k=5)

        # Should trigger KNN exhaustion warning
        knn_warnings = [rec for rec in caplog.records if "k-nearest neighbors" in rec.message.lower()]
        assert len(knn_warnings) > 0, "Expected KNN exhaustion warning with insufficient k"

        # Should still produce spanning tree (19 edges for 20 points)
        assert len(mst["source"]) == 19

        # Test with k=15 (sufficient: should NOT warn)
        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            mst = minimum_spanning_tree(X, k=15)

        knn_warnings = [rec for rec in caplog.records if "k-nearest neighbors" in rec.message.lower()]
        assert len(knn_warnings) == 0, "Unexpected warning with sufficient k"
