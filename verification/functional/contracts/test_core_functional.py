"""Verify that core functional components produce correct output types.

Maps to meta repo test cases:
  - TC-10.2: Core functional interface (Hashing, Clustering, Mutual Information)
"""

import numpy as np
import pytest


class TestCoreFunctional:
    """Verify core functional components."""

    def test_xxhash_produces_consistent_hashes(self):
        from dataeval.core import xxhash

        data = np.zeros((10, 3, 16, 16), dtype=np.uint8)
        # xxhash in core handles a single image
        hashes = [xxhash(img) for img in data]
        assert len(hashes) == 10
        assert isinstance(hashes[0], str)
        assert len(hashes[0]) == 16
        assert hashes[0] == xxhash(data[0])

    def test_cluster_performs_clustering(self):
        from dataeval.core import cluster

        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 8)).astype(np.float32)

        # Test HDBSCAN (default)
        result = cluster(data, algorithm="hdbscan")
        assert isinstance(result, dict)
        assert "clusters" in result
        assert "mst" in result

        # Test KMeans
        result = cluster(data, algorithm="kmeans", n_clusters=3)
        assert len(np.unique(result["clusters"])) <= 3

    def test_mutual_info_calculates_scores(self):
        from dataeval.core import mutual_info

        rng = np.random.default_rng(42)
        # mutual_info expects factor_data to be (N, k) 2D array
        factor = rng.integers(0, 2, (100, 1)).astype(np.float32)
        # mutual_info expects class_labels to be 1D array (N,)
        label = factor.flatten().astype(np.intp)

        # Fix: ensure label is 1D
        result = mutual_info(label, factor, discrete_features=[True])
        assert isinstance(result, dict)
        assert result["class_to_factor"][1] > 0.9

    def test_label_stats_computes_distribution(self):
        from dataeval.core import label_stats

        labels = np.array([0, 0, 1, 1, 1], dtype=np.intp)
        result = label_stats(labels)
        assert isinstance(result, dict)
        # result keys are label_counts_per_class, etc.
        assert result["label_counts_per_class"][0] == 2
        assert result["label_counts_per_class"][1] == 3

    def test_divergence_separates_distributions(self):
        from dataeval.core import divergence_fnn, divergence_mst

        rng = np.random.default_rng(0)
        base = rng.standard_normal((100, 4)).astype(np.float32)
        same = rng.standard_normal((100, 4)).astype(np.float32)
        shifted = rng.standard_normal((100, 4)).astype(np.float32) + 3

        for divergence in (divergence_mst, divergence_fnn):
            far = divergence(base, shifted)
            near = divergence(base, same)
            assert set(far) == {"divergence", "errors"}
            assert far["divergence"] > 0.9  # separable distributions
            assert near["divergence"] < far["divergence"]

    def test_mst_and_neighbors(self):
        from dataeval.core import compute_neighbors, minimum_spanning_tree

        data = np.random.default_rng(0).standard_normal((50, 4)).astype(np.float32)
        mst = minimum_spanning_tree(data, k=5)
        assert len(mst["source"]) == len(mst["target"]) == 49  # a spanning tree has n - 1 edges

        neighbors = compute_neighbors(data, data[:3], k=4)
        assert neighbors.shape == (3, 4)
        assert neighbors[:, 0].tolist() == [0, 1, 2]  # each query's nearest neighbor is itself

    def test_label_errors_flag_flipped_labels(self):
        from dataeval.core import label_errors

        rng = np.random.default_rng(0)
        embeddings = np.concatenate([rng.standard_normal((100, 4)), rng.standard_normal((100, 4)) + 6]).astype(
            np.float32
        )
        labels = np.array([0] * 100 + [1] * 100)
        noisy = labels.copy()
        noisy[:5] = 1  # five class-0 samples carry the wrong label

        result = label_errors(embeddings, noisy, k=10)
        assert set(result) == {"errors", "error_rank", "scores"}
        assert len(result["scores"]) == 200
        assert set(result["error_rank"][:5].tolist()) == {0, 1, 2, 3, 4}  # the flipped samples rank first

    def test_label_parity_and_factor_parity(self):
        from dataeval.core import label_parity, parity

        same = label_parity(np.array([0, 0, 1, 1, 2, 2] * 10), np.array([0, 0, 1, 1, 2, 2] * 10))
        assert same["p_value"] == pytest.approx(1.0)
        flipped = label_parity(np.array([0] * 50 + [1] * 10), np.array([1] * 50 + [0] * 10))
        assert flipped["p_value"] < 0.01
        assert flipped["chi_squared"] > same["chi_squared"]

        rng = np.random.default_rng(0)
        result = parity(rng.integers(0, 3, (200, 2)), rng.integers(0, 2, 200))
        assert len(result["scores"]) == len(result["p_values"]) == 2

    def test_rank_functions_return_full_ranking(self):
        from dataeval.core import rank_hdbscan_distance, rank_kmeans_distance, rank_knn

        rng = np.random.default_rng(0)
        data = np.concatenate([rng.standard_normal((60, 4)), rng.standard_normal((30, 4)) + 8]).astype(np.float32)
        data[0] = 40  # outlier

        for rank in (rank_knn, rank_kmeans_distance, rank_hdbscan_distance):
            result = rank(data)
            assert sorted(result["indices"].tolist()) == list(range(90))
            assert result["scores"] is not None
            assert len(result["scores"]) == 90
        assert rank_knn(data, k=5)["indices"][-1] == 0  # the outlier is the least typical sample

    def test_nullmodel_metrics(self):
        from dataeval.core import nullmodel_metrics

        result = nullmodel_metrics(np.array([0, 0, 1, 2, 2, 2]), np.array([0, 0, 0, 1, 2, 2]))
        assert set(result) == {"uniform_random", "dominant_class", "proportional_random"}
        uniform: dict = result["uniform_random"]  # pyright: ignore[reportAssignmentType]
        assert uniform["multiclass_accuracy"] == pytest.approx(1 / 3)
        assert all(0.0 <= metrics["precision_micro"] <= 1.0 for metrics in result.values())  # pyright: ignore[reportIndexIssue]
