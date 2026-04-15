"""Verify that core functional components produce correct output types.

Maps to meta repo test cases:
  - TC-10.2: Core functional interface (Hashing, Clustering, Mutual Information)
"""

import numpy as np
import pytest


@pytest.mark.test_case("10-2")
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
        result = mutual_info(label, factor)
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
