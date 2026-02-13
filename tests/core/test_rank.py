from unittest.mock import patch

import numpy as np
import pytest

from dataeval.core._clusterer import (
    _Clusters,
    _ComplexitySorter,
    _DistanceSorter,
    _KNNSorter,
)


class TestClusters:
    labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    cluster_centers = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]])
    clst = _Clusters(labels, cluster_centers)

    def test_clusters_init(self):
        assert self.clst.labels.tolist() == self.labels.tolist()
        assert self.clst.cluster_centers.tolist() == self.cluster_centers.tolist()
        assert self.clst.unique_labels.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_clusters_dist2center(self):
        dist = self.clst._dist2center(self.cluster_centers)
        assert dist.shape == (10,)
        assert all(d == 0 for d in dist)

    def test_clusters_complexity(self):
        comp = self.clst._complexity(self.cluster_centers)
        assert comp.shape == (10,)
        assert all(c == 0.1 for c in comp)

    def test_clusters_sort_by_weights(self):
        inds = self.clst._sort_by_weights(self.cluster_centers)
        assert inds.shape == (10,)


class TestSorters:
    embeddings = np.random.random((10, 10))
    reference = np.random.random((10, 10))

    def test_knn_sorter(self):
        sorter = _KNNSorter(k=2, samples=len(self.embeddings))
        inds = sorter._sort(self.embeddings)
        assert inds.shape == (10,)

    def test_knn_sorter_with_reference(self):
        sorter = _KNNSorter(k=2, samples=len(self.embeddings))
        inds = sorter._sort(self.embeddings, self.reference)
        assert inds.shape == (10,)

    def test_knn_sorter_with_k_zero(self):
        sorter = _KNNSorter(k=0, samples=len(self.embeddings))
        assert sorter._k == int(np.sqrt(len(self.embeddings)))

    def test_knn_sorter_with_k_greater_than_samples_raises_valueerror(self):
        with pytest.raises(ValueError, match="should be less than dataset size"):
            _KNNSorter(k=10, samples=10)

    def test_knn_sorter_with_k_greater_than_recommended_warns(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            _KNNSorter(k=9, samples=10)
        assert len(caplog.records) >= 1

    def test_kmeans_distance_sorter(self):
        sorter = _DistanceSorter(samples=len(self.embeddings), algorithm="kmeans", c=2)
        inds = sorter._sort(self.embeddings)
        assert inds.shape == (10,)

    def test_kmeans_distance_sorter_with_reference(self):
        sorter = _DistanceSorter(samples=len(self.embeddings), algorithm="kmeans", c=2)
        inds = sorter._sort(self.embeddings, self.reference)
        assert inds.shape == (10,)

    def test_kmeans_complexity_sorter(self):
        sorter = _ComplexitySorter(samples=len(self.embeddings), algorithm="kmeans", c=2)
        inds = sorter._sort(self.embeddings)
        assert inds.shape == (10,)

    def test_complexity_sorter_with_reference(self):
        for alg in ["hdbscan", "kmeans"]:
            sorter = _ComplexitySorter(samples=len(self.embeddings), algorithm=alg, c=2)  # type: ignore
            inds = sorter._sort(self.embeddings, self.reference)
            assert inds.shape == (10,)

    def test_kmeans_sorter_with_c_zero(self):
        sorter = _ComplexitySorter(samples=len(self.embeddings), algorithm="kmeans", c=0)
        assert sorter._clusterer._c == int(np.sqrt(len(self.embeddings)))  # type: ignore

    def test_kmeans_sorter_with_c_greater_than_samples_raises_valueerror(self):
        with pytest.raises(ValueError, match="should be less than dataset size"):
            _ComplexitySorter(samples=10, algorithm="kmeans", c=10)

    @patch("dataeval.core._clusterer.KMeans")
    def test_kmeans_sorter_kmeans_returns_none(self, mock_kmeans_cls):
        mock_kmeans = mock_kmeans_cls.return_value
        mock_kmeans.labels_ = None
        mock_kmeans.cluster_centers_ = None
        sorter = _DistanceSorter(samples=len(self.embeddings), algorithm="kmeans", c=2)
        with pytest.raises(ValueError, match="Clustering failed to produce"):
            sorter._sort(self.embeddings)

    @patch("dataeval.core._clusterer._HDBSCAN")
    def test_hdbscan_sorter_hdbscan_returns_none(self, mock_cls):
        mock_hdbscan = mock_cls.return_value
        mock_hdbscan.labels_ = None
        mock_hdbscan.cluster_centers_ = None
        sorter = _DistanceSorter(samples=len(self.embeddings), algorithm="hdbscan", c=2)
        with pytest.raises(ValueError, match="Clustering failed to produce"):
            sorter._sort(self.embeddings)
