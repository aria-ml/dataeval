from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from dataeval.utils.data._embeddings import Embeddings
from dataeval.utils.data._selection import Select
from dataeval.utils.data.selections._prioritize import (
    Prioritize,
    _Clusters,
    _KMeansComplexitySorter,
    _KMeansDistanceSorter,
    _KNNSorter,
)


class TestPrioritizeClusters:
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


class TestPrioritizeSorters:
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
        with pytest.raises(ValueError):
            _KNNSorter(k=10, samples=10)

    def test_knn_sorter_with_k_greater_than_recommended_warns(self):
        with pytest.warns(UserWarning):
            _KNNSorter(k=9, samples=10)

    def test_kmeans_distance_sorter(self):
        sorter = _KMeansDistanceSorter(c=2, samples=len(self.embeddings))
        inds = sorter._sort(self.embeddings)
        assert inds.shape == (10,)

    def test_kmeans_distance_sorter_with_reference(self):
        sorter = _KMeansDistanceSorter(c=2, samples=len(self.embeddings))
        inds = sorter._sort(self.embeddings, self.reference)
        assert inds.shape == (10,)

    def test_kmeans_complexity_sorter(self):
        sorter = _KMeansComplexitySorter(c=2, samples=len(self.embeddings))
        inds = sorter._sort(self.embeddings)
        assert inds.shape == (10,)

    def test_kmeans_complexity_sorter_with_reference(self):
        sorter = _KMeansComplexitySorter(c=2, samples=len(self.embeddings))
        inds = sorter._sort(self.embeddings, self.reference)
        assert inds.shape == (10,)

    def test_kmeans_sorter_with_c_zero(self):
        sorter = _KMeansComplexitySorter(c=0, samples=len(self.embeddings))
        assert sorter._c == int(np.sqrt(len(self.embeddings)))

    def test_kmeans_sorter_with_c_greater_than_samples_raises_valueerror(self):
        with pytest.raises(ValueError):
            _KMeansComplexitySorter(c=10, samples=10)


class TestPrioritizeSelection:
    model = torch.nn.Flatten()
    batch_size = 10
    device = torch.device("cpu")

    def get_dataset(self, n: int = 1000) -> Select:
        mock = MagicMock(spec=Select)
        mock._selection = list(range(n))
        mock.__len__.return_value = n
        mock.__getitem__.return_value = np.random.random((1, 10, 10)), np.zeros(10), {}
        return mock

    def get_embeddings(self, n: int = 1000) -> Embeddings:
        return Embeddings(self.get_dataset(n), batch_size=self.batch_size, model=self.model, device=self.device)

    @pytest.mark.parametrize(
        "method, method_kwargs",
        (
            ("knn", {"k": 10}),
            ("kmeans_distance", {"c": 100}),
            ("kmeans_complexity", {"c": 100}),
        ),
    )
    def test_prioritize_init(self, method, method_kwargs):
        p = Prioritize(self.model, self.batch_size, self.device, method, **method_kwargs)
        assert p._method == method
        assert (p._c == method_kwargs["c"]) if "c" in method_kwargs else (p._c is None)
        assert (p._k == method_kwargs["k"]) if "k" in method_kwargs else (p._k is None)
        assert p._get_sorter(1000) is not None
        dataset = self.get_dataset()
        p(dataset)
        assert any(i != j for i, j in zip(dataset._selection, range(1000)))

    def test_prioritize_invalid_method_raises_valueerror(self):
        with pytest.raises(ValueError):
            Prioritize(self.model, self.batch_size, self.device, "invalid")  # type: ignore

    @pytest.mark.parametrize(
        "emb_ref",
        (
            (False, True),
            (True, False),
            (True, True),
        ),
    )
    @pytest.mark.parametrize(
        "method, method_kwargs",
        (
            ("knn", {"k": 10}),
            ("kmeans_distance", {"c": 100}),
            ("kmeans_complexity", {"c": 100}),
        ),
    )
    def test_prioritize_using(self, method, method_kwargs, emb_ref):
        emb_kwargs = {key: self.get_embeddings() for i, key in enumerate(["embeddings", "reference"]) if emb_ref[i]}
        p = Prioritize.using(method, **method_kwargs, **emb_kwargs)  # type: ignore
        assert p._method == method
        assert (p._c == method_kwargs["c"]) if "c" in method_kwargs else (p._c is None)
        assert (p._k == method_kwargs["k"]) if "k" in method_kwargs else (p._k is None)

        assert (p._embeddings is not None) == emb_ref[0]
        assert (p._reference is not None) == emb_ref[1]
        assert p._get_sorter(1000) is not None
        dataset = self.get_dataset()
        p(dataset)
        assert any(i != j for i, j in zip(dataset._selection, range(1000)))

    def test_prioritize_using_no_embeddings_raises_valueerror(self):
        with pytest.raises(ValueError):
            Prioritize.using("knn")

    def test_prioritize_using_different_embeddings_raises_valueerror(self):
        emb = self.get_embeddings(100)
        p = Prioritize.using("knn", embeddings=emb)
        dataset = self.get_dataset()
        with pytest.raises(ValueError):
            p(dataset)
