from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from dataeval.data._embeddings import Embeddings
from dataeval.data._selection import Select
from dataeval.data.selections._prioritize import (
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

    @patch("dataeval.data.selections._prioritize.KMeans")
    def test_kmeans_sorter_kmeans_returns_none(self, mock_kmeans_cls):
        mock_kmeans = mock_kmeans_cls.return_value
        mock_kmeans.labels_ = None
        mock_kmeans.cluster_centers_ = None
        sorter = _KMeansDistanceSorter(c=2, samples=len(self.embeddings))
        with pytest.raises(ValueError):
            sorter._sort(self.embeddings)


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
        "method, policy, method_kwargs",
        (
            ("knn", "hard_first", {"k": 10}),
            ("kmeans_distance", "hard_first", {"c": 100}),
            ("kmeans_complexity", "hard_first", {"c": 100}),
        ),
    )
    def test_prioritize_init(self, method, policy, method_kwargs):
        p = Prioritize(self.model, self.batch_size, self.device, method, policy, **method_kwargs)
        assert p._method == method
        assert (p._c == method_kwargs["c"]) if "c" in method_kwargs else (p._c is None)
        assert (p._k == method_kwargs["k"]) if "k" in method_kwargs else (p._k is None)
        assert p._get_sorter(1000) is not None
        dataset = self.get_dataset()
        p(dataset)
        assert any(i != j for i, j in zip(dataset._selection, range(1000)))

    @pytest.mark.parametrize(
        "method, policy, policy_kwargs",
        (
            ("knn", "hard_first", {"k": 10}),
            ("knn", "easy_first", {"k": 10}),
            ("knn", "stratified", {"k": 10}),
            ("knn", "class_balance", {"k": 10, "class_label": np.random.randint(low=0, high=10, size=1000)}),
        ),
    )
    def test_prioritize_init_with_correct_policy_params(self, method, policy, policy_kwargs):
        p = Prioritize(self.model, self.batch_size, self.device, method, policy, **policy_kwargs)
        assert p._policy == policy
        assert p._method == method
        assert (p._c == policy_kwargs["c"]) if "c" in policy_kwargs else (p._c is None)
        assert (p._k == policy_kwargs["k"]) if "k" in policy_kwargs else (p._k is None)
        assert p._get_sorter(1000) is not None
        dataset = self.get_dataset()
        p(dataset)

        assert any(i != j for i, j in zip(dataset._selection, range(1000)))

    def test_prioritize_invalid_method_raises_valueerror(self):
        with pytest.raises(ValueError):
            Prioritize(self.model, self.batch_size, self.device, "invalid", "hard_first")  # type: ignore

    def test_prioritize_invalid_policy_raises_valueerror(self):
        with pytest.raises(ValueError):
            Prioritize(self.model, self.batch_size, self.device, "knn", "invalid")  # type: ignore

    def test_prioritize_class_balance_without_labels_raises_valueerror(self):
        with pytest.raises(ValueError):
            emb = self.get_dataset(n=100)
            p = Prioritize.using("knn", "class_balance", k=5)
            p(emb)

    def test_prioritize_stratified_no_scores_raises_valueerror(self):
        with pytest.raises(ValueError):
            emb = self.get_dataset(n=100)
            p = Prioritize.using("kmeans_complexity", "stratified", c=5)
            p(emb)

    def test_class_balance_rerank(self):
        classes = np.array([0, 0, 1, 1, 1, 1, 1], dtype=np.int32)
        p = Prioritize.using("knn", "class_balance", k=5, class_label=classes, embeddings=self.get_embeddings(7))
        # initial ordering -- easy to hard
        inds_0 = np.arange(len(classes), dtype=np.int32)
        # _rerank defaults to hard first with alternating classes
        inds = p._rerank(inds_0)  # indices that rerank inds_0
        # classes should alternate until the minority class is exhausted
        # we're testing indices rather than classes here
        assert all(inds_0[inds] == np.array([6, 1, 5, 0, 4, 3, 2]))

    def test_stratified_rerank(self):
        # strongly clustered and separated scores
        scores = np.array([0.10, 0.11, 0.81, 0.82, 0.83, 0.84, 0.85], dtype=np.float64)
        p = Prioritize.using("knn", "stratified", k=5, embeddings=self.get_embeddings(7))
        inds_0 = np.arange(len(scores), dtype=np.int32)
        # reversing to hard first occurs in _rerank(); this function assumes
        #   decreasing priority
        inds = p._stratified_rerank(scores, inds_0, num_bins=2)
        # low score samples are in the first bin, and scores should alternate
        assert np.all(scores[inds] == np.array([0.1, 0.81, 0.11, 0.82, 0.83, 0.84, 0.85]))

    @pytest.mark.parametrize(
        "emb_ref",
        (
            (False, True),
            (True, False),
            (True, True),
        ),
    )
    @pytest.mark.parametrize(
        "method, policy, method_kwargs",
        (
            ("knn", "hard_first", {"k": 10}),
            ("kmeans_distance", "hard_first", {"c": 100}),
            ("kmeans_complexity", "hard_first", {"c": 100}),
        ),
    )
    def test_prioritize_using(self, method, policy, method_kwargs, emb_ref):
        emb_kwargs = {key: self.get_embeddings() for i, key in enumerate(["embeddings", "reference"]) if emb_ref[i]}
        p = Prioritize.using(method, policy, **method_kwargs, **emb_kwargs)  # type: ignore
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
            Prioritize.using("knn", "hard_first")

    def test_prioritize_using_different_embeddings_raises_valueerror(self):
        emb = self.get_embeddings(100)
        p = Prioritize.using("knn", "easy_first", embeddings=emb)
        dataset = self.get_dataset()
        with pytest.raises(ValueError):
            p(dataset)
