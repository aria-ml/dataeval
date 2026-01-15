from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from dataeval._embeddings import Embeddings
from dataeval.protocols import AnnotatedDataset
from dataeval.quality import Prioritize, PrioritizeOutput


class TestPrioritize:
    model = torch.nn.Flatten()
    batch_size = 10
    device = torch.device("cpu")

    def get_dataset(self, n: int = 1000):
        mock = MagicMock(spec=AnnotatedDataset)
        mock.__len__.return_value = n
        # Use side_effect to return different random data for each call (not the same data every time)
        mock.__getitem__.side_effect = lambda _: (np.random.random((1, 10, 10)), np.zeros(10), {})
        mock.metadata = {"id": "mock_dataset", "index2label": {i: str(i) for i in range(10)}}
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
    def test_prioritize_evaluate(self, method, method_kwargs):
        p = Prioritize(self.model, batch_size=self.batch_size, device=self.device)
        dataset = self.get_dataset()

        # Configure using builder pattern
        if method == "knn":
            p.with_knn(**method_kwargs)
        elif method == "kmeans_distance":
            p.with_kmeans_distance(**method_kwargs)
        elif method == "kmeans_complexity":
            p.with_kmeans_complexity(**method_kwargs)

        result = p.hard_first().evaluate(dataset)

        # Check result type and attributes
        assert isinstance(result, PrioritizeOutput)
        assert len(result.indices) == 1000
        assert result.method == method
        assert result.policy == "hard_first"
        # Check that indices are actually different from default order
        assert any(i != j for i, j in zip(result.indices, range(1000)))

    @pytest.mark.parametrize(
        "policy",
        ("hard_first", "easy_first", "stratified", "class_balance"),
    )
    def test_prioritize_policies(self, policy):
        p = Prioritize(self.model, batch_size=self.batch_size, device=self.device)
        dataset = self.get_dataset()
        labels = np.random.randint(low=0, high=10, size=1000)

        # Configure method
        p.with_knn(k=10)

        # Configure policy
        if policy == "hard_first":
            result = p.hard_first().evaluate(dataset)
        elif policy == "easy_first":
            result = p.easy_first().evaluate(dataset)
        elif policy == "stratified":
            result = p.stratified().evaluate(dataset)
        else:  # class_balance
            result = p.class_balanced(labels).evaluate(dataset)

        assert isinstance(result, PrioritizeOutput)
        assert result.policy == policy
        assert result.method == "knn"
        assert len(result.indices) == 1000
        assert any(i != j for i, j in zip(result.indices, range(1000)))

    def test_prioritize_method_not_configured_raises_valueerror(self):
        p = Prioritize(self.model, batch_size=self.batch_size, device=self.device)
        dataset = self.get_dataset()
        with pytest.raises(ValueError, match="Method not configured"):
            p.hard_first().evaluate(dataset)

    def test_prioritize_policy_not_configured_raises_valueerror(self):
        p = Prioritize(self.model, batch_size=self.batch_size, device=self.device)
        dataset = self.get_dataset()
        with pytest.raises(ValueError, match="Policy not configured"):
            p.with_knn(k=5).evaluate(dataset)

    def test_prioritize_dataset_class_balance_without_labels_does_not_raise_valueerror(self):
        p = Prioritize(self.model, batch_size=self.batch_size, device=self.device)
        dataset = self.get_dataset(n=100)
        # class_balanced() with no labels will extract from AnnotatedDataset
        p.with_knn(k=5).class_balanced().evaluate(dataset)

    def test_prioritize_embeddings_class_balance_without_labels_raises_valueerror(self):
        p = Prioritize(self.model, batch_size=self.batch_size, device=self.device)
        embeddings = self.get_embeddings(n=100)
        with pytest.raises(ValueError, match="class_labels must be provided"):
            p.with_knn(k=5).class_balanced().evaluate(embeddings)

    def test_prioritize_stratified_no_scores_raises_valueerror(self):
        p = Prioritize(self.model, batch_size=self.batch_size, device=self.device)
        dataset = self.get_dataset(n=100)
        with pytest.raises(ValueError, match="Ranking scores are necessary"):
            p.with_kmeans_complexity(c=5).stratified().evaluate(dataset)

    @pytest.mark.parametrize("use_embeddings", [False, True])
    @pytest.mark.parametrize("use_reference", [False, True])
    @pytest.mark.parametrize(
        "method, method_kwargs",
        (
            ("knn", {"k": 10}),
            ("kmeans_distance", {"c": 100}),
            ("kmeans_complexity", {"c": 100}),
        ),
    )
    def test_prioritize_with_embeddings(self, method, method_kwargs, use_embeddings, use_reference):
        # Setup reference in __init__ if needed
        reference = self.get_embeddings() if use_reference else None
        p = Prioritize(self.model, reference=reference, batch_size=self.batch_size, device=self.device)

        # Pass either Embeddings or AnnotatedDataset as dataset parameter
        dataset = self.get_embeddings() if use_embeddings else self.get_dataset()

        # Configure using builder pattern
        if method == "knn":
            p.with_knn(**method_kwargs)
        elif method == "kmeans_distance":
            p.with_kmeans_distance(**method_kwargs)
        elif method == "kmeans_complexity":
            p.with_kmeans_complexity(**method_kwargs)

        result = p.hard_first().evaluate(dataset)

        assert isinstance(result, PrioritizeOutput)
        assert len(result.indices) == 1000
        assert result.method == method
        assert result.policy == "hard_first"
        assert any(i != j for i, j in zip(result.indices, range(1000)))

    def test_prioritize_with_precomputed_embeddings(self):
        """Test that we can pass Embeddings directly as the dataset."""
        p = Prioritize(self.model, batch_size=self.batch_size, device=self.device)
        embeddings = self.get_embeddings(100)
        result = p.with_knn(k=5).easy_first().evaluate(embeddings)

        assert isinstance(result, PrioritizeOutput)
        assert len(result.indices) == 100
        assert result.method == "knn"
        assert result.policy == "easy_first"

    def test_prioritize_output_data(self):
        """Test PrioritizeOutput.data() method."""
        indices = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.intp)
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)
        output = PrioritizeOutput(indices=indices, scores=scores, method="knn", policy="hard_first")
        assert np.array_equal(output.data(), indices)
        assert len(output) == 8

    def test_prioritize_output_without_scores(self):
        """Test PrioritizeOutput with scores=None."""
        indices = np.array([3, 1, 4], dtype=np.intp)
        output = PrioritizeOutput(indices=indices, scores=None, method="kmeans_complexity", policy="easy_first")
        assert output.scores is None
        assert len(output) == 3
        assert output.method == "kmeans_complexity"
        assert output.policy == "easy_first"

    def test_prioritize_with_reference_dataset(self):
        """Test Prioritize with reference dataset provided at init."""
        reference_dataset = self.get_dataset(500)
        p = Prioritize(self.model, reference=reference_dataset, batch_size=self.batch_size, device=self.device)

        dataset = self.get_dataset(100)
        result = p.with_knn(k=10).hard_first().evaluate(dataset)

        assert isinstance(result, PrioritizeOutput)
        assert len(result.indices) == 100
        assert result.method == "knn"
        assert result.policy == "hard_first"

    def test_prioritize_override_batch_size_device(self):
        """Test overriding batch_size and device in evaluate()."""
        p = Prioritize(self.model, batch_size=32, device=self.device)
        dataset = self.get_dataset(100)

        # Should work with overridden parameters
        result = p.with_knn(k=5).hard_first().evaluate(dataset, batch_size=16)

        assert isinstance(result, PrioritizeOutput)
        assert len(result.indices) == 100

    def test_prioritize_reconfigure_and_reuse(self):
        """Test that we can reconfigure and reuse the same prioritizer instance."""
        p = Prioritize(self.model, batch_size=self.batch_size, device=self.device)
        dataset1 = self.get_dataset(100)
        dataset2 = self.get_dataset(200)

        # First configuration
        result1 = p.with_knn(k=5).hard_first().evaluate(dataset1)
        assert len(result1.indices) == 100
        assert result1.method == "knn"

        # Reconfigure with different method
        result2 = p.with_kmeans_distance(c=10).easy_first().evaluate(dataset2)
        assert len(result2.indices) == 200
        assert result2.method == "kmeans_distance"
        assert result2.policy == "easy_first"
