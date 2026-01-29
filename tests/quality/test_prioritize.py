from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from dataeval._embeddings import Embeddings
from dataeval.encoders import TorchEmbeddingEncoder
from dataeval.protocols import AnnotatedDataset
from dataeval.quality import Prioritize, PrioritizeOutput


class TestPrioritize:
    @pytest.fixture
    def encoder(self):
        """Create a TorchEmbeddingEncoder for testing."""
        return TorchEmbeddingEncoder(torch.nn.Flatten(), batch_size=10, device="cpu")

    def get_dataset(self, n: int = 1000):
        mock = MagicMock(spec=AnnotatedDataset)
        mock.__len__.return_value = n
        # Use side_effect to return different random data for each call (not the same data every time)
        mock.__getitem__.side_effect = lambda _: (np.random.random((1, 10, 10)), np.zeros(10), {})
        mock.metadata = {"id": "mock_dataset", "index2label": {i: str(i) for i in range(10)}}
        return mock

    def get_embeddings(self, encoder, n: int = 1000) -> Embeddings:
        return Embeddings(self.get_dataset(n), encoder=encoder)

    @pytest.mark.parametrize(
        "method, method_kwargs",
        (
            ("knn", {"k": 10}),
            ("kmeans_distance", {"c": 100}),
            ("kmeans_complexity", {"c": 100}),
        ),
    )
    def test_prioritize_evaluate(self, encoder, method, method_kwargs):
        dataset = self.get_dataset()

        # Use factory class methods
        if method == "knn":
            p = Prioritize.knn(encoder, **method_kwargs)
        elif method == "kmeans_distance":
            p = Prioritize.kmeans_distance(encoder, **method_kwargs)
        elif method == "kmeans_complexity":
            p = Prioritize.kmeans_complexity(encoder, **method_kwargs)
        else:
            assert False, f"Unknown method: {method}"

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
    def test_prioritize_policies(self, encoder, policy):
        dataset = self.get_dataset()
        labels = np.random.randint(low=0, high=10, size=1000)

        # Use factory method to create base instance
        p = Prioritize.knn(encoder, k=10)

        # Configure policy using fluent methods
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

    def test_prioritize_encoder_required(self):
        """Test that encoder must be provided."""
        with pytest.raises(ValueError, match="encoder must be provided"):
            Prioritize()

    def test_prioritize_default_method_and_policy(self, encoder):
        """Test that default method (knn) and policy (hard_first) are used."""
        p = Prioritize(encoder=encoder)
        dataset = self.get_dataset(n=100)
        result = p.evaluate(dataset)

        assert result.method == "knn"
        assert result.policy == "hard_first"
        assert len(result.indices) == 100

    def test_prioritize_dataset_class_balance_without_labels_does_not_raise_valueerror(self, encoder):
        dataset = self.get_dataset(n=100)
        # class_balanced() with no labels will extract from AnnotatedDataset
        Prioritize.knn(encoder, k=5).class_balanced().evaluate(dataset)

    def test_prioritize_embeddings_class_balance_without_labels_raises_valueerror(self, encoder):
        embeddings = self.get_embeddings(encoder, n=100)
        with pytest.raises(ValueError, match="class_labels must be provided"):
            Prioritize.knn(encoder, k=5).class_balanced().evaluate(embeddings)

    def test_prioritize_stratified_no_scores_raises_valueerror(self, encoder):
        dataset = self.get_dataset(n=100)
        with pytest.raises(ValueError, match="stratified policy is not available"):
            Prioritize.kmeans_complexity(encoder, c=5).stratified().evaluate(dataset)

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
    def test_prioritize_with_embeddings(self, encoder, method, method_kwargs, use_embeddings, use_reference):
        # Setup reference if needed
        reference = self.get_embeddings(encoder) if use_reference else None

        # Pass either Embeddings or AnnotatedDataset as dataset parameter
        dataset = self.get_embeddings(encoder) if use_embeddings else self.get_dataset()

        # Use factory class methods with reference
        if method == "knn":
            p = Prioritize.knn(encoder, reference=reference, **method_kwargs)
        elif method == "kmeans_distance":
            p = Prioritize.kmeans_distance(encoder, reference=reference, **method_kwargs)
        elif method == "kmeans_complexity":
            p = Prioritize.kmeans_complexity(encoder, reference=reference, **method_kwargs)
        else:
            assert False, f"Unknown method: {method}"

        result = p.hard_first().evaluate(dataset)

        assert isinstance(result, PrioritizeOutput)
        assert len(result.indices) == 1000
        assert result.method == method
        assert result.policy == "hard_first"
        assert any(i != j for i, j in zip(result.indices, range(1000)))

    def test_prioritize_with_precomputed_embeddings(self, encoder):
        """Test that we can pass Embeddings directly as the dataset."""
        embeddings = self.get_embeddings(encoder, 100)
        result = Prioritize.knn(encoder, k=5).easy_first().evaluate(embeddings)

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

    def test_prioritize_with_reference_dataset(self, encoder):
        """Test Prioritize with reference dataset provided at init."""
        reference_dataset = self.get_dataset(500)
        dataset = self.get_dataset(100)

        result = Prioritize.knn(encoder, k=10, reference=reference_dataset).hard_first().evaluate(dataset)

        assert isinstance(result, PrioritizeOutput)
        assert len(result.indices) == 100
        assert result.method == "knn"
        assert result.policy == "hard_first"

    def test_prioritize_immutable_policy_methods(self, encoder):
        """Test that policy methods return new instances (immutable pattern)."""
        p1 = Prioritize.knn(encoder, k=5)
        p2 = p1.easy_first()
        p3 = p1.hard_first()

        # p1 should still have default policy
        assert p1.policy == "hard_first"
        # p2 and p3 are new instances with different policies
        assert p2.policy == "easy_first"
        assert p3.policy == "hard_first"
        # They should be different objects
        assert p1 is not p2
        assert p1 is not p3
        assert p2 is not p3

    def test_prioritize_direct_instantiation(self, encoder):
        """Test direct instantiation with all parameters."""
        dataset = self.get_dataset(100)

        p = Prioritize(
            encoder=encoder,
            method="kmeans_distance",
            c=10,
            policy="stratified",
            num_bins=20,
        )
        result = p.evaluate(dataset)

        assert result.method == "kmeans_distance"
        assert result.policy == "stratified"
        assert len(result.indices) == 100

    def test_prioritize_config_usage(self, encoder):
        """Test using Config object for configuration."""
        config = Prioritize.Config(
            encoder=encoder,
            method="knn",
            k=10,
            policy="easy_first",
        )
        p = Prioritize(config=config)
        dataset = self.get_dataset(100)
        result = p.evaluate(dataset)

        assert result.method == "knn"
        assert result.policy == "easy_first"
        assert len(result.indices) == 100
