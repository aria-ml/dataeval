from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from dataeval._embeddings import Embeddings
from dataeval.core._rank import RankResult
from dataeval.encoders import TorchEmbeddingEncoder
from dataeval.protocols import AnnotatedDataset
from dataeval.quality._prioritize import Prioritize, PrioritizeOutput


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
            ("hdbscan_distance", {"c": 10}),
            ("hdbscan_complexity", {"c": 10}),
        ),
    )
    def test_prioritize_evaluate(self, encoder, method, method_kwargs):
        dataset = self.get_dataset()

        # Use factory class methods (Configures the "Measure" phase)
        if method == "knn":
            p = Prioritize.knn(encoder, **method_kwargs)
        elif method == "kmeans_distance":
            p = Prioritize.kmeans_distance(encoder, **method_kwargs)
        elif method == "kmeans_complexity":
            p = Prioritize.kmeans_complexity(encoder, **method_kwargs)
        elif method == "hdbscan_distance":
            p = Prioritize.hdbscan_distance(encoder, **method_kwargs)
        elif method == "hdbscan_complexity":
            p = Prioritize.hdbscan_complexity(encoder, **method_kwargs)
        else:
            assert False, f"Unknown method: {method}"

        # Evaluate (Measure) then Apply Policy (Cut)
        result = p.evaluate(dataset).hard_first()

        # Check result type and attributes
        assert isinstance(result, PrioritizeOutput)
        assert len(result.indices) == 1000
        assert result.method == method
        assert result.order == "hard_first"
        # Check that indices are actually different from default order
        assert any(i != j for i, j in zip(result.indices, range(1000)))

    @pytest.mark.parametrize(
        "order,policy",
        [
            ("hard_first", "difficulty"),
            ("easy_first", "difficulty"),
            ("easy_first", "stratified"),
            ("easy_first", "class_balanced"),
        ],
    )
    def test_prioritize_policies(self, encoder, order, policy):
        dataset = self.get_dataset()
        labels = np.random.randint(low=0, high=10, size=1000)

        # Use factory method to create base instance
        p = Prioritize.knn(encoder, k=10)
        base_result = p.evaluate(dataset)

        # Configure policy using fluent methods on PrioritizeOutput
        if order == "hard_first" and policy == "difficulty":
            result = base_result.hard_first()
        elif order == "easy_first" and policy == "difficulty":
            result = base_result.easy_first()
        elif policy == "stratified":
            result = base_result.stratified()
        else:  # class_balanced
            result = base_result.class_balanced(labels)

        assert isinstance(result, PrioritizeOutput)
        assert result.order == order
        assert result.policy == policy
        assert result.method == "knn"
        assert len(result.indices) == 1000
        assert any(i != j for i, j in zip(result.indices, range(1000)))

    def test_prioritize_encoder_required(self):
        """Test that encoder must be provided."""
        with pytest.raises(ValueError, match="encoder must be provided"):
            Prioritize()

    def test_prioritize_default_method_and_policy(self, encoder):
        """Test that default method (knn) and order (easy_first) are used."""
        p = Prioritize(encoder=encoder)
        dataset = self.get_dataset(n=100)
        result = p.evaluate(dataset)

        assert result.method == "knn"
        assert result.order == "easy_first"
        assert result.policy == "difficulty"
        assert len(result.indices) == 100

    def test_prioritize_dataset_class_balance_without_labels_does_not_raise_valueerror(self, encoder):
        dataset = self.get_dataset(n=100)
        # class_balanced() with no labels will extract metadata from AnnotatedDataset automatically
        # Note: evaluate() extracts the labels, then class_balanced() uses them.
        Prioritize.knn(encoder, k=5).evaluate(dataset).class_balanced()

    def test_prioritize_embeddings_class_balance_without_labels_raises_valueerror(self, encoder):
        embeddings = self.get_embeddings(encoder, n=100)

        # Evaluate works fine (just calculating scores)
        result = Prioritize.knn(encoder, k=5).evaluate(embeddings)

        # Apply policy fails because embeddings have no metadata
        with pytest.raises(ValueError, match="class_labels must be provided"):
            result.class_balanced()

    def test_prioritize_stratified_no_scores_raises_valueerror(self, encoder):
        dataset = self.get_dataset(n=100)

        # Evaluate works fine
        result = Prioritize.kmeans_complexity(encoder, c=5).evaluate(dataset)

        # Apply policy fails because complexity methods don't have scores
        with pytest.raises(ValueError, match="Cannot apply stratified policy"):
            result.stratified()

    @pytest.mark.parametrize("use_embeddings", [False, True])
    @pytest.mark.parametrize("use_reference", [False, True])
    @pytest.mark.parametrize(
        "method, method_kwargs",
        (
            ("knn", {"k": 10}),
            ("kmeans_distance", {"c": 100}),
            ("kmeans_complexity", {"c": 100}),
            ("hdbscan_distance", {"c": 10}),
            ("hdbscan_complexity", {"c": 10}),
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
        elif method == "hdbscan_distance":
            p = Prioritize.hdbscan_distance(encoder, reference=reference, **method_kwargs)
        elif method == "hdbscan_complexity":
            p = Prioritize.hdbscan_complexity(encoder, reference=reference, **method_kwargs)
        else:
            assert False, f"Unknown method: {method}"

        # Test flow: Evaluate -> Rebucket
        result = p.evaluate(dataset).hard_first()

        assert isinstance(result, PrioritizeOutput)
        assert len(result.indices) == 1000
        assert result.method == method
        assert result.order == "hard_first"
        assert any(i != j for i, j in zip(result.indices, range(1000)))

    def test_prioritize_with_precomputed_embeddings(self, encoder):
        """Test that we can pass Embeddings directly as the dataset."""
        embeddings = self.get_embeddings(encoder, 100)
        # Evaluate -> Easy First
        result = Prioritize.knn(encoder, k=5).evaluate(embeddings).easy_first()

        assert isinstance(result, PrioritizeOutput)
        assert len(result.indices) == 100
        assert result.method == "knn"
        assert result.order == "easy_first"

    def test_rank_result_indices(self):
        """Test RankResult.indices property."""
        indices = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.intp)
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)
        rank_result = RankResult(indices=indices, scores=scores)

        output = PrioritizeOutput(rank_result=rank_result, method="knn", order="hard_first")
        assert np.array_equal(output.indices, indices[::-1])  # hard_first reverses indices
        assert output.method == "knn"
        assert output.order == "hard_first"

    def test_rank_result_without_scores(self):
        """Test RankResult with scores=None."""
        indices = np.array([3, 1, 4], dtype=np.intp)
        rank_result = RankResult(indices=indices, scores=None)

        output = PrioritizeOutput(rank_result=rank_result, method="kmeans_complexity", order="easy_first")
        assert output.scores is None
        assert output.method == "kmeans_complexity"
        assert output.order == "easy_first"

    def test_prioritize_with_reference_dataset(self, encoder):
        """Test Prioritize with reference dataset provided at init."""
        reference_dataset = self.get_dataset(500)
        dataset = self.get_dataset(100)

        # Config(reference) -> evaluate(data) -> hard_first()
        result = Prioritize.knn(encoder, k=10, reference=reference_dataset).evaluate(dataset).hard_first()

        assert isinstance(result, PrioritizeOutput)
        assert len(result.indices) == 100
        assert result.method == "knn"
        assert result.order == "hard_first"

    def test_prioritize_output_immutable_methods(self, encoder):
        """Test that Output transformation methods return new instances (immutable pattern)."""
        dataset = self.get_dataset(100)

        # Get initial result
        r1 = Prioritize.knn(encoder, k=5).evaluate(dataset)

        # Create derived results
        r2 = r1.hard_first()
        r3 = r2.easy_first()

        # r1 should still have default order (usually easy_first or whatever evaluate produces)
        # Note: evaluate() uses default order from Config if provided, otherwise defaults to 'easy_first'
        assert r1.order == "easy_first"

        # r2 and r3 are new instances with different orders
        assert r2.order == "hard_first"
        assert r3.order == "easy_first"

        # They should be different objects
        assert r1 is not r2
        assert r1 is not r3
        assert r2 is not r3

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
            order="hard_first",
        )
        p = Prioritize(config=config)
        dataset = self.get_dataset(100)
        result = p.evaluate(dataset)

        assert result.method == "knn"
        assert result.order == "hard_first"
        assert len(result.indices) == 100


class TestPrioritizeEdgeCases:
    def test_prioritize_output_methods(self):
        """
        Covers magic methods (__iter__, __len__, __contains__, keys, values, items, get)
        and idempotency of fluent methods.

        """
        # Construct dummy output
        rank_result = RankResult(indices=np.array([0, 1]), scores=np.array([0.9, 0.1]))
        output = PrioritizeOutput(rank_result, method="knn", order="easy_first")

        # Test Magic Methods
        assert "method" in output
        assert len(output) == 5  # fields count
        assert output.get("method") == "knn"
        assert output.get("invalid_key", "default") == "default"
        assert list(output.keys()) == ["indices", "scores", "method", "order", "policy"]

        # Test __repr__
        assert "PriorityOutput" in repr(output)

        # Test Idempotency (logging check implies coverage of return self)
        easy = output.easy_first()
        assert easy is output  # Should return self if already easy_first

        hard = output.hard_first()
        assert hard.order == "hard_first"
        hard2 = hard.hard_first()
        assert hard2 is hard  # Should return self

    def test_evaluate_error_invalid_data(self):
        # Class balanced without labels (Line 234, 788)
        embeddings = MagicMock()
        embeddings.__class__.__name__ = "Embeddings"

        # Use dummy encoder for instantiation
        p = Prioritize(encoder=MagicMock(), method="knn", policy="class_balanced")

        # If we pass a dataset that isn't annotated/embeddings (Line 759)
        with pytest.raises(TypeError, match="must be either an AnnotatedDataset or Embeddings"):
            p.evaluate("invalid_string_dataset")  # type: ignore

    def test_evaluate_error_invalid_method(self):
        # We bypass __init__ validation by modifying attribute directly or sub-classing
        p = Prioritize(encoder=MagicMock())
        p.method = "invalid_method"  # type: ignore

        # Mock embeddings to pass first check
        mock_emb = MagicMock()
        # Mock __array__ to return numpy
        mock_emb.__array__ = MagicMock(return_value=np.zeros((5, 5)))

        # Set reference to None to hit simple path
        p._reference = None
        p._embeddings = mock_emb
        p._ref_embeddings = None

        # Calling _perform_ranking directly or via evaluate catch
        with pytest.raises(ValueError, match="Invalid method"):
            p._perform_ranking(np.zeros((5, 5)), None)

    def test_stratified_error_no_scores(self):
        """Covers error when applying stratified to methods without scores."""
        rank_result = RankResult(indices=np.array([0, 1]), scores=None)  # No scores
        output = PrioritizeOutput(rank_result, method="kmeans_complexity")

        with pytest.raises(ValueError, match="scores are not available"):
            output.stratified()
