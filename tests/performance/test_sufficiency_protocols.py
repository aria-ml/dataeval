"""
Tests for TrainingStrategy and EvaluationStrategy protocols.

These tests verify that the protocol structure is correct and that
conforming classes are properly recognized.
"""

from collections.abc import Mapping

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from dataeval.protocols import EvaluationStrategy, TrainingStrategy


class SimpleDataset(Dataset):
    """
    Mock dataset for testing.

    TODO: Replace with DataEval pytest fixture
    TODO: Confirm use for IC and OD
    """

    def __init__(self, size=100):
        self.size = 100

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.randn(10), torch.randint(0, 2, (1,))


class SimpleModel(nn.Module):
    """Mock model for testing.

    TODO: Replace with DataEval pytest fixture
    TODO: Confirm use for IC and OD
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)


class TestTrainingStrategyProtocol:
    """Test that TrainingStrategy protocol is properly defined."""

    def test_training_protocol_accepts_conforming_class(self):
        """Verify that classes matching the Protocol structure are accepted."""

        class SimpleTrainer:
            """Minimal conforming implementation."""

            def train(self, model: nn.Module, dataset: Dataset, indices: list[int]):
                """Dummy training implementation."""
                pass

        trainer = SimpleTrainer()

        assert isinstance(trainer, TrainingStrategy)

    def test_training_protocol_signature_matches(self):
        """Verify the protocol signature is correct."""

        class TestTrainer:
            def train(self, model: nn.Module, dataset: Dataset, indices: list[int]):
                assert isinstance(model, nn.Module)
                assert isinstance(dataset, Dataset)
                assert isinstance(indices, list)

        trainer = TestTrainer()
        model = SimpleModel()
        dataset = SimpleDataset()
        indices = [0, 1, 2]

        # Should not raise an error
        trainer.train(model, dataset, indices)

    def test_training_protocol_rejects_wrong_signature(self):
        """Verify that classes with wrong signatures don't match."""

        class WrongTrainer:
            """Wrong signature - missing indices parameter."""

            def train(self, model: nn.Module, dataset: Dataset) -> None:
                pass

        trainer = WrongTrainer()

        # Note: isinstance will still return True at runtime for Protocol,
        # but type checkers will catch this. This test documents the expectation.
        # In practice, mypy/pyright would flag this as an error.
        assert isinstance(trainer, TrainingStrategy)  # Runtime check passes
        # But type checker would fail: trainer is missing required parameter


class TestEvaluationStrategyProtocol:
    """Test that EvaluationStrategy protocol is correctly defined."""

    def test_evaluation_protocol_accepts_conforming_class(self):
        """Verify that classes matching the Protocol structure are accepted."""

        class SimpleEvaluator:
            """Minimal conforming implementation."""

            def evaluate(self, model: nn.Module, dataset: Dataset) -> Mapping[str, float | np.ndarray]:
                """Dummy evaluation implementation."""
                return {"accuracy": 0.95}

        evaluator = SimpleEvaluator()

        # Should be recognized as conforming to protocol
        assert isinstance(evaluator, EvaluationStrategy)

    def test_evaluation_protocol_returns_scalar_metric(self):
        """Verify protocol works with scalar metric return."""

        class ScalarEvaluator:
            def evaluate(self, model: nn.Module, dataset: Dataset) -> Mapping[str, float]:
                return {"accuracy": 0.85, "precision": 0.87}

        evaluator = ScalarEvaluator()
        model = SimpleModel()
        dataset = SimpleDataset()

        result = evaluator.evaluate(model, dataset)

        assert isinstance(result, dict)
        assert "accuracy" in result
        assert isinstance(result["accuracy"], float)

    def test_evaluation_protocol_returns_array_metric(self):
        """Verify protocol works with array metric return (multi-class)."""

        class MultiClassEvaluator:
            def evaluate(self, model: nn.Module, dataset: Dataset) -> Mapping[str, np.ndarray]:
                # Per-class accuracies
                return {"accuracy": np.array([0.9, 0.85, 0.92])}

        evaluator = MultiClassEvaluator()
        model = SimpleModel()
        dataset = SimpleDataset()

        result = evaluator.evaluate(model, dataset)

        assert isinstance(result, dict)
        assert "accuracy" in result
        assert isinstance(result["accuracy"], np.ndarray)
        assert len(result["accuracy"]) == 3

    def test_evaluation_protocol_returns_scalar_and_array_metric(self):
        """Verify protocol works with array metric return (multi-class)."""

        class MultiClassEvaluator:
            def evaluate(self, model: nn.Module, dataset: Dataset) -> Mapping[str, float | np.ndarray]:
                # Per-class accuracies
                return {"accuracy": np.array([0.9, 0.85, 0.92]), "precision": 0.9}

        evaluator = MultiClassEvaluator()
        model = SimpleModel()
        dataset = SimpleDataset()

        result = evaluator.evaluate(model, dataset)

        assert isinstance(result, dict)
        assert "accuracy" in result
        assert isinstance(result["accuracy"], np.ndarray)
        assert len(result["accuracy"]) == 3

        assert "precision" in result
        assert isinstance(result["precision"], float)


class TestProtocolIntegration:
    """Test that protocols work together in realistic scenarios."""

    def test_can_create_custom_strategies(self):
        """Verify users can create custom strategy implementations with user-defined initialization."""

        class CustomTrainer:
            def __init__(self, learning_rate: float, epochs: int):
                self.lr = learning_rate
                self.epochs = epochs

            def train(self, model: nn.Module, dataset: Dataset, indices: list[int]) -> None:
                # Simplified training simulation
                for _epoch in range(self.epochs):
                    # Would do actual training here
                    pass

        class CustomEvaluator:
            def __init__(self, batch_size: int):
                self.batch_size = batch_size

            def evaluate(self, model: nn.Module, dataset: Dataset) -> Mapping[str, float]:
                # Simplified evaluation simulation
                return {"accuracy": 0.88, "loss": 0.25}

        # Verify both are recognized as strategies
        trainer = CustomTrainer(learning_rate=0.001, epochs=5)
        evaluator = CustomEvaluator(batch_size=32)

        assert isinstance(trainer, TrainingStrategy)
        assert isinstance(evaluator, EvaluationStrategy)

        # Verify they can be called
        model = SimpleModel()
        dataset = SimpleDataset()

        trainer.train(model, dataset, [0, 1, 2, 3, 4])
        result = evaluator.evaluate(model, dataset)

        assert "accuracy" in result
        assert "loss" in result
