"""
Tests for SufficiencyConfig dataclass.

These tests verify that the configuration class correctly stores
and validates sufficiency analysis parameters.
"""

from dataclasses import FrozenInstanceError

import pytest

from dataeval.protocols import EvaluationStrategy, TrainingStrategy
from dataeval.workflows.sufficiency import SufficiencyConfig


class MockTrainingStrategy:
    """Simple training strategy for testing."""

    def train(self, model, dataset, indices):
        pass


class MockEvaluationStrategy:
    """Simple evaluation strategy for testing."""

    def evaluate(self, model, dataset):
        return {"accuracy": 0.95}


class TestSufficiencyConfigConstruction:
    """Test SufficiencyConfig construction and initialization."""

    def test_config_stores_strategies(self):
        """Verify config correctly stores strategy objects."""
        training = MockTrainingStrategy()
        evaluation = MockEvaluationStrategy()

        config = SufficiencyConfig(training, evaluation)

        assert config.training_strategy is training
        assert config.evaluation_strategy is evaluation

    def test_config_stores_run_parameters(self):
        """Verify config stores runs and substeps."""
        training = MockTrainingStrategy()
        evaluation = MockEvaluationStrategy()

        config = SufficiencyConfig(training, evaluation, runs=5, substeps=10)

        assert config.runs == 5
        assert config.substeps == 10

    def test_config_default_values(self):
        """Verify config applies correct default values."""
        training = MockTrainingStrategy()
        evaluation = MockEvaluationStrategy()

        config = SufficiencyConfig(training, evaluation)

        assert config.runs == 1
        assert config.substeps == 5
        assert config.unit_interval

    def test_config_with_custom_unit_interval(self):
        """Verify config accepts custom unit_interval value."""
        training = MockTrainingStrategy()
        evaluation = MockEvaluationStrategy()

        config = SufficiencyConfig(training, evaluation, unit_interval=False)

        assert not config.unit_interval


class TestSufficiencyConfigValidation:
    """Test SufficiencyConfig validation logic."""

    def test_config_rejects_negative_runs(self):
        """Verify config validates runs is positive."""
        training = MockTrainingStrategy()
        evaluation = MockEvaluationStrategy()

        with pytest.raises(ValueError, match="runs must be positive"):
            SufficiencyConfig(training, evaluation, runs=-1)

    def test_config_rejects_zero_runs(self):
        """Verify config validates runs is greater than zero."""
        training = MockTrainingStrategy()
        evaluation = MockEvaluationStrategy()

        with pytest.raises(ValueError, match="runs must be positive"):
            SufficiencyConfig(training, evaluation, runs=0)

    def test_config_rejects_negative_substeps(self):
        """Verify config validates substeps is positive."""
        training = MockTrainingStrategy()
        evaluation = MockEvaluationStrategy()

        with pytest.raises(ValueError, match="substeps must be positive"):
            SufficiencyConfig(training, evaluation, substeps=-1)

    def test_config_rejects_zero_substeps(self):
        """Verify config validates substeps is greater than zero."""
        training = MockTrainingStrategy()
        evaluation = MockEvaluationStrategy()

        with pytest.raises(ValueError, match="substeps must be positive"):
            SufficiencyConfig(training, evaluation, substeps=0)

    def test_config_accepts_positive_values(self):
        """Verify config accepts valid positive values."""
        training = MockTrainingStrategy()
        evaluation = MockEvaluationStrategy()

        # Should not raise
        config = SufficiencyConfig(training, evaluation, runs=10, substeps=20)

        assert config.runs == 10
        assert config.substeps == 20


class TestSufficiencyConfigImmutability:
    """Test that SufficiencyConfig is immutable (frozen)."""

    def test_config_is_frozen(self):
        """Verify config cannot be modified after creation."""
        training = MockTrainingStrategy()
        evaluation = MockEvaluationStrategy()

        config = SufficiencyConfig(training, evaluation, runs=3)

        with pytest.raises(FrozenInstanceError):
            config.runs = 5  # pyright: ignore[reportAttributeAccessIssue] --> This is what we are testing

    def test_config_training_strategy_is_frozen(self):
        """Verify training_strategy field cannot be modified."""
        training = MockTrainingStrategy()
        evaluation = MockEvaluationStrategy()

        config = SufficiencyConfig(training, evaluation)

        new_training = MockTrainingStrategy()
        with pytest.raises(FrozenInstanceError):
            config.training_strategy = new_training  # pyright: ignore[reportAttributeAccessIssue] --> This is what we are testing

    def test_config_evaluation_strategy_is_frozen(self):
        """Verify evaluation_strategy field cannot be modified."""
        training = MockTrainingStrategy()
        evaluation = MockEvaluationStrategy()

        config = SufficiencyConfig(training, evaluation)

        new_evaluation = MockEvaluationStrategy()
        with pytest.raises(FrozenInstanceError):
            config.evaluation_strategy = new_evaluation  # pyright: ignore[reportAttributeAccessIssue] --> This is what we are testing


class TestSufficiencyConfigTypeChecking:
    """Test that config enforces protocol conformance."""

    def test_config_accepts_training_strategy(self):
        """Verify config accepts objects conforming to TrainingStrategy."""
        training = MockTrainingStrategy()
        evaluation = MockEvaluationStrategy()

        # Should not raise
        config = SufficiencyConfig(training, evaluation)

        assert isinstance(config.training_strategy, TrainingStrategy)

    def test_config_accepts_evaluation_strategy(self):
        """Verify config accepts objects conforming to EvaluationStrategy."""
        training = MockTrainingStrategy()
        evaluation = MockEvaluationStrategy()

        # Should not raise
        config = SufficiencyConfig(training, evaluation)

        assert isinstance(config.evaluation_strategy, EvaluationStrategy)


class TestSufficiencyConfigEquality:
    """Test SufficiencyConfig equality and hashing."""

    def test_config_equality_with_same_values(self):
        """Verify configs with same values are equal."""
        training = MockTrainingStrategy()
        evaluation = MockEvaluationStrategy()

        config1 = SufficiencyConfig(training, evaluation, runs=3, substeps=5)
        config2 = SufficiencyConfig(training, evaluation, runs=3, substeps=5)

        # Note: They use the same strategy instances
        assert config1 == config2

    def test_config_inequality_with_different_runs(self):
        """Verify configs with different runs are not equal."""
        training = MockTrainingStrategy()
        evaluation = MockEvaluationStrategy()

        config1 = SufficiencyConfig(training, evaluation, runs=3)
        config2 = SufficiencyConfig(training, evaluation, runs=5)

        assert config1 != config2

    def test_config_repr(self):
        """Verify config has useful string representation."""
        training = MockTrainingStrategy()
        evaluation = MockEvaluationStrategy()

        config = SufficiencyConfig(training, evaluation, runs=3, substeps=10)
        repr_str = repr(config)

        # Should contain class name and key parameters
        assert "SufficiencyConfig" in repr_str
        assert "runs=3" in repr_str
        assert "substeps=10" in repr_str
