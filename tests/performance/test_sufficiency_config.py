"""
Tests for SufficiencyConfig dataclass.

These tests verify that the configuration class correctly stores
and validates sufficiency analysis parameters.
"""

from dataclasses import FrozenInstanceError

import pytest

from dataeval.performance import SufficiencyConfig
from dataeval.protocols import EvaluationStrategy, TrainingStrategy


class TestSufficiencyConfigConstruction:
    """Test SufficiencyConfig construction and initialization."""

    def test_config_stores_strategies(self, mock_training_strategy, mock_evaluation_strategy):
        """Verify config correctly stores strategy objects."""

        config = SufficiencyConfig(mock_training_strategy, mock_evaluation_strategy)

        assert config.training_strategy is mock_training_strategy
        assert config.evaluation_strategy is mock_evaluation_strategy

    def test_config_stores_run_parameters(self, mock_training_strategy, mock_evaluation_strategy):
        """Verify config stores runs and substeps."""

        config = SufficiencyConfig(mock_training_strategy, mock_evaluation_strategy, runs=5, substeps=10)

        assert config.runs == 5
        assert config.substeps == 10

    def test_config_default_values(self, mock_training_strategy, mock_evaluation_strategy):
        """Verify config applies correct default values."""

        config = SufficiencyConfig(mock_training_strategy, mock_evaluation_strategy)

        assert config.runs == 1
        assert config.substeps == 5
        assert config.unit_interval

    def test_config_with_custom_unit_interval(self, mock_training_strategy, mock_evaluation_strategy):
        """Verify config accepts custom unit_interval value."""

        config = SufficiencyConfig(mock_training_strategy, mock_evaluation_strategy, unit_interval=False)

        assert not config.unit_interval


class TestSufficiencyConfigValidation:
    """Test SufficiencyConfig validation logic."""

    @pytest.mark.parametrize("runs", [-1, 0])
    def test_config_rejects_negative_runs(self, mock_training_strategy, mock_evaluation_strategy, runs):
        """Verify config validates runs is positive."""

        with pytest.raises(ValueError, match="runs must be positive"):
            SufficiencyConfig(mock_training_strategy, mock_evaluation_strategy, runs=runs)

    @pytest.mark.parametrize("substeps", [-1, 0])
    def test_config_rejects_negative_substeps(self, mock_training_strategy, mock_evaluation_strategy, substeps):
        """Verify config validates substeps is positive."""
        with pytest.raises(ValueError, match="substeps must be positive"):
            SufficiencyConfig(mock_training_strategy, mock_evaluation_strategy, substeps=substeps)

    def test_config_accepts_positive_values(self, mock_training_strategy, mock_evaluation_strategy):
        """Verify config accepts valid positive values."""

        config = SufficiencyConfig(mock_training_strategy, mock_evaluation_strategy, runs=10, substeps=20)

        assert config.runs == 10
        assert config.substeps == 20


class TestSufficiencyConfigImmutability:
    """Test that SufficiencyConfig is immutable (frozen)."""

    def test_config_is_frozen(self, mock_training_strategy, mock_evaluation_strategy):
        """Verify config cannot be modified after creation."""

        config = SufficiencyConfig(mock_training_strategy, mock_evaluation_strategy, runs=3)

        with pytest.raises(FrozenInstanceError):
            config.runs = 5  # pyright: ignore[reportAttributeAccessIssue] --> This is what we are testing

    def test_config_training_strategy_is_frozen(self, mock_training_strategy, mock_evaluation_strategy):
        """Verify training_strategy field cannot be modified."""

        config = SufficiencyConfig(mock_training_strategy, mock_evaluation_strategy)

        new_training = mock_training_strategy
        with pytest.raises(FrozenInstanceError):
            config.training_strategy = new_training  # pyright: ignore[reportAttributeAccessIssue] --> This is what we are testing

    def test_config_evaluation_strategy_is_frozen(self, mock_training_strategy, mock_evaluation_strategy):
        """Verify evaluation_strategy field cannot be modified."""

        config = SufficiencyConfig(mock_training_strategy, mock_evaluation_strategy)

        new_evaluation = mock_evaluation_strategy
        with pytest.raises(FrozenInstanceError):
            config.evaluation_strategy = new_evaluation  # pyright: ignore[reportAttributeAccessIssue] --> This is what we are testing


class TestSufficiencyConfigTypeChecking:
    """Test that config enforces protocol conformance."""

    def test_config_accepts_training_strategy(self, mock_training_strategy, mock_evaluation_strategy):
        """Verify config accepts objects conforming to TrainingStrategy."""

        # Should not raise
        config = SufficiencyConfig(mock_training_strategy, mock_evaluation_strategy)

        assert isinstance(config.training_strategy, TrainingStrategy)

    def test_config_accepts_evaluation_strategy(self, mock_training_strategy, mock_evaluation_strategy):
        """Verify config accepts objects conforming to EvaluationStrategy."""

        # Should not raise
        config = SufficiencyConfig(mock_training_strategy, mock_evaluation_strategy)

        assert isinstance(config.evaluation_strategy, EvaluationStrategy)


class TestSufficiencyConfigEquality:
    """Test SufficiencyConfig equality and hashing."""

    def test_config_equality_with_same_values(self, mock_training_strategy, mock_evaluation_strategy):
        """Verify configs with same values are equal."""

        config1 = SufficiencyConfig(mock_training_strategy, mock_evaluation_strategy, runs=3, substeps=5)
        config2 = SufficiencyConfig(mock_training_strategy, mock_evaluation_strategy, runs=3, substeps=5)

        # Note: They use the same strategy instances
        assert config1 == config2

    def test_config_inequality_with_different_runs(self, mock_training_strategy, mock_evaluation_strategy):
        """Verify configs with different runs are not equal."""

        config1 = SufficiencyConfig(mock_training_strategy, mock_evaluation_strategy, runs=3)
        config2 = SufficiencyConfig(mock_training_strategy, mock_evaluation_strategy, runs=5)

        assert config1 != config2

    def test_config_repr(self, mock_training_strategy, mock_evaluation_strategy):
        """Verify config has useful string representation."""

        config = SufficiencyConfig(mock_training_strategy, mock_evaluation_strategy, runs=3, substeps=10)
        repr_str = repr(config)

        # Should contain class name and key parameters
        assert "SufficiencyConfig" in repr_str
        assert "runs=3" in repr_str
        assert "substeps=10" in repr_str
