"""
Tests for Sufficiency.Config dataclass.

These tests verify that the configuration class correctly stores
and validates sufficiency analysis parameters.
"""

import pytest

from dataeval.performance import Sufficiency
from dataeval.protocols import EvaluationStrategy, TrainingStrategy


class TestSufficiencyConfigConstruction:
    """Test Sufficiency.Config construction and initialization."""

    def test_config_stores_strategies(self, mock_train, mock_eval):
        """Verify config correctly stores strategy objects."""
        config = Sufficiency.Config(
            training_strategy=mock_train,
            evaluation_strategy=mock_eval,
        )

        assert config.training_strategy is mock_train
        assert config.evaluation_strategy is mock_eval

    def test_config_stores_run_parameters(self, mock_train, mock_eval):
        """Verify config stores runs and substeps."""
        config = Sufficiency.Config(training_strategy=mock_train, evaluation_strategy=mock_eval, runs=5, substeps=10)

        assert config.runs == 5
        assert config.substeps == 10

    def test_config_default_values(self, mock_train, mock_eval):
        """Verify config applies correct default values."""
        config = Sufficiency.Config(training_strategy=mock_train, evaluation_strategy=mock_eval)

        assert config.runs == 1
        assert config.substeps == 5
        assert config.unit_interval

    def test_config_with_custom_unit_interval(self, mock_train, mock_eval):
        """Verify config accepts custom unit_interval value."""
        config = Sufficiency.Config(
            training_strategy=mock_train,
            evaluation_strategy=mock_eval,
            unit_interval=False,
        )

        assert not config.unit_interval


class TestSufficiencyConfigValidation:
    """Test Sufficiency.Config validation logic."""

    @pytest.mark.parametrize("runs", [-1, 0])
    def test_config_rejects_negative_runs(self, mock_train, mock_eval, runs):
        """Verify config validates runs is positive."""
        with pytest.raises(ValueError, match="must be positive"):
            Sufficiency.Config(training_strategy=mock_train, evaluation_strategy=mock_eval, runs=runs)

    @pytest.mark.parametrize("substeps", [-1, 0])
    def test_config_rejects_negative_substeps(self, mock_train, mock_eval, substeps):
        """Verify config validates substeps is positive."""
        with pytest.raises(ValueError, match="must be positive"):
            Sufficiency.Config(training_strategy=mock_train, evaluation_strategy=mock_eval, substeps=substeps)

    def test_config_accepts_positive_values(self, mock_train, mock_eval):
        """Verify config accepts valid positive values."""
        config = Sufficiency.Config(training_strategy=mock_train, evaluation_strategy=mock_eval, runs=10, substeps=20)

        assert config.runs == 10
        assert config.substeps == 20


class TestSufficiencyConfigTypeChecking:
    """Test that config enforces protocol conformance."""

    def test_config_accepts_training_strategy(self, mock_train, mock_eval):
        """Verify config accepts objects conforming to TrainingStrategy."""
        # Should not raise
        config = Sufficiency.Config(training_strategy=mock_train, evaluation_strategy=mock_eval)

        assert isinstance(config.training_strategy, TrainingStrategy)

    def test_config_accepts_evaluation_strategy(self, mock_train, mock_eval):
        """Verify config accepts objects conforming to EvaluationStrategy."""
        # Should not raise
        config = Sufficiency.Config(training_strategy=mock_train, evaluation_strategy=mock_eval)

        assert isinstance(config.evaluation_strategy, EvaluationStrategy)


class TestSufficiencyConfigEquality:
    """Test Sufficiency.Config equality and hashing."""

    def test_config_equality_with_same_values(self, mock_train, mock_eval):
        """Verify configs with same values are equal."""
        config1 = Sufficiency.Config(training_strategy=mock_train, evaluation_strategy=mock_eval, runs=3, substeps=5)
        config2 = Sufficiency.Config(training_strategy=mock_train, evaluation_strategy=mock_eval, runs=3, substeps=5)

        # Note: They use the same strategy instances
        assert config1 == config2

    def test_config_inequality_with_different_runs(self, mock_train, mock_eval):
        """Verify configs with different runs are not equal."""
        config1 = Sufficiency.Config(training_strategy=mock_train, evaluation_strategy=mock_eval, runs=3)
        config2 = Sufficiency.Config(training_strategy=mock_train, evaluation_strategy=mock_eval, runs=5)

        assert config1 != config2

    def test_config_repr(self, mock_train, mock_eval):
        """Verify config has useful string representation."""
        config = Sufficiency.Config(training_strategy=mock_train, evaluation_strategy=mock_eval, runs=3, substeps=10)
        repr_str = repr(config)

        # Should contain class name and key parameters
        assert "Config" in repr_str
        assert "runs=3" in repr_str
        assert "substeps=10" in repr_str
