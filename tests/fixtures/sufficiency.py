"""
Pytest fixtures for Sufficiency workflow tests.

These fixtures provide generic mocks and test data for testing
the Sufficiency orchestration workflow, independent of specific
ML task types (IC, OD, etc.).
"""

from unittest.mock import Mock

import pytest

from dataeval.workflows.sufficiency import EvaluationStrategy, SufficiencyConfig, TrainingStrategy

# ========== STRATEGY FIXTURES ==========


@pytest.fixture(scope="function")
def mock_training_strategy() -> TrainingStrategy:
    """
    Mock training strategy for testing.

    Returns a Mock conforming to TrainingStrategy protocol.
    """

    strategy = Mock(spec=TrainingStrategy)
    strategy.train = Mock(return_value=None)
    return strategy


@pytest.fixture(scope="function")
def mock_evaluation_strategy() -> EvaluationStrategy:
    """
    Mock evaluation strategy for testing.

    Returns a Mock conforming to EvaluationStrategy protocol.
    """

    strategy = Mock(spec=EvaluationStrategy)
    strategy.evaluate = Mock(return_value={"accuracy": 0.95})
    return strategy


# ========== CONFIG FIXTURES ==========


@pytest.fixture(scope="function")
def basic_config(mock_training_strategy, mock_evaluation_strategy) -> SufficiencyConfig:
    """
    Basic SufficiencyConfig with default parameters.

    Uses runs=1, substeps=5 (defaults).
    """

    return SufficiencyConfig(
        training_strategy=mock_training_strategy,
        evaluation_strategy=mock_evaluation_strategy,
        runs=1,
        substeps=5,
    )


@pytest.fixture(scope="function")
def multi_run_config(mock_training_strategy, mock_evaluation_strategy) -> SufficiencyConfig:
    """
    Config for multiple runs (faster testing).

    Uses runs=3, substeps=2.
    """

    return SufficiencyConfig(
        training_strategy=mock_training_strategy,
        evaluation_strategy=mock_evaluation_strategy,
        runs=3,
        substeps=2,
    )


# ========== LEGACY API FIXTURES ==========


@pytest.fixture(scope="function")
def mock_train_fn():
    """Mock training function for legacy API tests."""
    return Mock(return_value=None)


@pytest.fixture(scope="function")
def mock_eval_fn():
    """Mock evaluation function for legacy API tests."""
    return Mock(return_value={"accuracy": 0.95})
