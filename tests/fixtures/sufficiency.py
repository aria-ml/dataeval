"""
Pytest fixtures for Sufficiency workflow tests.

These fixtures provide generic mocks and test data for testing
the Sufficiency orchestration workflow, independent of specific
ML task types (IC, OD, etc.).
"""

from unittest.mock import MagicMock, NonCallableMagicMock

import pytest

from dataeval.performance._sufficiency import EvaluationStrategy, SufficiencyConfig, TrainingStrategy

# ========== STRATEGY FIXTURES ==========


@pytest.fixture(scope="function")
def mock_training_strategy() -> TrainingStrategy:
    """
    Mock training strategy for testing.

    Returns a Mock conforming to TrainingStrategy protocol.
    """

    strategy = MagicMock(spec=TrainingStrategy)
    strategy.train = MagicMock(return_value=None)
    return strategy


@pytest.fixture(scope="function")
def mock_evaluation_strategy() -> EvaluationStrategy:
    """
    Mock evaluation strategy for testing.

    Returns a Mock conforming to EvaluationStrategy protocol.
    """

    strategy = MagicMock(spec=EvaluationStrategy)
    strategy.evaluate = MagicMock(return_value={"accuracy": 0.95})
    return strategy


@pytest.fixture(scope="function")
def mock_eval_mixed_metric_strategy() -> EvaluationStrategy:
    """Mock evaluation strategy with multiple metrics including a per-class metric"""

    eval_strategy = MagicMock(spec=EvaluationStrategy)
    eval_strategy.evaluate = MagicMock(return_value={"Accuracy": 0.95, "Precision": [1.0, 2.0]})
    return eval_strategy


@pytest.fixture(scope="function")
def mock_eval_scalar_metrics_strategy() -> EvaluationStrategy:
    """
    Mock evaluation strategy returning multiple scalar metrics only.

    Returns
    -------
    - Accuracy: scalar (1.0)
    - Precision: scalar (1.0)
    """
    strategy = MagicMock(spec=EvaluationStrategy)
    strategy.evaluate = MagicMock(return_value={"Accuracy": 1.0, "Precision": 1.0})
    return strategy


@pytest.fixture(scope="function")
def mock_eval_classwise_strategy() -> EvaluationStrategy:
    """
    Mock evaluation strategy retuning a single classwise metric

    Returns
    -------
    - Accuracy: array ([0.2, 0.4, 0.6, 0.8]) - 4 classes"""

    strategy = MagicMock(spec=EvaluationStrategy)
    strategy.evaluate = MagicMock(return_value={"Accuracy": [0.2, 0.4, 0.6, 0.8]})
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


@pytest.fixture(scope="function")
def non_callable_sufficiency_config() -> SufficiencyConfig:
    return SufficiencyConfig(
        NonCallableMagicMock(),
        NonCallableMagicMock(),
    )


# ========== LEGACY API FIXTURES ==========


@pytest.fixture(scope="function")
def mock_train_fn():
    """Mock training function for legacy API tests."""
    return MagicMock(return_value=None)


@pytest.fixture(scope="function")
def mock_eval_fn():
    """Mock evaluation function for legacy API tests."""
    return MagicMock(return_value={"accuracy": 0.95})
