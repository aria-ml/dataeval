"""
Pytest fixtures for Sufficiency workflow tests.

These fixtures provide generic mocks and test data for testing
the Sufficiency orchestration workflow, independent of specific
ML task types (IC, OD, etc.).
"""

from typing import Any
from unittest.mock import MagicMock, NonCallableMagicMock

import numpy as np
import pytest
from numpy.typing import NDArray

from dataeval.performance import Sufficiency
from dataeval.protocols import EvaluationStrategy, TrainingStrategy

# ========== TYPING ALIASES ==========

DatumType = tuple[NDArray[np.float32], int, dict[str, Any]]


# ========== RESET STRATEGY FIXTURES ==========


def _mock_reset(model: Any) -> Any:
    """Mock reset strategy that returns the model unchanged."""
    return model


@pytest.fixture(scope="function")
def mock_reset():
    """
    Mock reset strategy for testing with non-PyTorch models.

    Returns a callable that simply returns the model unchanged.
    """
    return _mock_reset


# ========== STRATEGY FIXTURES ==========


@pytest.fixture(scope="function")
def mock_train() -> TrainingStrategy[DatumType]:
    """
    Mock training strategy for testing.

    Returns a Mock conforming to TrainingStrategy protocol.
    """

    strategy = MagicMock(spec=TrainingStrategy)
    strategy.train = MagicMock(return_value=None)
    return strategy


@pytest.fixture(scope="function")
def mock_eval() -> EvaluationStrategy[DatumType]:
    """
    Mock evaluation strategy for testing.

    Returns a Mock conforming to EvaluationStrategy protocol.
    """

    strategy = MagicMock(spec=EvaluationStrategy)
    strategy.evaluate = MagicMock(return_value={"accuracy": 0.95})
    return strategy


@pytest.fixture(scope="function")
def mock_eval_mixed_metric_strategy() -> EvaluationStrategy[DatumType]:
    """Mock evaluation strategy with multiple metrics including a per-class metric"""

    eval_strategy = MagicMock(spec=EvaluationStrategy)
    eval_strategy.evaluate = MagicMock(return_value={"Accuracy": 0.95, "Precision": [1.0, 2.0]})
    return eval_strategy


@pytest.fixture(scope="function")
def mock_eval_scalar_metrics_strategy() -> EvaluationStrategy[DatumType]:
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
def mock_eval_classwise() -> EvaluationStrategy[DatumType]:
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
def basic_config(mock_train, mock_eval, mock_reset) -> Sufficiency.Config[DatumType, Any]:
    """
    Basic Sufficiency.Config with default parameters.

    Uses runs=1, substeps=5 (defaults).
    Includes a mock reset_strategy for use with non-PyTorch mock models.
    """

    return Sufficiency.Config(
        training_strategy=mock_train,
        evaluation_strategy=mock_eval,
        reset_strategy=mock_reset,
        runs=1,
        substeps=5,
    )


@pytest.fixture(scope="function")
def multi_run_config(mock_train, mock_eval, mock_reset) -> Sufficiency.Config[DatumType, Any]:
    """
    Config for multiple runs (faster testing).

    Uses runs=3, substeps=2.
    Includes a mock reset_strategy for use with non-PyTorch mock models.
    """

    return Sufficiency.Config(
        training_strategy=mock_train,
        evaluation_strategy=mock_eval,
        reset_strategy=mock_reset,
        runs=3,
        substeps=2,
    )


@pytest.fixture(scope="function")
def non_callable_sufficiency_config(mock_reset) -> Sufficiency.Config[DatumType, Any]:
    return Sufficiency.Config(
        training_strategy=NonCallableMagicMock(),
        evaluation_strategy=NonCallableMagicMock(),
        reset_strategy=mock_reset,
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
