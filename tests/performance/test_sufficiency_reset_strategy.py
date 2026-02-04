"""
Tests for Sufficiency reset strategy functionality.

These tests verify the backend-agnostic reset strategy feature
that allows users to provide custom model reset logic for
non-PyTorch models (ONNX, TensorFlow, etc.).
"""

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch.nn as nn

from dataeval.performance import Sufficiency
from tests.conftest import SimpleDataset


class TestSufficiencyResetStrategy:
    """Test Sufficiency class with custom reset strategies."""

    def test_uses_custom_reset_strategy(self, mock_train, mock_eval, simple_dataset: SimpleDataset):
        """Verify custom reset strategy is used when provided."""

        def custom_reset(model: Any) -> Any:
            return model

        model = nn.Linear(10, 5)

        suff = Sufficiency(
            model=model,
            train_ds=simple_dataset,
            test_ds=simple_dataset,
            training_strategy=mock_train,
            evaluation_strategy=mock_eval,
            reset_strategy=custom_reset,
        )

        assert suff.reset_strategy is custom_reset

    def test_raises_error_for_non_torch_model_without_reset_strategy(
        self, mock_train, mock_eval, simple_dataset: SimpleDataset
    ):
        """Verify error is raised for non-PyTorch model without reset_strategy."""
        non_torch_model = MagicMock()  # Not an nn.Module

        with pytest.raises(ValueError, match="reset_strategy is required"):
            Sufficiency(
                model=non_torch_model,
                train_ds=simple_dataset,
                test_ds=simple_dataset,
                training_strategy=mock_train,
                evaluation_strategy=mock_eval,
            )

    def test_accepts_non_torch_model_with_reset_strategy(self, mock_train, mock_eval, simple_dataset: SimpleDataset):
        """Verify non-PyTorch model is accepted with custom reset_strategy."""

        def custom_reset(model: Any) -> Any:
            return model

        non_torch_model = MagicMock()

        # Should not raise
        suff = Sufficiency(
            model=non_torch_model,
            train_ds=simple_dataset,
            test_ds=simple_dataset,
            training_strategy=mock_train,
            evaluation_strategy=mock_eval,
            reset_strategy=custom_reset,
        )

        assert suff.model is non_torch_model
        assert suff.reset_strategy is custom_reset

    def test_reset_strategy_called_during_execute_run(self, mock_train, mock_eval, simple_dataset: SimpleDataset):
        """Verify reset_strategy is called during _execute_run."""
        mock_eval.evaluate.return_value = {"accuracy": 0.95}
        reset_mock = MagicMock(return_value=nn.Linear(10, 5))

        model = nn.Linear(10, 5)
        config = Sufficiency.Config(
            training_strategy=mock_train,
            evaluation_strategy=mock_eval,
            runs=1,
            substeps=2,
        )

        suff = Sufficiency(
            model=model,
            train_ds=simple_dataset,
            test_ds=simple_dataset,
            reset_strategy=reset_mock,
            config=config,
        )

        suff.evaluate()

        # Reset should have been called once per run
        assert reset_mock.call_count == 1

    def test_reset_strategy_called_per_run(self, mock_train, mock_eval, simple_dataset: SimpleDataset):
        """Verify reset_strategy is called once per run."""
        mock_eval.evaluate.return_value = {"accuracy": 0.95}
        reset_mock = MagicMock(return_value=nn.Linear(10, 5))

        model = nn.Linear(10, 5)
        config = Sufficiency.Config(
            training_strategy=mock_train,
            evaluation_strategy=mock_eval,
            runs=3,  # Multiple runs
            substeps=2,
        )

        suff = Sufficiency(
            model=model,
            train_ds=simple_dataset,
            test_ds=simple_dataset,
            reset_strategy=reset_mock,
            config=config,
        )

        suff.evaluate()

        # Reset should have been called once per run
        assert reset_mock.call_count == 3


class TestSufficiencyConfigResetStrategy:
    """Test Sufficiency.Config with reset_strategy."""

    def test_config_accepts_reset_strategy(self, mock_train, mock_eval):
        """Verify Config accepts reset_strategy parameter."""

        def custom_reset(model: Any) -> Any:
            return model

        config = Sufficiency.Config(
            training_strategy=mock_train,
            evaluation_strategy=mock_eval,
            reset_strategy=custom_reset,
        )

        assert config.reset_strategy is custom_reset

    def test_config_reset_strategy_defaults_to_none(self, mock_train, mock_eval):
        """Verify Config reset_strategy defaults to None."""
        config = Sufficiency.Config(
            training_strategy=mock_train,
            evaluation_strategy=mock_eval,
        )

        assert config.reset_strategy is None

    def test_sufficiency_uses_config_reset_strategy(self, mock_train, mock_eval, simple_dataset: SimpleDataset):
        """Verify Sufficiency uses reset_strategy from config."""

        def custom_reset(model: Any) -> Any:
            return model

        config = Sufficiency.Config(
            training_strategy=mock_train,
            evaluation_strategy=mock_eval,
            reset_strategy=custom_reset,
        )

        model = nn.Linear(10, 5)
        suff = Sufficiency(
            model=model,
            train_ds=simple_dataset,
            test_ds=simple_dataset,
            config=config,
        )

        assert suff.reset_strategy is custom_reset

    def test_direct_param_overrides_config_reset_strategy(self, mock_train, mock_eval, simple_dataset: SimpleDataset):
        """Verify direct reset_strategy param overrides config."""

        def config_reset(model: Any) -> Any:
            return model

        def direct_reset(model: Any) -> Any:
            return model

        config = Sufficiency.Config(
            training_strategy=mock_train,
            evaluation_strategy=mock_eval,
            reset_strategy=config_reset,
        )

        model = nn.Linear(10, 5)
        suff = Sufficiency(
            model=model,
            train_ds=simple_dataset,
            test_ds=simple_dataset,
            reset_strategy=direct_reset,  # Direct param should win
            config=config,
        )

        assert suff.reset_strategy is direct_reset
