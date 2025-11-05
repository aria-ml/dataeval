"""
Tests for Sufficiency constructor during transition.

These tests verify that BOTH old and new constructor signatures work
during the compatibility period. The old signature will be removed in
a future step.
"""

from unittest.mock import MagicMock

import pytest
import torch.nn as nn

from dataeval.workflows.sufficiency import (
    Sufficiency,
    SufficiencyConfig,
    _FunctionEvaluationStrategy,
    _FunctionTrainingStrategy,
)


class TestSufficiencyOldConstructor:
    """Test that old constructor signature still works."""

    def test_old_signature_with_minimal_args(self, simple_dataset, mock_train_fn, mock_eval_fn):
        """Verify old constructor works with required args only."""
        model = nn.Linear(1, 1)

        suff = Sufficiency(
            model=model,
            train_ds=simple_dataset,
            test_ds=simple_dataset,
            train_fn=mock_train_fn,
            eval_fn=mock_eval_fn,
        )

        assert suff.model is model
        assert suff.train_ds is simple_dataset
        assert suff.test_ds is simple_dataset
        assert suff.runs == 1  # Default
        assert suff.substeps == 5  # Default

    def test_old_signature_with_all_args(self, mock_model, simple_dataset, mock_train_fn, mock_eval_fn):
        """Verify old constructor works with all arguments."""

        suff = Sufficiency(
            model=mock_model,
            train_ds=simple_dataset,
            test_ds=simple_dataset,
            train_fn=mock_train_fn,
            eval_fn=mock_eval_fn,
            runs=3,
            substeps=10,
            train_kwargs={"lr": 0.001},
            eval_kwargs={"batch_size": 32},
            unit_interval=False,
        )

        assert suff.runs == 3
        assert suff.substeps == 10
        assert not suff.unit_interval

    def test_old_signature_creates_internal_adapters(self, mock_model, simple_dataset, mock_train_fn, mock_eval_fn):
        """Verify old constructor creates internal strategy adapters."""

        suff = Sufficiency(
            model=mock_model,
            train_ds=simple_dataset,
            test_ds=simple_dataset,
            train_fn=mock_train_fn,
            eval_fn=mock_eval_fn,
        )

        # Should create internal config with adapter strategies
        assert hasattr(suff, "config")
        assert isinstance(suff.config, SufficiencyConfig)
        assert isinstance(suff.config.training_strategy, _FunctionTrainingStrategy)
        assert isinstance(suff.config.evaluation_strategy, _FunctionEvaluationStrategy)

    def test_old_signature_stores_kwargs_in_adapters(self, mock_model, simple_dataset, mock_train_fn, mock_eval_fn):
        """Verify old constructor passes kwargs to strategy adapters."""

        train_kwargs = {"lr": 0.001, "epochs": 10}
        eval_kwargs = {"batch_size": 32}

        suff = Sufficiency(
            model=mock_model,
            train_ds=simple_dataset,
            test_ds=simple_dataset,
            train_fn=mock_train_fn,
            eval_fn=mock_eval_fn,
            train_kwargs=train_kwargs,
            eval_kwargs=eval_kwargs,
        )

        # Adapters should store kwargs internally as defined by _Function*Strategy
        assert suff.config.training_strategy._kwargs == train_kwargs  # pyright: ignore[reportAttributeAccessIssue]
        assert suff.config.evaluation_strategy._kwargs == eval_kwargs  # pyright: ignore[reportAttributeAccessIssue]


class TestSufficiencyNewConstructor:
    """Test that new constructor signature works."""

    def test_new_signature_with_config(self, mock_model, simple_dataset, basic_config):
        """Verify new constructor works with SufficiencyConfig."""

        suff = Sufficiency(
            model=mock_model,
            train_ds=simple_dataset,
            test_ds=simple_dataset,
            config=basic_config,
        )

        assert suff.model is mock_model
        assert suff.train_ds is simple_dataset
        assert suff.test_ds is simple_dataset
        assert suff.config is basic_config
        assert suff.runs == 1
        assert suff.substeps == 5

    def test_new_signature_stores_config_reference(
        self, mock_model, simple_dataset, mock_training_strategy, mock_evaluation_strategy
    ):
        """Verify new constructor stores config object."""

        config = SufficiencyConfig(mock_training_strategy, mock_evaluation_strategy)

        suff = Sufficiency(mock_model, simple_dataset, simple_dataset, config)

        assert suff.config is config
        assert suff.config.training_strategy is mock_training_strategy
        assert suff.config.evaluation_strategy is mock_evaluation_strategy

    def test_new_signature_delegates_to_config(self, mock_model, simple_dataset, multi_run_config):
        """Verify new constructor delegates properties to config."""

        suff = Sufficiency(mock_model, simple_dataset, simple_dataset, multi_run_config)

        # Should delegate to config
        assert suff.runs == multi_run_config.runs
        assert suff.substeps == multi_run_config.substeps
        assert suff.unit_interval == multi_run_config.unit_interval


class TestConstructorDetection:
    """Test that constructor correctly detects which signature is used."""

    def test_detects_old_signature_by_train_fn(self, mock_model, simple_dataset, mock_train_fn, mock_eval_fn):
        """Verify constructor detects old signature when train_fn provided."""

        # Old signature has train_fn
        suff = Sufficiency(
            model=mock_model,
            train_ds=simple_dataset,
            test_ds=simple_dataset,
            train_fn=mock_train_fn,
            eval_fn=mock_eval_fn,
        )

        # Should work without error and create adapters
        assert suff.model is mock_model
        assert isinstance(suff.config.training_strategy, _FunctionTrainingStrategy)

    def test_detects_new_signature_by_config(self, mock_model, simple_dataset, basic_config):
        """Verify constructor detects new signature when config provided."""

        # New signature has config
        suff = Sufficiency(mock_model, simple_dataset, simple_dataset, basic_config)

        # Should work without error
        assert suff.model is mock_model
        assert suff.config is basic_config

    def test_raises_error_if_both_signatures_mixed(
        self, mock_model, simple_dataset, basic_config, mock_train_fn, mock_eval_fn
    ):
        """Verify constructor rejects mixed old and new signatures."""

        # Can't provide both config AND train_fn
        with pytest.raises(ValueError, match="Cannot provide both"):
            Sufficiency(
                model=mock_model,
                train_ds=simple_dataset,
                test_ds=simple_dataset,
                config=basic_config,
                train_fn=mock_train_fn,  # ← Conflict!
                eval_fn=mock_eval_fn,
            )

    def test_raises_error_if_neither_signature(self, mock_model, simple_dataset):
        """Verify constructor requires either old or new signature."""

        # Missing both config AND train_fn/eval_fn
        with pytest.raises(ValueError, match="Must provide either"):
            Sufficiency(mock_model, simple_dataset, simple_dataset)


class TestSufficiencyProperties:
    """Test that properties work with both constructor signatures."""

    def test_properties_work_with_old_signature(self, mock_model, simple_dataset, mock_train_fn, mock_eval_fn):
        """Verify properties accessible with old constructor."""

        suff = Sufficiency(
            model=mock_model,
            train_ds=simple_dataset,
            test_ds=simple_dataset,
            train_fn=mock_train_fn,
            eval_fn=mock_eval_fn,
            runs=3,
            substeps=7,
        )

        assert suff.runs == 3
        assert suff.substeps == 7
        assert suff.model is mock_model

    def test_properties_work_with_new_signature(self, mock_model, simple_dataset, multi_run_config):
        """Verify properties accessible with new constructor."""

        suff = Sufficiency(mock_model, simple_dataset, simple_dataset, multi_run_config)

        assert suff.runs == 3
        assert suff.substeps == 2
        assert suff.model is mock_model


class TestLegacyPropertyAccess:
    """Test that old property accessors still work with old API."""

    def test_train_fn_property_returns_original_function(self, mock_model, simple_dataset, mock_eval_fn):
        """Verify train_fn property unwraps adapter to return original function."""

        def my_train_fn(model, dataset, indices, **kwargs):
            pass

        suff = Sufficiency(mock_model, simple_dataset, simple_dataset, train_fn=my_train_fn, eval_fn=mock_eval_fn)

        # Should return the original function, not the adapter
        assert suff.train_fn is my_train_fn

    def test_eval_fn_property_returns_original_function(self, mock_model, simple_dataset, mock_train_fn):
        """Verify eval_fn property unwraps adapter to return original function."""

        def my_eval_fn(model, dataset, **kwargs):
            return {"accuracy": 0.95}

        suff = Sufficiency(mock_model, simple_dataset, simple_dataset, train_fn=mock_train_fn, eval_fn=my_eval_fn)

        # Should return the original function, not the adapter
        assert suff.eval_fn is my_eval_fn

    def test_train_kwargs_property_returns_kwargs(self, mock_model, simple_dataset, mock_train_fn, mock_eval_fn):
        """Verify train_kwargs property returns stored kwargs."""

        train_kwargs = {"lr": 0.001, "epochs": 10}

        suff = Sufficiency(
            mock_model,
            simple_dataset,
            simple_dataset,
            train_fn=mock_train_fn,
            eval_fn=mock_eval_fn,
            train_kwargs=train_kwargs,
        )

        assert suff.train_kwargs == train_kwargs

    def test_eval_kwargs_property_returns_kwargs(self, mock_model, simple_dataset, mock_train_fn, mock_eval_fn):
        """Verify eval_kwargs property returns stored kwargs."""

        eval_kwargs = {"batch_size": 32}

        suff = Sufficiency(
            mock_model,
            simple_dataset,
            simple_dataset,
            train_fn=mock_train_fn,
            eval_fn=mock_eval_fn,
            eval_kwargs=eval_kwargs,
        )

        assert suff.eval_kwargs == eval_kwargs

    def test_train_fn_raises_with_new_api(self, mock_model, simple_dataset, basic_config):
        """Verify train_fn property raises error with new API."""

        suff = Sufficiency(mock_model, simple_dataset, simple_dataset, basic_config)

        # Should raise because new API doesn't have train_fn
        with pytest.raises(AttributeError, match="train_fn.*only available.*legacy"):
            _ = suff.train_fn

    def test_eval_fn_raises_with_new_api(self, mock_model, simple_dataset, basic_config):
        """Verify eval_fn property raises error with new API."""

        suff = Sufficiency(mock_model, simple_dataset, simple_dataset, basic_config)

        # Should raise because new API doesn't have eval_fn
        with pytest.raises(AttributeError, match="eval_fn.*only available.*legacy"):
            _ = suff.eval_fn

    def test_kwargs_return_empty_with_new_api(self, mock_model, simple_dataset, basic_config):
        """Verify kwargs properties return empty dict with new API."""

        suff = Sufficiency(mock_model, simple_dataset, simple_dataset, basic_config)

        # Should return empty dicts since new API doesn't use kwargs
        assert suff.train_kwargs == {}
        assert suff.eval_kwargs == {}


class TestFunctionAdapters:
    """Test the internal adapter classes that wrap legacy functions."""

    def test_training_adapter_calls_function(self, mock_model, simple_dataset):
        """Verify _FunctionTrainingStrategy calls wrapped function."""

        mock_train_fn = MagicMock()

        # Create adapter with kwargs
        adapter = _FunctionTrainingStrategy(mock_train_fn, kwargs={"lr": 0.001, "epochs": 10})

        # Call through adapter
        indices = [0, 1, 2, 3, 4]

        adapter.train(mock_model, simple_dataset, indices)

        # Verify function was called with correct args
        mock_train_fn.assert_called_once_with(mock_model, simple_dataset, indices, lr=0.001, epochs=10)

    def test_training_adapter_handles_none_kwargs(self, mock_model, simple_dataset):
        """Verify adapter handles None kwargs correctly."""

        mock_train_fn = MagicMock()

        # Create adapter with no kwargs
        adapter = _FunctionTrainingStrategy(mock_train_fn, kwargs=None)

        indices = [0, 1, 2]

        adapter.train(mock_model, simple_dataset, indices)

        # Should call with no kwargs (empty dict unpacked)
        mock_train_fn.assert_called_once_with(mock_model, simple_dataset, indices)

    def test_evaluation_adapter_calls_function(self, mock_model, simple_dataset):
        """Verify _FunctionEvaluationStrategy calls wrapped function."""

        mock_eval_fn = MagicMock(return_value={"accuracy": 0.95})

        # Create adapter with kwargs
        adapter = _FunctionEvaluationStrategy(mock_eval_fn, kwargs={"batch_size": 32})

        # Call through adapter
        result = adapter.evaluate(mock_model, simple_dataset)

        # Verify function was called with correct args
        mock_eval_fn.assert_called_once_with(mock_model, simple_dataset, batch_size=32)
        assert result == {"accuracy": 0.95}

    def test_evaluation_adapter_handles_none_kwargs(self, mock_model, simple_dataset):
        """Verify adapter handles None kwargs correctly."""

        mock_eval_fn = MagicMock(return_value={"accuracy": 0.90})

        # Create adapter with no kwargs
        adapter = _FunctionEvaluationStrategy(mock_eval_fn, kwargs=None)

        adapter.evaluate(mock_model, simple_dataset)

        # Should call with no kwargs
        mock_eval_fn.assert_called_once_with(mock_model, simple_dataset)

    def test_evaluation_adapter_returns_function_result(self, mock_model, simple_dataset):
        """Verify adapter returns whatever the function returns."""

        expected_result = {"accuracy": 0.88, "f1": 0.85, "loss": 0.12}
        mock_eval_fn = MagicMock(return_value=expected_result)

        adapter = _FunctionEvaluationStrategy(mock_eval_fn)

        result = adapter.evaluate(mock_model, simple_dataset)

        assert result == expected_result
        mock_eval_fn.assert_called_once()
