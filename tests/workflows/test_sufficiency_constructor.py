"""
Tests for Sufficiency constructor during transition.

These tests verify that BOTH old and new constructor signatures work
during the compatibility period. The old signature will be removed in
a future step.
"""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from dataeval.workflows.sufficiency import (
    Sufficiency,
    SufficiencyConfig,
    _FunctionEvaluationStrategy,
    _FunctionTrainingStrategy,
)


class SimpleDataset(Dataset):
    """Mock dataset for testing."""

    def __init__(self, size=100):
        self.data = [(torch.randn(10), torch.randint(0, 2, (1,))) for _ in range(size)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SimpleModel(nn.Module):
    """Mock model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)


class MockTrainingStrategy:
    """Mock training strategy for testing."""

    def train(self, model, dataset, indices):
        # Dummy training
        pass


class MockEvaluationStrategy:
    """Mock evaluation strategy for testing."""

    def evaluate(self, model, dataset):
        return {"accuracy": 0.95}


def mock_train_fn(model, dataset, indices, **kwargs):
    """Mock training function (old API)."""
    pass


def mock_eval_fn(model, dataset, **kwargs):
    """Mock evaluation function (old API)."""
    return {"accuracy": 0.95}


class TestSufficiencyOldConstructor:
    """Test that old constructor signature still works."""

    def test_old_signature_with_minimal_args(self):
        """Verify old constructor works with required args only."""
        model = SimpleModel()
        train_ds = SimpleDataset(100)
        test_ds = SimpleDataset(50)

        suff = Sufficiency(
            model=model,
            train_ds=train_ds,
            test_ds=test_ds,
            train_fn=mock_train_fn,
            eval_fn=mock_eval_fn,
        )

        assert suff.model is model
        assert suff.train_ds is train_ds
        assert suff.test_ds is test_ds
        assert suff.runs == 1  # Default
        assert suff.substeps == 5  # Default

    def test_old_signature_with_all_args(self):
        """Verify old constructor works with all arguments."""
        model = SimpleModel()
        train_ds = SimpleDataset(100)
        test_ds = SimpleDataset(50)

        suff = Sufficiency(
            model=model,
            train_ds=train_ds,
            test_ds=test_ds,
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

    def test_old_signature_stores_train_fn(self):
        """Verify old constructor stores training function."""
        model = SimpleModel()
        train_ds = SimpleDataset(100)
        test_ds = SimpleDataset(50)

        suff = Sufficiency(
            model=model,
            train_ds=train_ds,
            test_ds=test_ds,
            train_fn=mock_train_fn,
            eval_fn=mock_eval_fn,
        )

        # Should create internal strategy wrapper
        assert hasattr(suff, "config")
        assert suff.config is not None

    def test_old_signature_stores_kwargs(self):
        """Verify old constructor handles kwargs."""
        model = SimpleModel()
        train_ds = SimpleDataset(100)
        test_ds = SimpleDataset(50)

        train_kwargs = {"lr": 0.001, "epochs": 10}
        eval_kwargs = {"batch_size": 32}

        suff = Sufficiency(
            model=model,
            train_ds=train_ds,
            test_ds=test_ds,
            train_fn=mock_train_fn,
            eval_fn=mock_eval_fn,
            train_kwargs=train_kwargs,
            eval_kwargs=eval_kwargs,
        )

        # Should have config internally
        assert suff.config is not None


class TestSufficiencyNewConstructor:
    """Test that new constructor signature works."""

    def test_new_signature_with_config(self):
        """Verify new constructor works with SufficiencyConfig."""
        model = SimpleModel()
        train_ds = SimpleDataset(100)
        test_ds = SimpleDataset(50)

        training = MockTrainingStrategy()
        evaluation = MockEvaluationStrategy()
        config = SufficiencyConfig(training, evaluation, runs=3, substeps=5)

        suff = Sufficiency(
            model=model,
            train_ds=train_ds,
            test_ds=test_ds,
            config=config,
        )

        assert suff.model is model
        assert suff.train_ds is train_ds
        assert suff.test_ds is test_ds
        assert suff.config is config
        assert suff.runs == 3
        assert suff.substeps == 5

    def test_new_signature_stores_config_reference(self):
        """Verify new constructor stores config object."""
        model = SimpleModel()
        train_ds = SimpleDataset(100)
        test_ds = SimpleDataset(50)

        training = MockTrainingStrategy()
        evaluation = MockEvaluationStrategy()
        config = SufficiencyConfig(training, evaluation)

        suff = Sufficiency(model, train_ds, test_ds, config)

        assert suff.config is config
        assert suff.config.training_strategy is training
        assert suff.config.evaluation_strategy is evaluation

    def test_new_signature_delegates_to_config(self):
        """Verify new constructor delegates properties to config."""
        model = SimpleModel()
        train_ds = SimpleDataset(100)
        test_ds = SimpleDataset(50)

        training = MockTrainingStrategy()
        evaluation = MockEvaluationStrategy()
        config = SufficiencyConfig(training, evaluation, runs=5, substeps=10)

        suff = Sufficiency(model, train_ds, test_ds, config)

        # Should delegate to config
        assert suff.runs == config.runs
        assert suff.substeps == config.substeps
        assert suff.unit_interval == config.unit_interval


class TestConstructorDetection:
    """Test that constructor correctly detects which signature is used."""

    def test_detects_old_signature_by_train_fn(self):
        """Verify constructor detects old signature when train_fn provided."""
        model = SimpleModel()
        train_ds = SimpleDataset(100)
        test_ds = SimpleDataset(50)

        # Old signature has train_fn
        suff = Sufficiency(
            model=model,
            train_ds=train_ds,
            test_ds=test_ds,
            train_fn=mock_train_fn,
            eval_fn=mock_eval_fn,
        )

        # Should work without error
        assert suff.model is model

    def test_detects_new_signature_by_config(self):
        """Verify constructor detects new signature when config provided."""
        model = SimpleModel()
        train_ds = SimpleDataset(100)
        test_ds = SimpleDataset(50)

        training = MockTrainingStrategy()
        evaluation = MockEvaluationStrategy()
        config = SufficiencyConfig(training, evaluation)

        # New signature has config
        suff = Sufficiency(model, train_ds, test_ds, config)

        # Should work without error
        assert suff.model is model
        assert suff.config is config

    def test_raises_error_if_both_signatures_mixed(self):
        """Verify constructor rejects mixed old and new signatures."""
        model = SimpleModel()
        train_ds = SimpleDataset(100)
        test_ds = SimpleDataset(50)

        training = MockTrainingStrategy()
        evaluation = MockEvaluationStrategy()
        config = SufficiencyConfig(training, evaluation)

        # Can't provide both config AND train_fn
        with pytest.raises(ValueError, match="Cannot provide both"):
            Sufficiency(
                model=model,
                train_ds=train_ds,
                test_ds=test_ds,
                config=config,
                train_fn=mock_train_fn,  # ← Conflict!
                eval_fn=mock_eval_fn,
            )

    def test_raises_error_if_neither_signature(self):
        """Verify constructor requires either old or new signature."""
        model = SimpleModel()
        train_ds = SimpleDataset(100)
        test_ds = SimpleDataset(50)

        # Missing both config AND train_fn/eval_fn
        with pytest.raises(ValueError, match="Must provide either"):
            Sufficiency(model, train_ds, test_ds)


class TestSufficiencyProperties:
    """Test that properties work with both constructor signatures."""

    def test_properties_work_with_old_signature(self):
        """Verify properties accessible with old constructor."""
        model = SimpleModel()
        train_ds = SimpleDataset(100)
        test_ds = SimpleDataset(50)

        suff = Sufficiency(
            model=model,
            train_ds=train_ds,
            test_ds=test_ds,
            train_fn=mock_train_fn,
            eval_fn=mock_eval_fn,
            runs=3,
            substeps=7,
        )

        assert suff.runs == 3
        assert suff.substeps == 7
        assert suff.model is model

    def test_properties_work_with_new_signature(self):
        """Verify properties accessible with new constructor."""
        model = SimpleModel()
        train_ds = SimpleDataset(100)
        test_ds = SimpleDataset(50)

        training = MockTrainingStrategy()
        evaluation = MockEvaluationStrategy()
        config = SufficiencyConfig(training, evaluation, runs=5, substeps=10)

        suff = Sufficiency(model, train_ds, test_ds, config)

        assert suff.runs == 5
        assert suff.substeps == 10
        assert suff.model is model


class TestLegacyPropertyAccess:
    """Test that old property accessors still work with old API."""

    def test_train_fn_property_returns_original_function(self):
        """Verify train_fn property unwraps adapter to return original function."""
        model = SimpleModel()
        train_ds = SimpleDataset(100)
        test_ds = SimpleDataset(50)

        def my_train_fn(model, dataset, indices, **kwargs):
            pass

        suff = Sufficiency(model, train_ds, test_ds, train_fn=my_train_fn, eval_fn=mock_eval_fn)

        # Should return the original function, not the adapter
        assert suff.train_fn is my_train_fn

    def test_eval_fn_property_returns_original_function(self):
        """Verify eval_fn property unwraps adapter to return original function."""
        model = SimpleModel()
        train_ds = SimpleDataset(100)
        test_ds = SimpleDataset(50)

        def my_eval_fn(model, dataset, **kwargs):
            return {"accuracy": 0.95}

        suff = Sufficiency(model, train_ds, test_ds, train_fn=mock_train_fn, eval_fn=my_eval_fn)

        # Should return the original function, not the adapter
        assert suff.eval_fn is my_eval_fn

    def test_train_kwargs_property_returns_kwargs(self):
        """Verify train_kwargs property returns stored kwargs."""
        model = SimpleModel()
        train_ds = SimpleDataset(100)
        test_ds = SimpleDataset(50)

        train_kwargs = {"lr": 0.001, "epochs": 10}

        suff = Sufficiency(
            model, train_ds, test_ds, train_fn=mock_train_fn, eval_fn=mock_eval_fn, train_kwargs=train_kwargs
        )

        assert suff.train_kwargs == train_kwargs

    def test_eval_kwargs_property_returns_kwargs(self):
        """Verify eval_kwargs property returns stored kwargs."""
        model = SimpleModel()
        train_ds = SimpleDataset(100)
        test_ds = SimpleDataset(50)

        eval_kwargs = {"batch_size": 32}

        suff = Sufficiency(
            model, train_ds, test_ds, train_fn=mock_train_fn, eval_fn=mock_eval_fn, eval_kwargs=eval_kwargs
        )

        assert suff.eval_kwargs == eval_kwargs

    def test_train_fn_raises_with_new_api(self):
        """Verify train_fn property raises error with new API."""
        model = SimpleModel()
        train_ds = SimpleDataset(100)
        test_ds = SimpleDataset(50)

        training = MockTrainingStrategy()
        evaluation = MockEvaluationStrategy()
        config = SufficiencyConfig(training, evaluation)

        suff = Sufficiency(model, train_ds, test_ds, config)

        # Should raise because new API doesn't have train_fn
        with pytest.raises(AttributeError, match="train_fn.*only available.*legacy"):
            _ = suff.train_fn

    def test_eval_fn_raises_with_new_api(self):
        """Verify eval_fn property raises error with new API."""
        model = SimpleModel()
        train_ds = SimpleDataset(100)
        test_ds = SimpleDataset(50)

        training = MockTrainingStrategy()
        evaluation = MockEvaluationStrategy()
        config = SufficiencyConfig(training, evaluation)

        suff = Sufficiency(model, train_ds, test_ds, config)

        # Should raise because new API doesn't have eval_fn
        with pytest.raises(AttributeError, match="eval_fn.*only available.*legacy"):
            _ = suff.eval_fn

    def test_kwargs_return_empty_with_new_api(self):
        """Verify kwargs properties return empty dict with new API."""
        model = SimpleModel()
        train_ds = SimpleDataset(100)
        test_ds = SimpleDataset(50)

        training = MockTrainingStrategy()
        evaluation = MockEvaluationStrategy()
        config = SufficiencyConfig(training, evaluation)

        suff = Sufficiency(model, train_ds, test_ds, config)

        # Should return empty dicts since new API doesn't use kwargs
        assert suff.train_kwargs == {}
        assert suff.eval_kwargs == {}


class TestFunctionAdapters:
    """Test the internal adapter classes that wrap legacy functions."""

    def test_training_adapter_calls_function(self):
        """Verify _FunctionTrainingStrategy calls wrapped function."""

        mock_train_fn = MagicMock()

        # Create adapter with kwargs
        adapter = _FunctionTrainingStrategy(mock_train_fn, kwargs={"lr": 0.001, "epochs": 10})

        # Call through adapter
        model = SimpleModel()
        dataset = SimpleDataset(100)
        indices = [0, 1, 2, 3, 4]

        adapter.train(model, dataset, indices)

        # Verify function was called with correct args
        mock_train_fn.assert_called_once_with(model, dataset, indices, lr=0.001, epochs=10)

    def test_training_adapter_handles_none_kwargs(self):
        """Verify adapter handles None kwargs correctly."""

        mock_train_fn = MagicMock()

        # Create adapter with no kwargs
        adapter = _FunctionTrainingStrategy(mock_train_fn, kwargs=None)

        model = SimpleModel()
        dataset = SimpleDataset(100)
        indices = [0, 1, 2]

        adapter.train(model, dataset, indices)

        # Should call with no kwargs (empty dict unpacked)
        mock_train_fn.assert_called_once_with(model, dataset, indices)

    def test_evaluation_adapter_calls_function(self):
        """Verify _FunctionEvaluationStrategy calls wrapped function."""

        mock_eval_fn = MagicMock(return_value={"accuracy": 0.95})

        # Create adapter with kwargs
        adapter = _FunctionEvaluationStrategy(mock_eval_fn, kwargs={"batch_size": 32})

        # Call through adapter
        model = SimpleModel()
        dataset = SimpleDataset(50)

        result = adapter.evaluate(model, dataset)

        # Verify function was called with correct args
        mock_eval_fn.assert_called_once_with(model, dataset, batch_size=32)
        assert result == {"accuracy": 0.95}

    def test_evaluation_adapter_handles_none_kwargs(self):
        """Verify adapter handles None kwargs correctly."""

        mock_eval_fn = MagicMock(return_value={"accuracy": 0.90})

        # Create adapter with no kwargs
        adapter = _FunctionEvaluationStrategy(mock_eval_fn, kwargs=None)

        model = SimpleModel()
        dataset = SimpleDataset(50)

        adapter.evaluate(model, dataset)

        # Should call with no kwargs
        mock_eval_fn.assert_called_once_with(model, dataset)

    def test_evaluation_adapter_returns_function_result(self):
        """Verify adapter returns whatever the function returns."""

        expected_result = {"accuracy": 0.88, "f1": 0.85, "loss": 0.12}
        mock_eval_fn = MagicMock(return_value=expected_result)

        adapter = _FunctionEvaluationStrategy(mock_eval_fn)

        result = adapter.evaluate(SimpleModel(), SimpleDataset(50))

        assert result == expected_result
        mock_eval_fn.assert_called_once()
