"""
Tests for Sufficiency helper methods.

These tests verify internal helper methods that support the
evaluate() orchestration.
"""

import numpy as np
import torch.nn as nn

from dataeval.performance._aggregator import ResultAggregator
from dataeval.performance._sufficiency import Sufficiency
from dataeval.performance.schedules import GeometricSchedule, ManualSchedule


class TestCreateScheduleHelper:
    """Test _create_schedule helper method."""

    def test_creates_geometric_schedule_when_eval_at_none(self, mock_model, simple_dataset, basic_config):
        """Verify default geometric schedule when eval_at is None."""
        suff = Sufficiency(mock_model, simple_dataset, simple_dataset, basic_config)

        schedule = suff._create_schedule(schedule=None)

        # Should return GeometricSchedule
        assert isinstance(schedule, GeometricSchedule)
        assert schedule.substeps == basic_config.substeps

    def test_creates_custom_schedule_when_eval_at_int(self, mock_model, simple_dataset, basic_config):
        """Verify custom schedule when eval_at is single int."""
        suff = Sufficiency(mock_model, simple_dataset, simple_dataset, basic_config)

        schedule = suff._create_schedule(schedule=50)

        # Should return CustomSchedule
        assert isinstance(schedule, ManualSchedule)

    def test_creates_custom_schedule_when_eval_at_list(self, mock_model, simple_dataset, basic_config):
        """Verify custom schedule when eval_at is list."""
        suff = Sufficiency(mock_model, simple_dataset, simple_dataset, basic_config)

        schedule = suff._create_schedule(schedule=[10, 20, 30])

        # Should return CustomSchedule
        assert isinstance(schedule, ManualSchedule)


class TestExecuteRunHelper:
    """Test _execute_run helper method."""

    def test_executes_single_run(self, mock_train, mock_eval, simple_dataset):
        """Verify _execute_run trains and evaluates for all steps."""
        mock_eval.evaluate.return_value = {"accuracy": 0.95}

        config = Sufficiency.Config(training_strategy=mock_train, evaluation_strategy=mock_eval, runs=1, substeps=3)
        model = nn.Linear(10, 2)
        suff = Sufficiency(model, simple_dataset, simple_dataset, config=config)

        # Create aggregator and steps
        aggregator = ResultAggregator(runs=1, substeps=3)
        steps = np.array([10, 20, 30], dtype=np.uint32)

        # Execute one run
        suff._execute_run(run_index=0, steps=steps, aggregator=aggregator)

        # Should have called training and evaluation for each step
        assert mock_train.train.call_count == 3
        assert mock_eval.evaluate.call_count == 3

    def test_passes_correct_indices_to_training(self, mock_train, mock_eval, simple_dataset):
        """Verify _execute_run passes correct indices to training."""
        mock_eval.evaluate.return_value = {"accuracy": 0.95}

        config = Sufficiency.Config(training_strategy=mock_train, evaluation_strategy=mock_eval, runs=1, substeps=2)
        model = nn.Linear(10, 2)
        suff = Sufficiency(model, simple_dataset, simple_dataset, config=config)

        aggregator = ResultAggregator(runs=1, substeps=2)
        steps = np.array([5, 10], dtype=np.uint32)

        suff._execute_run(0, steps, aggregator)

        # Check first call - should train on first 5 indices
        first_call_indices = mock_train.train.call_args_list[0][0][2]
        assert len(first_call_indices) == 5

        # Check second call - should train on first 10 indices
        second_call_indices = mock_train.train.call_args_list[1][0][2]
        assert len(second_call_indices) == 10

    def test_passes_model_to_training(self, mock_train, mock_eval, simple_dataset):
        """Verify _execute_run passes model to training strategy."""
        mock_eval.evaluate.return_value = {"accuracy": 0.95}

        config = Sufficiency.Config(training_strategy=mock_train, evaluation_strategy=mock_eval, runs=1, substeps=1)
        model = nn.Linear(10, 2)
        suff = Sufficiency(model, simple_dataset, simple_dataset, config=config)

        aggregator = ResultAggregator(runs=1, substeps=1)
        steps = np.array([10], dtype=np.uint32)

        suff._execute_run(0, steps, aggregator)

        # Model should be passed to training
        call_args = mock_train.train.call_args[0]
        passed_model = call_args[0]
        assert isinstance(passed_model, nn.Module)

    def test_passes_test_dataset_to_evaluation(self, mock_train, mock_eval, simple_dataset):
        """Verify _execute_run passes test dataset to evaluation strategy."""
        mock_eval.evaluate.return_value = {"accuracy": 0.95}

        config = Sufficiency.Config(training_strategy=mock_train, evaluation_strategy=mock_eval, runs=1, substeps=1)
        model = nn.Linear(10, 2)
        test_ds = simple_dataset
        suff = Sufficiency(model, simple_dataset, test_ds, config=config)

        aggregator = ResultAggregator(runs=1, substeps=1)
        steps = np.array([10], dtype=np.uint32)

        suff._execute_run(0, steps, aggregator)

        # Test dataset should be passed to evaluation
        call_args = mock_eval.evaluate.call_args[0]
        passed_dataset = call_args[1]
        assert passed_dataset is test_ds

    def test_stores_results_in_aggregator(self, mock_train, mock_eval, simple_dataset):
        """Verify _execute_run stores results in aggregator."""
        mock_eval.evaluate.return_value = {"accuracy": 0.95, "loss": 0.05}

        config = Sufficiency.Config(training_strategy=mock_train, evaluation_strategy=mock_eval, runs=1, substeps=2)
        model = nn.Linear(10, 2)
        suff = Sufficiency(model, simple_dataset, simple_dataset, config=config)

        aggregator = ResultAggregator(runs=1, substeps=2)
        steps = np.array([5, 10], dtype=np.uint32)

        suff._execute_run(0, steps, aggregator)

        # Results should be stored in aggregator
        results = aggregator.get_results()
        assert "accuracy" in results
        assert "loss" in results
        assert results["accuracy"][0, 0] == 0.95
        assert results["loss"][0, 0] == 0.05
