"""Tests for evaluate() method using strategy pattern."""

import torch.nn as nn

from dataeval.performance import Sufficiency


class TestEvaluateUsesStrategies:
    """Test that evaluate() delegates to strategies correctly."""

    def test_evaluate_calls_training_strategy(self, basic_config, simple_dataset, mock_model):
        """Verify evaluate calls training_strategy.train()."""
        suff = Sufficiency(
            model=mock_model,
            train_ds=simple_dataset,
            test_ds=simple_dataset,
            config=basic_config,
        )

        suff.evaluate()

        # Should call training for each substep (at least 2 times)
        assert basic_config.training_strategy.train.call_count >= 2

    def test_evaluate_calls_evaluation_strategy(self, basic_config, simple_dataset, mock_model):
        """Verify evaluate calls evaluation_strategy.evaluate()."""
        suff = Sufficiency(
            model=mock_model,
            train_ds=simple_dataset,
            test_ds=simple_dataset,
            config=basic_config,
        )

        suff.evaluate()

        # Should call evaluation for each substep (at least 2 times)
        assert basic_config.evaluation_strategy.evaluate.call_count >= 2

    # @patch("dataeval.performance._sufficiency.reset_parameters")
    def test_evaluate_passes_model_to_training(self, basic_config, simple_dataset):
        """Verify evaluate passes model to training strategy."""
        model = nn.Linear(16, 16)
        suff = Sufficiency(
            model=model,
            train_ds=simple_dataset,
            test_ds=simple_dataset,
            config=basic_config,
        )

        suff.evaluate()

        # Check that model was passed to training
        basic_config.training_strategy.train.assert_called()
        call_args = basic_config.training_strategy.train.call_args[0]
        assert call_args[0] is model

    def test_evaluate_passes_train_dataset_to_training(self, basic_config, simple_dataset, mock_model):
        """Verify evaluate passes training dataset to training strategy."""
        suff = Sufficiency(
            model=mock_model,
            train_ds=simple_dataset,
            test_ds=simple_dataset,
            config=basic_config,
        )

        suff.evaluate()

        # Check that train_ds was passed
        call_args = basic_config.training_strategy.train.call_args[0]
        assert call_args[1] is simple_dataset

    def test_evaluate_passes_test_dataset_to_evaluation(self, basic_config, simple_dataset, mock_model):
        """Verify evaluate passes test dataset to evaluation strategy."""
        suff = Sufficiency(
            model=mock_model,
            train_ds=simple_dataset,
            test_ds=simple_dataset,
            config=basic_config,
        )

        suff.evaluate()

        # Check that test_ds was passed
        call_args = basic_config.evaluation_strategy.evaluate.call_args[0]
        assert call_args[1] is simple_dataset

    def test_evaluate_passes_indices_to_training(self, basic_config, simple_dataset, mock_model):
        """Verify evaluate passes indices to training strategy."""
        suff = Sufficiency(
            model=mock_model,
            train_ds=simple_dataset,
            test_ds=simple_dataset,
            config=basic_config,
        )

        suff.evaluate()

        # Check that indices were passed
        call_args = basic_config.training_strategy.train.call_args[0]
        indices = call_args[2]
        assert isinstance(indices, list)
        assert len(indices) > 0

    def test_evaluate_respects_runs_parameter(self, multi_run_config, simple_dataset, mock_model):
        """Verify evaluate runs correct number of times."""
        suff = Sufficiency(
            model=mock_model,
            train_ds=simple_dataset,
            test_ds=simple_dataset,
            config=multi_run_config,
        )

        suff.evaluate()

        # Should call training 3 runs × 2 substeps = 6 times
        assert multi_run_config.training_strategy.train.call_count == 6
        assert multi_run_config.evaluation_strategy.evaluate.call_count == 6
