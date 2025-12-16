"""
Tests for Sufficiency constructor during transition.

These tests verify that BOTH old and new constructor signatures work
during the compatibility period. The old signature will be removed in
a future step.
"""

from dataeval.performance import Sufficiency, SufficiencyConfig


class TestSufficiencyConfigConstructor:
    """Test that new constructor signature works."""

    def test_sufficiency_reads_from_config(self, mock_model, simple_dataset, basic_config):
        """Verify constructor stores params and SufficiencyConfig."""

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
        self,
        mock_model,
        simple_dataset,
        mock_training_strategy,
        mock_evaluation_strategy,
    ):
        """Verify constructor stores config objects."""

        config = SufficiencyConfig(mock_training_strategy, mock_evaluation_strategy)

        suff = Sufficiency(mock_model, simple_dataset, simple_dataset, config)

        assert suff.config is config
        assert suff.config.training_strategy is mock_training_strategy
        assert suff.config.evaluation_strategy is mock_evaluation_strategy

    def test_sufficiency_object_delegates_to_config(self, mock_model, simple_dataset, multi_run_config):
        """Verify new constructor delegates properties to config."""

        suff = Sufficiency(mock_model, simple_dataset, simple_dataset, multi_run_config)

        # Should delegate to config
        assert suff.runs == multi_run_config.runs
        assert suff.substeps == multi_run_config.substeps
        assert suff.unit_interval == multi_run_config.unit_interval
