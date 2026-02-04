"""
Tests for Sufficiency constructor during transition.

These tests verify that BOTH old and new constructor signatures work
during the compatibility period. The old signature will be removed in
a future step.
"""

from dataeval.performance import Sufficiency


class TestSufficiencyConfigConstructor:
    """Test that new constructor signature works."""

    def test_sufficiency_reads_from_config(self, mock_model, simple_dataset, basic_config):
        """Verify constructor stores params and Sufficiency.Config."""

        suff = Sufficiency(
            model=mock_model,
            train_ds=simple_dataset,
            test_ds=simple_dataset,
            config=basic_config,
        )

        assert suff.model is mock_model
        assert suff.train_ds is simple_dataset
        assert suff.test_ds is simple_dataset
        assert isinstance(suff.config, Sufficiency.Config)
        assert suff.runs == 1
        assert suff.substeps == 5

    def test_new_signature_stores_config_reference(
        self,
        mock_model,
        simple_dataset,
        mock_train,
        mock_eval,
        mock_reset,
    ):
        """Verify constructor stores config objects."""

        config = Sufficiency.Config(
            training_strategy=mock_train,
            evaluation_strategy=mock_eval,
            reset_strategy=mock_reset,
        )

        suff = Sufficiency(mock_model, simple_dataset, simple_dataset, config=config)

        assert isinstance(suff.config, Sufficiency.Config)
        assert suff.config.training_strategy is mock_train
        assert suff.config.evaluation_strategy is mock_eval

    def test_sufficiency_object_delegates_to_config(self, mock_model, simple_dataset, multi_run_config):
        """Verify new constructor delegates properties to config."""

        suff = Sufficiency(mock_model, simple_dataset, simple_dataset, config=multi_run_config)

        # Should delegate to config
        assert isinstance(multi_run_config, Sufficiency.Config)
        assert suff.runs == multi_run_config.runs
        assert suff.substeps == multi_run_config.substeps
        assert suff.unit_interval == multi_run_config.unit_interval
