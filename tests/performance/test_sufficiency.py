from typing import Literal
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from dataeval.performance import Sufficiency
from dataeval.performance._output import SufficiencyOutput
from dataeval.protocols import EvaluationStrategy, TrainingStrategy
from tests.conftest import SimpleDataset

np.random.seed(0)
torch.manual_seed(0)


def mock_ds(length: int | None):
    ds = MagicMock()
    if length is None:
        delattr(ds, "__len__")
    else:
        ds.__len__.return_value = length
    return ds


@pytest.mark.required
class TestSufficiency:
    def test_mock_run(self, basic_config: Sufficiency.Config, mock_reset, simple_dataset: SimpleDataset) -> None:
        """Verify return value of evaluate is the correct output type."""
        suff = Sufficiency(
            model=MagicMock(),
            reset_strategy=mock_reset,
            config=basic_config,
        )

        results = suff.evaluate(simple_dataset, simple_dataset)
        assert isinstance(results, SufficiencyOutput)

    def test_mock_run_at_value(
        self, basic_config: Sufficiency.Config, mock_reset, simple_dataset: SimpleDataset
    ) -> None:
        """Verify return value of evaluate is the correct output type when run at a specific substep."""
        suff = Sufficiency(model=MagicMock(), reset_strategy=mock_reset, config=basic_config)

        results = suff.evaluate(simple_dataset, simple_dataset, schedule=np.array([1]))
        assert isinstance(results, SufficiencyOutput)

    def test_run_with_invalid_schedule(
        self, basic_config: Sufficiency.Config, mock_reset, simple_dataset: SimpleDataset
    ) -> None:
        """Verifies an invalid schedule type raises a ValueError due to CustomSchedule auto-numpy conversion."""
        suff = Sufficiency(model=MagicMock(), reset_strategy=mock_reset, config=basic_config)

        with pytest.raises(ValueError, match="invalid literal"):
            suff.evaluate(simple_dataset, simple_dataset, schedule="hello world")  # type: ignore

    def test_multiple_runs_multiple_metrics(
        self,
        simple_dataset: SimpleDataset,
        mock_train: TrainingStrategy,
        mock_eval_mixed_metric_strategy: EvaluationStrategy,
        mock_reset,
    ) -> None:
        """Verifies multiple runs, multiple steps, and multiple mixed metrics have the proper output shape."""
        patch("torch.utils.data.DataLoader").start()

        RUNS = 5
        SUBSTEPS = 3
        METRIC_COUNT = 2  # Accuracy (scalar) + Precision (array)
        CLASSES = 2  # Precision has 2 classes

        multi_metric_config = Sufficiency.Config(
            training_strategy=mock_train,
            evaluation_strategy=mock_eval_mixed_metric_strategy,
            runs=RUNS,
            substeps=SUBSTEPS,
        )

        suff = Sufficiency(
            model=MagicMock(),
            reset_strategy=mock_reset,
            config=multi_metric_config,
        )
        output = suff.evaluate(simple_dataset, simple_dataset)

        assert isinstance(output, SufficiencyOutput)
        # Exact curve shape not important so can reduce expensive curve fitting
        assert len(output.get_params(n_iter=10)) == METRIC_COUNT
        assert len(output.measures) == METRIC_COUNT
        assert len(output.averaged_measures) == METRIC_COUNT

        # Scalar metrics: Accuracy
        assert output.measures["Accuracy"].shape == (RUNS, SUBSTEPS)
        assert output.averaged_measures["Accuracy"].shape == (SUBSTEPS,)

        # Array metric: Precision (per-class)
        assert output.measures["Precision"].shape == (RUNS, SUBSTEPS, CLASSES)
        assert output.averaged_measures["Precision"].shape == (CLASSES, SUBSTEPS)

    def test_run_multiple_scalar_metrics(
        self,
        simple_dataset: SimpleDataset,
        mock_train: TrainingStrategy,
        mock_eval_scalar_metrics_strategy: EvaluationStrategy,
        mock_reset,
    ) -> None:
        """Verifies single run with multiple scalar runs has proper output shape."""
        patch("torch.utils.data.DataLoader").start()

        RUNS = 1
        SUBSTEPS = 2
        METRIC_COUNT = 2  # Accuracy + Precision (scalars)

        config = Sufficiency.Config(
            training_strategy=mock_train,
            evaluation_strategy=mock_eval_scalar_metrics_strategy,
            runs=RUNS,
            substeps=SUBSTEPS,
        )

        suff = Sufficiency(
            model=MagicMock(),
            reset_strategy=mock_reset,
            config=config,
        )

        output = suff.evaluate(simple_dataset, simple_dataset)

        # Exact curve shape not important so can reduce expensive curve fitting
        assert len(output.get_params(n_iter=10)) == METRIC_COUNT
        assert len(output.measures) == METRIC_COUNT
        assert len(output.averaged_measures) == METRIC_COUNT

        # Both scalar metrics
        assert output.measures["Accuracy"].shape == (RUNS, SUBSTEPS)
        assert output.averaged_measures["Accuracy"].shape == (SUBSTEPS,)
        assert output.measures["Precision"].shape == (RUNS, SUBSTEPS)
        assert output.averaged_measures["Precision"].shape == (SUBSTEPS,)

    def test_run_classwise(
        self,
        simple_dataset: SimpleDataset,
        mock_train: TrainingStrategy,
        mock_eval_classwise: EvaluationStrategy,
        mock_reset,
    ) -> None:
        """Verifies single run with classwise array metric has proper shape."""
        patch("torch.utils.data.DataLoader").start()

        RUNS = 1
        SUBSTEPS = 2
        CLASSES = 4  # Accuracy returns 4-element array
        METRIC_COUNT = 1  # Accuracy

        config = Sufficiency.Config(
            training_strategy=mock_train,
            evaluation_strategy=mock_eval_classwise,
            runs=RUNS,
            substeps=SUBSTEPS,
        )

        suff = Sufficiency(
            model=MagicMock(),
            reset_strategy=mock_reset,
            config=config,
        )

        output = suff.evaluate(simple_dataset, simple_dataset)

        assert isinstance(output, SufficiencyOutput)
        assert len(output.measures) == METRIC_COUNT
        assert len(output.averaged_measures) == METRIC_COUNT

        # Classwise metric has additional dimension
        # Exact curve shape not important so can reduce expensive curve fitting
        assert output.get_params(n_iter=10)["Accuracy"].shape == (CLASSES, 3)  # 3 curve params per class
        assert output.measures["Accuracy"].shape == (RUNS, SUBSTEPS, CLASSES)
        assert output.averaged_measures["Accuracy"].shape == (CLASSES, SUBSTEPS)

    @pytest.mark.parametrize(
        ("train_ds_len", "test_ds_len", "expected_error"),
        [
            (None, 1, TypeError),
            (1, None, TypeError),
            (0, 1, ValueError),
            (1, 0, ValueError),
            (1, 1, None),
        ],
    )
    def test_dataset_len(
        self,
        basic_config: Sufficiency.Config,
        mock_reset,
        train_ds_len: Literal[1, 0] | None,
        test_ds_len: Literal[1, 0] | None,
        expected_error: type[TypeError] | type[ValueError] | None,
    ):
        def call_evaluate(train_ds_len, test_ds_len):
            suff = Sufficiency(
                model=MagicMock(),
                reset_strategy=mock_reset,
                config=basic_config,
            )
            suff.evaluate(mock_ds(train_ds_len), mock_ds(test_ds_len))

        if expected_error is None:
            call_evaluate(train_ds_len, test_ds_len)
            return

        with pytest.raises(expected_error):
            call_evaluate(train_ds_len, test_ds_len)
