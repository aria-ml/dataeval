"""Tests for evaluate() method using strategy pattern."""

import numpy.testing as npt
import pytest
import torch.nn as nn

from dataeval.performance import Sufficiency


class TestEvaluateUsesStrategies:
    """Test that evaluate() delegates to strategies correctly."""

    def test_evaluate_calls_training_strategy(self, basic_config, simple_dataset, mock_model, mock_reset):
        """Verify evaluate calls training_strategy.train()."""
        suff = Sufficiency(
            model=mock_model,
            reset_strategy=mock_reset,
            config=basic_config,
        )

        suff.evaluate(simple_dataset, simple_dataset)

        # Should call training for each substep (at least 2 times)
        assert basic_config.training_strategy.train.call_count >= 2

    def test_evaluate_calls_evaluation_strategy(self, basic_config, simple_dataset, mock_model, mock_reset):
        """Verify evaluate calls evaluation_strategy.evaluate()."""
        suff = Sufficiency(
            model=mock_model,
            reset_strategy=mock_reset,
            config=basic_config,
        )

        suff.evaluate(simple_dataset, simple_dataset)

        # Should call evaluation for each substep (at least 2 times)
        assert basic_config.evaluation_strategy.evaluate.call_count >= 2

    def test_evaluate_passes_model_to_training(self, basic_config, simple_dataset):
        """Verify evaluate passes model to training strategy."""
        model = nn.Linear(16, 16)
        suff = Sufficiency(
            model=model,
            config=basic_config,
        )

        suff.evaluate(simple_dataset, simple_dataset)

        # Check that model was passed to training
        basic_config.training_strategy.train.assert_called()
        call_args = basic_config.training_strategy.train.call_args[0]
        assert isinstance(call_args[0], nn.Linear)

    def test_evaluate_passes_train_dataset_to_training(self, basic_config, simple_dataset, mock_model, mock_reset):
        """Verify evaluate passes training dataset to training strategy."""
        suff = Sufficiency(
            model=mock_model,
            reset_strategy=mock_reset,
            config=basic_config,
        )

        suff.evaluate(simple_dataset, simple_dataset)

        # Check that train_dataset was passed
        call_args = basic_config.training_strategy.train.call_args[0]
        assert call_args[1] is simple_dataset

    def test_evaluate_passes_test_dataset_to_evaluation(self, basic_config, simple_dataset, mock_model, mock_reset):
        """Verify evaluate passes test dataset to evaluation strategy."""
        suff = Sufficiency(
            model=mock_model,
            reset_strategy=mock_reset,
            config=basic_config,
        )

        suff.evaluate(simple_dataset, simple_dataset)

        # Check that test_dataset was passed
        call_args = basic_config.evaluation_strategy.evaluate.call_args[0]
        assert call_args[1] is simple_dataset

    def test_evaluate_passes_indices_to_training(self, basic_config, simple_dataset, mock_model, mock_reset):
        """Verify evaluate passes indices to training strategy."""
        suff = Sufficiency(
            model=mock_model,
            reset_strategy=mock_reset,
            config=basic_config,
        )

        suff.evaluate(simple_dataset, simple_dataset)

        # Check that indices were passed
        call_args = basic_config.training_strategy.train.call_args[0]
        indices = call_args[2]
        assert isinstance(indices, list)
        assert len(indices) > 0

    def test_evaluate_respects_runs_parameter(self, multi_run_config, simple_dataset, mock_model, mock_reset):
        """Verify evaluate runs correct number of times."""
        suff = Sufficiency(
            model=mock_model,
            reset_strategy=mock_reset,
            config=multi_run_config,
        )

        suff.evaluate(simple_dataset, simple_dataset)

        # Should call training 3 runs × 2 substeps = 6 times
        assert multi_run_config.training_strategy.train.call_count == 6
        assert multi_run_config.evaluation_strategy.evaluate.call_count == 6


class _RecordingTrain:
    """Training strategy that records the size of every training subset."""

    def __init__(self) -> None:
        self.sizes: list[int] = []

    def train(self, model, dataset, indices) -> None:
        self.sizes.append(len(indices))


class _SizeEval:
    """Evaluation strategy that scores the model by the size it was last trained on."""

    def __init__(self, train: _RecordingTrain) -> None:
        self.train = train

    def evaluate(self, model, dataset) -> dict[str, float]:
        return {"accuracy": self.train.sizes[-1] / 100}


class _Stop(Exception):
    pass


def _sufficiency(runs: int = 2) -> tuple[Sufficiency, _RecordingTrain]:
    train = _RecordingTrain()
    return (
        Sufficiency(
            model=nn.Linear(2, 2),
            training_strategy=train,
            evaluation_strategy=_SizeEval(train),
            reset_strategy=lambda model: model,
            runs=runs,
        ),
        train,
    )


class TestEvaluateProgressCallback:
    """evaluate() reports one progress call per run and step, after that step's metrics are stored."""

    def test_called_once_per_run_and_step_with_a_running_count(self, simple_dataset):
        suff, _ = _sufficiency(runs=2)
        calls: list[tuple[int, int | None, str | None, dict | None]] = []

        def callback(step, *, total=None, desc=None, extra_info=None):
            calls.append((step, total, desc, extra_info))

        suff.evaluate(simple_dataset, simple_dataset, schedule=[10, 20, 50], progress_callback=callback)

        # 2 runs x 3 steps: the count runs 1..6 across both runs, and the total is fixed
        assert [c[0] for c in calls] == [1, 2, 3, 4, 5, 6]
        assert {c[1] for c in calls} == {6}
        assert {c[2] for c in calls} == {"Sufficiency"}

    def test_extra_info_describes_the_run_the_step_and_its_metrics(self, simple_dataset):
        suff, _ = _sufficiency(runs=2)
        infos: list[dict] = []

        def callback(step, *, total=None, desc=None, extra_info=None):
            assert extra_info is not None
            infos.append(extra_info)

        suff.evaluate(simple_dataset, simple_dataset, schedule=[10, 20, 50], progress_callback=callback)

        assert infos[0] == {"run": 0, "runs": 2, "step": 0, "steps": 3, "size": 10, "metrics": {"accuracy": 0.1}}
        assert infos[-1] == {"run": 1, "runs": 2, "step": 2, "steps": 3, "size": 50, "metrics": {"accuracy": 0.5}}
        assert [(i["run"], i["step"], i["size"]) for i in infos] == [
            (run, step, size) for run in (0, 1) for step, size in enumerate((10, 20, 50))
        ]

    def test_an_exception_from_the_callback_stops_the_analysis(self, simple_dataset):
        suff, train = _sufficiency(runs=2)
        seen: list[dict] = []

        def callback(step, *, total=None, desc=None, extra_info=None):
            assert extra_info is not None
            seen.append(extra_info["metrics"])
            if step == 2:
                raise _Stop

        with pytest.raises(_Stop):
            suff.evaluate(simple_dataset, simple_dataset, schedule=[10, 20, 50], progress_callback=callback)

        # nothing was trained after the second evaluation, and the callback holds the metrics of both
        assert train.sizes == [10, 20]
        assert seen == [{"accuracy": 0.1}, {"accuracy": 0.2}]

    def test_the_output_is_the_same_with_or_without_a_callback(self, simple_dataset):
        plain, _ = _sufficiency(runs=2)
        watched, _ = _sufficiency(runs=2)

        expected = plain.evaluate(simple_dataset, simple_dataset, schedule=[10, 20, 50])
        actual = watched.evaluate(
            simple_dataset, simple_dataset, schedule=[10, 20, 50], progress_callback=lambda *a, **k: None
        )

        npt.assert_array_equal(actual.steps, expected.steps)
        npt.assert_array_equal(actual.measures["accuracy"], expected.measures["accuracy"])

    def test_the_call_is_still_recorded_in_the_output_metadata(self, simple_dataset):
        suff, _ = _sufficiency(runs=1)

        output = suff.evaluate(
            simple_dataset, simple_dataset, schedule=[10, 20], progress_callback=lambda *a, **k: None
        )

        assert output.meta().name.endswith("Sufficiency.evaluate")
