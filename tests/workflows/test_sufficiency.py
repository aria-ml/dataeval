from __future__ import annotations

from unittest.mock import MagicMock, NonCallableMagicMock, patch

import numpy as np
import numpy.testing as npt
import pytest
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from torch.utils.data import DataLoader

from dataeval._internal.workflows.sufficiency import (
    SufficiencyOutput,
    f_inv_out,
    f_out,
    inv_project_steps,
    project_steps,
)
from dataeval.workflows import Sufficiency
from tests.utils.data import DataEvalDataset

np.random.seed(0)
torch.manual_seed(0)


def load_cls_dataset() -> tuple[DataEvalDataset, DataEvalDataset]:
    images = np.ones(shape=(1, 32, 32, 3))
    labels = np.ones(shape=(1, 1))

    train_ds = DataEvalDataset(images, labels)
    test_ds = DataEvalDataset(images, labels)

    return train_ds, test_ds


def load_od_dataset() -> tuple[DataEvalDataset, DataEvalDataset]:
    images = np.ones(shape=(1, 32, 32, 3))
    labels = np.ones(shape=(1, 1))
    boxes = np.ones(shape=(1, 1, 4))

    train_ds = DataEvalDataset(images, labels, boxes)
    test_ds = DataEvalDataset(images, labels, boxes)

    return train_ds, test_ds


def eval_100(model: nn.Module, dl: DataLoader) -> dict[str, float]:
    """Eval should always return a float, and error if not"""
    return {"eval": 1.0}


def mock_ds(length: int | None):
    ds = MagicMock()
    if length is None:
        delattr(ds, "__len__")
    else:
        ds.__len__.return_value = length
    return ds


class TestSufficiency:
    def test_mock_run(self) -> None:
        eval_fn = MagicMock()
        eval_fn.return_value = {"test": 1.0}
        patch("torch.utils.data.DataLoader").start()

        suff = Sufficiency(
            model=MagicMock(),
            train_ds=mock_ds(2),
            test_ds=mock_ds(2),
            train_fn=MagicMock(),
            eval_fn=eval_fn,
            runs=1,
            substeps=2,
        )

        results = suff.evaluate(niter=100)
        assert isinstance(results, SufficiencyOutput)

    def test_mock_run_at_value(self) -> None:
        eval_fn = MagicMock()
        eval_fn.return_value = {"test": 1.0}
        patch("torch.utils.data.DataLoader").start()

        suff = Sufficiency(
            model=MagicMock(),
            train_ds=mock_ds(2),
            test_ds=mock_ds(2),
            train_fn=MagicMock(),
            eval_fn=eval_fn,
            runs=1,
            substeps=2,
        )

        results = suff.evaluate(np.array([1]))
        assert isinstance(results, SufficiencyOutput)

    def test_mock_run_with_kwargs(self) -> None:
        train_fn = MagicMock()
        eval_fn = MagicMock()
        eval_fn.return_value = {"test": 1.0}
        train_kwargs = {"train": 1}
        eval_kwargs = {"eval": 1}
        patch("torch.utils.data.DataLoader").start()

        suff = Sufficiency(
            model=MagicMock(),
            train_ds=mock_ds(2),
            test_ds=mock_ds(2),
            train_fn=train_fn,
            eval_fn=eval_fn,
            runs=1,
            substeps=2,
            train_kwargs=train_kwargs,
            eval_kwargs=eval_kwargs,
        )

        results = suff.evaluate(niter=100)

        assert train_fn.call_count == 2
        assert train_kwargs == train_fn.call_args.kwargs

        assert eval_fn.call_count == 2
        assert eval_kwargs == eval_fn.call_args.kwargs

        assert isinstance(results, SufficiencyOutput)

    def test_run_with_invalid_eval_at(self) -> None:
        suff = Sufficiency(
            model=MagicMock(),
            train_ds=mock_ds(2),
            test_ds=mock_ds(2),
            train_fn=MagicMock(),
            eval_fn=MagicMock(),
            runs=1,
            substeps=2,
        )

        with pytest.raises(ValueError):
            suff.evaluate("hello world")  # type: ignore

    def test_run_multiple_metrics(self) -> None:
        eval_fn = MagicMock()
        eval_fn.return_value = {"Accuracy": 1.0, "Precision": 1.0}
        patch("torch.utils.data.DataLoader").start()

        suff = Sufficiency(
            model=MagicMock(),
            train_ds=mock_ds(2),
            test_ds=mock_ds(2),
            train_fn=MagicMock(),
            eval_fn=eval_fn,
            runs=1,
            substeps=2,
        )

        output = suff.evaluate(niter=100)
        assert len(output.params) == 2
        assert len(output.measures) == 2

    def test_run_classwise(self) -> None:
        eval_fn = MagicMock()
        eval_fn.return_value = {"Accuracy": np.array([0.2, 0.4, 0.6, 0.8])}
        patch("torch.utils.data.DataLoader").start()

        suff = Sufficiency(
            model=MagicMock(),
            train_ds=mock_ds(2),
            test_ds=mock_ds(2),
            train_fn=MagicMock(),
            eval_fn=eval_fn,
            runs=1,
            substeps=2,
        )

        output = suff.evaluate(niter=100)
        assert output.params["Accuracy"].shape == (4, 3)
        assert len(output.measures) == 1

    @pytest.mark.parametrize(
        "train_ds_len, test_ds_len, expected_error",
        [
            (None, 1, TypeError),
            (1, None, TypeError),
            (0, 1, ValueError),
            (1, 0, ValueError),
            (1, 1, None),
        ],
    )
    def test_dataset_len(self, train_ds_len, test_ds_len, expected_error):
        def call_suff(train_ds_len, test_ds_len):
            Sufficiency(
                model=MagicMock(),
                train_ds=mock_ds(train_ds_len),
                test_ds=mock_ds(test_ds_len),
                train_fn=MagicMock(),
                eval_fn=MagicMock(),
            )

        if expected_error is None:
            call_suff(train_ds_len, test_ds_len)
            return

        with pytest.raises(expected_error):
            call_suff(train_ds_len, test_ds_len)

    def test_train_fn_is_non_callable(self):
        with pytest.raises(TypeError):
            Sufficiency(
                model=MagicMock(),
                train_ds=mock_ds(1),
                test_ds=mock_ds(1),
                train_fn=NonCallableMagicMock(),
                eval_fn=MagicMock(),
            )

    def test_eval_fn_is_non_callable(self):
        with pytest.raises(TypeError):
            Sufficiency(
                model=MagicMock(),
                train_ds=mock_ds(1),
                test_ds=mock_ds(1),
                train_fn=MagicMock(),
                eval_fn=NonCallableMagicMock(),
            )

    def test_output_params_measures_mismatch(self):
        with pytest.raises(ValueError):
            SufficiencyOutput(
                steps=np.array([1, 5, 10]),
                params={"a": np.array([0.1, 0.5, 1.0])},
                measures={"b": np.array([0.2, 0.4, 0.6])},
            )


class TestSufficiencyPlot:
    def test_plot(self):
        """Tests that a plot is generated"""
        # Only needed for plotting test
        output = SufficiencyOutput(
            steps=np.array([10, 100, 1000]),
            params={"test": np.array([-0.1, -1.0, 1.0])},
            measures={"test": np.array([0.2, 0.6, 0.9])},
        )
        result = output.plot()
        assert len(result) == 1
        assert isinstance(result[0], Figure)

    def test_multiplot(self):
        """Tests that the multiple plots are generated"""
        output = SufficiencyOutput(
            steps=np.array([10, 100, 1000]),
            params={
                "test1": np.array([-0.1, -1.0, 1.0]),
                "test2": np.array([-0.1, -1.0, 1.0]),
                "test3": np.array([-0.1, -1.0, 1.0]),
            },
            measures={
                "test1": np.array([0.2, 0.6, 0.9]),
                "test2": np.array([0.2, 0.6, 0.9]),
                "test3": np.array([0.2, 0.6, 0.9]),
            },
        )

        result = output.plot()
        assert len(result) == 3
        assert isinstance(result[0], Figure)

    def test_multiplot_classwise(self):
        output = SufficiencyOutput(
            steps=np.array([10, 100, 1000]),
            params={"test1": np.array([[-0.1, -1.0, 1.0], [-0.1, -1.0, 1.0]])},
            measures={"test1": np.array([[0.2, 0.6, 0.9], [0.3, 0.4, 0.8]])},
        )

        result = output.plot()
        assert len(result) == 2
        assert isinstance(result[0], Figure)

    def test_multiplot_classwise_invalid_names(self):
        output = SufficiencyOutput(
            steps=np.array([10, 100, 1000]),
            params={"test1": np.array([[-0.1, -1.0, 1.0], [-0.1, -1.0, 1.0]])},
            measures={"test1": np.array([[0.2, 0.6, 0.9], [0.3, 0.4, 0.8]])},
        )

        with pytest.raises(IndexError):
            output.plot(["A", "B", "C"])

    def test_multiplot_classwise_with_names(self):
        output = SufficiencyOutput(
            steps=np.array([10, 100, 1000]),
            params={"test1": np.array([[-0.1, -1.0, 1.0], [-0.1, -1.0, 1.0]])},
            measures={"test1": np.array([[0.2, 0.6, 0.9], [0.3, 0.4, 0.8]])},
        )

        result = output.plot(["A", "B"])
        assert result[0].axes[0].get_title().startswith("test1_A")

    def test_multiplot_classwise_without_names(self):
        output = SufficiencyOutput(
            steps=np.array([10, 100, 1000]),
            params={"test1": np.array([[-0.1, -1.0, 1.0], [-0.1, -1.0, 1.0]])},
            measures={"test1": np.array([[0.2, 0.6, 0.9], [0.3, 0.4, 0.8]])},
        )

        result = output.plot()
        assert result[0].axes[0].get_title().startswith("test1_0")

    def test_multiplot_mixed(self):
        output = SufficiencyOutput(
            steps=np.array([10, 100, 1000]),
            params={"test1": np.array([[-0.1, -1.0, 1.0], [-0.1, -1.0, 1.0]]), "test2": np.array([-0.1, -1.0, 1.0])},
            measures={"test1": np.array([[0.2, 0.6, 0.9], [0.3, 0.4, 0.8]]), "test2": np.array([0.2, 0.6, 0.9])},
        )

        result = output.plot()
        assert len(result) == 3
        assert result[0].axes[0].get_title().startswith("test1_0")
        assert result[1].axes[0].get_title().startswith("test1_1")
        assert result[2].axes[0].get_title().startswith("test2")


class TestSufficiencyProject:
    def test_measure_length_invalid(self):
        with pytest.raises(ValueError):
            SufficiencyOutput(
                steps=np.array([10, 100]),
                params={"test1": np.array([-0.1, -1.0, 1.0])},
                measures={"test1": np.array([0.2, 0.6, 0.9])},
            )

    @pytest.mark.parametrize("steps", [100.0, 100, [100], np.array([100])])
    def test_project(self, steps):
        output = SufficiencyOutput(
            steps=np.array([10, 100, 1000]),
            params={"test1": np.array([-0.1, -1.0, 1.0])},
            measures={"test1": np.array([0.2, 0.6, 0.9])},
        )
        result = output.project(steps)
        npt.assert_almost_equal(result.measures["test1"], [10.0], decimal=4)

    def test_project_invalid_steps(self):
        output = SufficiencyOutput(
            steps=np.array([10, 100, 1000]),
            params={"test1": np.array([-0.1, -1.0, 1.0])},
            measures={"test1": np.array([0.2, 0.6, 0.9])},
        )
        with pytest.raises(ValueError):
            output.project("not a number")  # type: ignore

    def test_project_classwise(self):
        output = SufficiencyOutput(
            steps=np.array([10, 100, 1000]),
            params={"test1": np.array([[-0.1, -1.0, 1.0], [-0.1, -1.0, 1.0]])},
            measures={"test1": np.array([[0.2, 0.6, 0.9], [0.3, 0.4, 0.8]])},
        )

        assert output.measures["test1"].shape == (2, 3)
        result = output.project([1000, 2000, 4000, 8000])
        assert len(result.measures) == 1
        assert result.measures["test1"].shape == (2, 4)

    def test_project_mixed(self):
        output = SufficiencyOutput(
            steps=np.array([10, 100, 1000]),
            params={"test1": np.array([[-0.1, -1.0, 1.0], [-0.1, -1.0, 1.0]]), "test2": np.array([-0.1, -1.0, 1.0])},
            measures={"test1": np.array([[0.2, 0.6, 0.9], [0.3, 0.4, 0.8]]), "test2": np.array([0.2, 0.6, 0.9])},
        )

        assert output.measures["test1"].shape == (2, 3)
        assert output.measures["test2"].shape == (3,)
        result = output.project([1000, 2000, 4000, 8000])
        assert len(result.measures) == 2
        assert result.measures["test1"].shape == (2, 4)
        assert result.measures["test2"].shape == (4,)

    def test_inv_project_mixed(self):
        output = SufficiencyOutput(
            steps=np.array([10, 100, 1000]),
            params={
                "test1": np.array([[-0.1, -1.0, 1.0], [-0.1, -1.0, 1.0]]),
                "test2": np.array([-0.1, -1.0, 1.0]),
            },
            measures={"test1": np.array([[0.2, 0.6, 0.9], [0.3, 0.4, 0.8]]), "test2": np.array([0.2, 0.6, 0.9])},
        )

        targets = {"test1": np.array([0.6, 0.7, 0.8, 0.9]), "test2": np.array([0.6, 0.7, 0.8, 0.9])}

        result = output.inv_project(targets)
        assert len(result.keys()) == 2
        assert result["test1"].shape == (2, 4)
        assert result["test2"].shape == (4,)

    def test_inv_project_ignore_unknown_measure(self):
        output = SufficiencyOutput(
            steps=np.array([10, 100, 1000]),
            params={"test1": np.array([[-0.1, -1.0, 1.0], [-0.1, -1.0, 1.0]])},
            measures={"test1": np.array([[0.2, 0.6, 0.9], [0.3, 0.4, 0.8]])},
        )

        targets = {"test1": np.array([0.6, 0.7, 0.8, 0.9]), "test2": np.array([0.6, 0.7, 0.8, 0.9])}

        result = output.inv_project(targets)
        assert len(result.keys()) == 1
        assert result["test1"].shape == (2, 4)


class TestSufficiencyInverseProject:
    def test_empty_data(self):
        """
        Verifies that inv_project returns empty data when fed empty data
        """
        data = SufficiencyOutput(np.array([]), params={}, measures={})
        desired_accuracies = {}
        assert len(data.inv_project(desired_accuracies)) == 0

    def test_can_invert_sufficiency(self):
        """
        Tests metric projection output can be inversed
        """
        num_samples = np.arange(20, 80, step=10, dtype=np.uint)
        accuracies = num_samples / 100.0

        params = np.array([-0.01, -1.0, 1.0])

        data = SufficiencyOutput(steps=num_samples, params={"Accuracy": params}, measures={"Accuracy": accuracies})

        desired_accuracies = {"Accuracy": np.array([0.4, 0.6])}
        needed_data = data.inv_project(desired_accuracies)["Accuracy"]

        target_needed_data = np.array([40, 60])
        npt.assert_array_equal(needed_data, target_needed_data)

    def test_f_inv_out(self):
        """
        Tests that f_inv_out exactly inverts f_out.
        """

        n_i = np.array([1.234])
        x = np.array([1.1, 2.2, 3.3])
        # Predict y from n_i evaluated on curve defined by x
        y = f_out(n_i, x)
        # Feed y into inverse function to get the original n_i back out
        n_i_recovered = f_inv_out(y, x)

        # Convert to uint32 for step sizes
        npt.assert_equal(np.uint32(n_i[0]), n_i_recovered[0])

    def test_inv_project_steps(self):
        """
        Verifies that inv_project_steps is the inverse of project_steps
        """
        projection = np.array([1, 2, 3])
        # Pre-calculated from other runs (not strict)
        params = np.array([-1.0, -1.0, 4.0])

        # Estimated accuracies at each step
        accuracies = project_steps(params, projection)
        # Estimated steps needed to get each accuracy
        predicted_proj = inv_project_steps(params, accuracies)

        # assert np.all(np.isclose(projection, predicted_proj, atol=1))
        npt.assert_array_equal(projection, predicted_proj)
