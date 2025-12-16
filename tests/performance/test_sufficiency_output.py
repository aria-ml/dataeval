from __future__ import annotations

import warnings

import numpy as np
import numpy.testing as npt
import pytest

from dataeval.performance._output import (
    SufficiencyOutput,
    f_inv_out,
    f_out,
    inv_project_steps,
    project_steps,
)

np.random.seed(0)


@pytest.fixture(scope="module")
def so_single_averaged_inputs() -> SufficiencyOutput:
    output = SufficiencyOutput(
        steps=np.array([10, 100, 1000]), averaged_measures={"test1": np.array([0.2, 0.6, 0.9])}, measures={}
    )
    output._params = {1000: {"test1": np.array([-0.1, -1.0, 1.0])}}
    return output


@pytest.fixture(scope="module")
def so_single_unaveraged_inputs() -> SufficiencyOutput:
    output = SufficiencyOutput(
        steps=np.array([10, 100, 1000]),
        measures={
            "test1": np.array(
                [
                    [0.2, 0.5, 0.9],
                    [0.1, 0.6, 0.9],
                    [0.3, 0.7, 0.9],
                ]
            )
        },
    )
    output._params = {1000: {"test1": np.array([-0.1, -1.0, 0.02])}}
    return output


@pytest.fixture(scope="module")
def so_multi_averaged_inputs() -> SufficiencyOutput:
    output = SufficiencyOutput(
        steps=np.array([10, 100, 1000]),
        measures={},
        averaged_measures={"test1": np.array([[0.2, 0.6, 0.9], [0.3, 0.4, 0.8]])},
    )
    output._params = {1000: {"test1": np.array([[-0.1, -1.0, 1.0], [-0.1, -1.0, 1.0]])}}
    return output


@pytest.fixture(scope="module")
def so_multi_unaveraged_inputs() -> SufficiencyOutput:
    output = SufficiencyOutput(
        steps=np.array([10, 100, 1000]),
        measures={
            "test1": np.array(
                [
                    [[0.2, 0.1], [0.5, 0.4], [0.9, 0.7]],
                    [[0.1, 0.3], [0.6, 0.4], [0.9, 0.8]],
                    [[0.3, 0.5], [0.7, 0.4], [0.9, 0.9]],
                ]
            )
        },
    )
    output._params = {1000: {"test1": np.array([[-0.1, -1.0, 0.5], [-0.1, -1.0, 1.0]])}}
    return output


@pytest.fixture(scope="module")
def so_mixed_averaged_inputs() -> SufficiencyOutput:
    output = SufficiencyOutput(
        steps=np.array([10, 100, 1000]),
        averaged_measures={"test1": np.array([[0.2, 0.6, 0.9], [0.3, 0.4, 0.8]]), "test2": np.array([0.2, 0.6, 0.9])},
        measures={},
    )
    output._params = {
        1000: {"test1": np.array([[-0.1, -1.0, 1.0], [-0.1, -1.0, 1.0]]), "test2": np.array([-0.1, -1.0, 1.0])}
    }
    return output


@pytest.fixture(scope="module")
def so_mixed_unaveraged_inputs() -> SufficiencyOutput:
    output = SufficiencyOutput(
        steps=np.array([10, 100, 1000]),
        measures={
            "test1": np.array(
                [
                    [[0.2, 0.1], [0.5, 0.4], [0.9, 0.7]],
                    [[0.1, 0.3], [0.6, 0.4], [0.9, 0.8]],
                    [[0.3, 0.5], [0.7, 0.4], [0.9, 0.9]],
                ]
            ),
            "test2": np.array([[0.1, 0.5, 0.9], [0.2, 0.6, 0.9], [0.3, 0.7, 0.9]]),
        },
    )
    output._params = {
        1000: {"test1": np.array([[-0.1, -1.0, 1.0], [-0.1, -1.0, 0.5]]), "test2": np.array([-0.1, -1.0, 1.0])}
    }
    return output


def find_horizontal_line(ax):
    """Find and return horizontal asymptote y value"""

    for line in ax.lines:
        ydata = line.get_ydata()
        if all(abs(y - ydata[0]) == 0 for y in ydata):
            return ydata[0]
    return None


def find_error_bars(ax, mcoll):
    """Return plotted error for each point"""
    err = []
    for coll in ax.collections:
        if isinstance(coll, mcoll.LineCollection):
            segments = coll.get_segments()
            for seg in segments:
                y0, y1 = seg[0][1], seg[1][1]
                if abs(y0 - y1) > 0:
                    # Get error from bar height
                    err.append((y1 - y0) / 2)
    err = np.array(err)
    err = err[~np.isclose(err, 0)]
    return err


@pytest.mark.required
class TestSufficiencyProject:
    def test_measure_length_invalid(self):
        with pytest.raises(ValueError):
            SufficiencyOutput(
                steps=np.array([10, 100]), averaged_measures={"test1": np.array([0.2, 0.6, 0.9])}, measures={}
            )

    def test_unaveraged_inputs_measure_length_invalid(self):
        with pytest.raises(ValueError):
            SufficiencyOutput(
                steps=np.array([10, 100]),
                measures={"test1": np.array([[0.2, 0.6, 0.9], [0.2, 0.6, 0.9], [0.2, 0.6, 0.9]])},
            )

    @pytest.mark.parametrize("steps", [100.0, 100, [100], np.array([100])])
    def test_input_project(self, steps, so_single_averaged_inputs):
        result = so_single_averaged_inputs.project(steps)
        npt.assert_almost_equal(result.averaged_measures["test1"], [10.0], decimal=4)

    def test_project_invalid_steps(self, so_single_averaged_inputs):
        with pytest.raises(ValueError):
            so_single_averaged_inputs.project("not a number")  # type: ignore

    def test_project_classwise(self, so_multi_averaged_inputs):
        assert so_multi_averaged_inputs.averaged_measures["test1"].shape == (2, 3)
        result = so_multi_averaged_inputs.project([1000, 2000, 4000, 8000])
        assert len(result.averaged_measures) == 1
        assert result.averaged_measures["test1"].shape == (2, 4)

    def test_unaveraged_inputs_project_classwise(self, so_multi_unaveraged_inputs):
        assert so_multi_unaveraged_inputs.averaged_measures["test1"].shape == (2, 3)
        assert so_multi_unaveraged_inputs.measures["test1"].shape == (3, 3, 2)
        test = np.array([[0.2, 0.6, 0.9], [0.3, 0.4, 0.8]])
        assert np.allclose(
            so_multi_unaveraged_inputs.averaged_measures["test1"],
            test,
            rtol=0,
            atol=1e-12,
        )

        result = so_multi_unaveraged_inputs.project([1000, 2000, 4000, 8000])
        assert len(result.averaged_measures) == 1
        assert result.averaged_measures["test1"].shape == (2, 4)

    def test_project_mixed(self, so_mixed_averaged_inputs):
        assert so_mixed_averaged_inputs.averaged_measures["test1"].shape == (2, 3)
        assert so_mixed_averaged_inputs.averaged_measures["test2"].shape == (3,)
        result = so_mixed_averaged_inputs.project([1000, 2000, 4000, 8000])
        assert len(result.averaged_measures) == 2
        assert result.averaged_measures["test1"].shape == (2, 4)
        assert result.averaged_measures["test2"].shape == (4,)

    def test_unaveraged_inputs_project_mixed(self, so_mixed_unaveraged_inputs):
        assert so_mixed_unaveraged_inputs.measures["test1"].shape == (3, 3, 2)
        assert so_mixed_unaveraged_inputs.measures["test2"].shape == (3, 3)
        assert so_mixed_unaveraged_inputs.averaged_measures["test1"].shape == (2, 3)
        assert so_mixed_unaveraged_inputs.averaged_measures["test2"].shape == (3,)
        result = so_mixed_unaveraged_inputs.project([1000, 2000, 4000, 8000])
        assert len(result.averaged_measures) == 2
        assert result.averaged_measures["test1"].shape == (2, 4)
        assert result.averaged_measures["test2"].shape == (4,)
        test = np.array([[0.2, 0.6, 0.9], [0.3, 0.4, 0.8]])
        assert np.allclose(
            so_mixed_unaveraged_inputs.averaged_measures["test1"],
            test,
            rtol=0,
            atol=1e-12,
        )
        test = np.array([[0.2, 0.6, 0.9]])
        assert np.allclose(
            so_mixed_unaveraged_inputs.averaged_measures["test2"],
            test,
            rtol=0,
            atol=1e-12,
        )

    def test_inv_project_mixed(self, so_mixed_averaged_inputs):
        targets = {"test1": np.array([0.6, 0.7, 0.8, 0.9]), "test2": np.array([0.6, 0.7, 0.8, 0.9])}

        result = so_mixed_averaged_inputs.inv_project(targets)
        assert len(result.keys()) == 2
        assert result["test1"].shape == (2, 4)
        assert result["test2"].shape == (4,)

    def test_inv_project_ignore_unknown_measure(self, so_multi_averaged_inputs):
        targets = {"test1": np.array([0.6, 0.7, 0.8, 0.9]), "test2": np.array([0.6, 0.7, 0.8, 0.9])}

        result = so_multi_averaged_inputs.inv_project(targets)
        assert len(result.keys()) == 1
        assert result["test1"].shape == (2, 4)


@pytest.mark.required
class TestSufficiencyInverseProject:
    def test_empty_data(self):
        """
        Verifies that inv_project returns empty data when fed empty data
        """
        data = SufficiencyOutput(np.array([]), measures={}, averaged_measures={})
        desired_accuracies = {}
        assert len(data.inv_project(desired_accuracies)) == 0

    def test_unaveraged_inputs_empty_data(self):
        """
        Verifies that inv_project returns empty data when fed empty data and initialized with unaveraged measures
        """
        data = SufficiencyOutput(np.array([]), measures={})
        desired_accuracies = {}
        assert len(data.inv_project(desired_accuracies)) == 0

    def test_can_invert_sufficiency(self):
        """
        Tests metric projection output can be inversed
        """
        num_samples = np.arange(20, 80, step=10, dtype=np.intp)
        accuracies = num_samples / 100.0

        data = SufficiencyOutput(steps=num_samples, measures={}, averaged_measures={"Accuracy": accuracies})
        data._params = {1000: {"Accuracy": np.array([-0.01, -1.0, 1.0])}}

        desired_accuracies = {"Accuracy": np.array([0.4, 0.6])}
        needed_data = data.inv_project(desired_accuracies)["Accuracy"]

        target_needed_data = np.array([40, 60])
        npt.assert_array_equal(needed_data, target_needed_data)

    def test_unaveraged_inputs_can_invert_sufficiency(self):
        """
        Tests metric projection output can be inversed
        """
        accuracies = np.array(
            [
                np.arange(10, 70, step=10, dtype=np.uint32) / 100,
                np.arange(20, 80, step=10, dtype=np.uint32) / 100,
                np.arange(30, 90, step=10, dtype=np.uint32) / 100,
            ]
        )
        num_samples = accuracies[1] * 100
        data = SufficiencyOutput(steps=num_samples, measures={"Accuracy": accuracies})
        data._params = {1000: {"Accuracy": np.array([-0.01, -1.0, 1.0])}}

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

    def test_f_inv_out_unachievable_targets(self, caplog):
        """
        Verifies that f_inv_out handles unachievable targets
        """
        import logging

        num_samples = np.arange(20, 80, step=10, dtype=np.intp)
        accuracies = num_samples / 100.0
        data = SufficiencyOutput(steps=num_samples, measures={}, averaged_measures={"Accuracy": accuracies})
        # upper bound for these parameters is 0.9369, any desired accuracy above is unachievable
        data._params = {1000: {"Accuracy": np.array([12.2746, 0.8502, 0.0631])}}
        desired_accuracies = {"Accuracy": np.array([0.00000001, 0.93689])}

        # ensure there are no warnings for valid input
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            needed_data = data.inv_project(desired_accuracies)["Accuracy"]

        # 0.90 and 0.93 targets achievable, 0.99 above curve upper bound
        desired_accuracies = {"Accuracy": np.array([0.90, 0.93, 0.99])}
        # expect warning for 0.99 target
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            needed_data = data.inv_project(desired_accuracies)["Accuracy"]
        assert "Number of samples could not be determined for target(s): [0.99] with asymptote of 0.9369" in caplog.text
        target_needed_data = np.array([925, 6649, -1])
        npt.assert_array_equal(needed_data, target_needed_data)

        # all target accuracies unachievable, 0.9368 returns value greater than int64
        desired_accuracies = {"Accuracy": np.array([0.9369, 1, 1.01])}
        # expect warning for all targets
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            needed_data = data.inv_project(desired_accuracies)["Accuracy"]
        assert "Number of samples could not be determined for target(s): [0.9369, 1.0, 1.01]" in caplog.text

        target_needed_data = np.array([-1, -1, -1])
        npt.assert_array_equal(needed_data, target_needed_data)
