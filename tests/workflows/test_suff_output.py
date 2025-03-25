from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

try:
    from matplotlib.figure import Figure
except ImportError:
    Figure = type(None)

from dataeval.outputs._workflows import (
    SufficiencyOutput,
    f_inv_out,
    f_out,
    inv_project_steps,
    project_steps,
)

np.random.seed(0)


@pytest.fixture(scope="module")
def so_single() -> SufficiencyOutput:
    output = SufficiencyOutput(
        steps=np.array([10, 100, 1000]),
        measures={"test1": np.array([0.2, 0.6, 0.9])},
    )
    output._params = {1000: {"test1": np.array([-0.1, -1.0, 1.0])}}
    return output


@pytest.fixture(scope="module")
def so_multi() -> SufficiencyOutput:
    output = SufficiencyOutput(
        steps=np.array([10, 100, 1000]),
        measures={"test1": np.array([[0.2, 0.6, 0.9], [0.3, 0.4, 0.8]])},
    )
    output._params = {1000: {"test1": np.array([[-0.1, -1.0, 1.0], [-0.1, -1.0, 1.0]])}}
    return output


@pytest.fixture(scope="module")
def so_mixed() -> SufficiencyOutput:
    output = SufficiencyOutput(
        steps=np.array([10, 100, 1000]),
        measures={"test1": np.array([[0.2, 0.6, 0.9], [0.3, 0.4, 0.8]]), "test2": np.array([0.2, 0.6, 0.9])},
    )
    output._params = {
        1000: {"test1": np.array([[-0.1, -1.0, 1.0], [-0.1, -1.0, 1.0]]), "test2": np.array([-0.1, -1.0, 1.0])}
    }
    return output


@pytest.mark.requires_all
@pytest.mark.required
class TestSufficiencyPlot:
    def test_plot(self, so_single):
        """Tests that a plot is generated"""
        # Only needed for plotting test
        result = so_single.plot()
        assert len(result) == 1
        assert isinstance(result[0], Figure)

    def test_multiplot(self, so_mixed):
        result = so_mixed.plot()
        assert len(result) == 3
        assert isinstance(result[0], Figure)

    def test_multiplot_classwise(self, so_multi):
        result = so_multi.plot()
        assert len(result) == 2
        assert isinstance(result[0], Figure)

    def test_multiplot_classwise_invalid_names(self, so_multi):
        with pytest.raises(IndexError):
            so_multi.plot(["A", "B", "C"])

    def test_multiplot_classwise_with_names(self, so_multi):
        result = so_multi.plot(["A", "B"])
        assert result[0].axes[0].get_title().startswith("test1_A")

    def test_multiplot_classwise_without_names(self, so_multi):
        result = so_multi.plot()
        assert result[0].axes[0].get_title().startswith("test1_0")

    def test_multiplot_mixed(self, so_mixed):
        result = so_mixed.plot()
        assert len(result) == 3
        assert result[0].axes[0].get_title().startswith("test1_0")
        assert result[1].axes[0].get_title().startswith("test1_1")
        assert result[2].axes[0].get_title().startswith("test2")


@pytest.mark.required
class TestSufficiencyProject:
    def test_measure_length_invalid(self):
        with pytest.raises(ValueError):
            SufficiencyOutput(
                steps=np.array([10, 100]),
                measures={"test1": np.array([0.2, 0.6, 0.9])},
            )

    @pytest.mark.parametrize("steps", [100.0, 100, [100], np.array([100])])
    def test_project(self, steps, so_single):
        result = so_single.project(steps)
        npt.assert_almost_equal(result.measures["test1"], [10.0], decimal=4)

    def test_project_invalid_steps(self, so_single):
        with pytest.raises(ValueError):
            so_single.project("not a number")  # type: ignore

    def test_project_classwise(self, so_multi):
        assert so_multi.measures["test1"].shape == (2, 3)
        result = so_multi.project([1000, 2000, 4000, 8000])
        assert len(result.measures) == 1
        assert result.measures["test1"].shape == (2, 4)

    def test_project_mixed(self, so_mixed):
        assert so_mixed.measures["test1"].shape == (2, 3)
        assert so_mixed.measures["test2"].shape == (3,)
        result = so_mixed.project([1000, 2000, 4000, 8000])
        assert len(result.measures) == 2
        assert result.measures["test1"].shape == (2, 4)
        assert result.measures["test2"].shape == (4,)

    def test_inv_project_mixed(self, so_mixed):
        targets = {"test1": np.array([0.6, 0.7, 0.8, 0.9]), "test2": np.array([0.6, 0.7, 0.8, 0.9])}

        result = so_mixed.inv_project(targets)
        assert len(result.keys()) == 2
        assert result["test1"].shape == (2, 4)
        assert result["test2"].shape == (4,)

    def test_inv_project_ignore_unknown_measure(self, so_multi):
        targets = {"test1": np.array([0.6, 0.7, 0.8, 0.9]), "test2": np.array([0.6, 0.7, 0.8, 0.9])}

        result = so_multi.inv_project(targets)
        assert len(result.keys()) == 1
        assert result["test1"].shape == (2, 4)


@pytest.mark.required
class TestSufficiencyInverseProject:
    def test_empty_data(self):
        """
        Verifies that inv_project returns empty data when fed empty data
        """
        data = SufficiencyOutput(np.array([]), measures={})
        desired_accuracies = {}
        assert len(data.inv_project(desired_accuracies)) == 0

    def test_can_invert_sufficiency(self):
        """
        Tests metric projection output can be inversed
        """
        num_samples = np.arange(20, 80, step=10, dtype=np.uint)
        accuracies = num_samples / 100.0

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
