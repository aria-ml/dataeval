from __future__ import annotations

import warnings

import numpy as np
import numpy.testing as npt
import pytest

try:
    from matplotlib.figure import Figure
except ImportError:
    Figure = type(None)
from dataeval.workflows._output import (
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


@pytest.mark.requires_all
@pytest.mark.required
class TestSufficiencyPlot:
    def setup_method(self):
        import matplotlib.collections as mcoll
        import matplotlib.pyplot as plt

        self.mcoll = mcoll
        self.plt = plt

    def test_plot(self, so_single_averaged_inputs):
        """Tests that a plot is generated"""
        result = so_single_averaged_inputs.plot(show_asymptote=False, show_error_bars=False)
        assert len(result) == 1
        assert isinstance(result[0], Figure)
        self.plt.close(result[0])

    def test_unaveraged_inputs_plot(self, so_single_unaveraged_inputs):
        """Tests that a plot is generated"""
        result = so_single_unaveraged_inputs.plot(show_asymptote=False, show_error_bars=False)
        assert len(result) == 1
        assert isinstance(result[0], Figure)
        self.plt.close(result[0])

    def test_single_plot_asymptote(self, so_single_averaged_inputs):
        """Tests that asymptote is plotted"""
        result = so_single_averaged_inputs.plot(show_asymptote=False, show_error_bars=False)
        y = find_horizontal_line(result[0].axes[0])
        assert y is None
        with pytest.warns(UserWarning, match=r"Error bars cannot be plotted without full, unaveraged data"):
            result = so_single_averaged_inputs.plot(show_asymptote=False)
        y = find_horizontal_line(result[0].axes[0])
        assert y is None
        result = so_single_averaged_inputs.plot(show_error_bars=False)
        y = find_horizontal_line(result[0].axes[0])
        assert y == 1 - so_single_averaged_inputs._params[so_single_averaged_inputs.n_iter]["test1"][2]
        self.plt.close(result[0])

    def test_single_plot_error_bars(self, so_single_unaveraged_inputs):
        """Tests that error bars are plotted"""
        result = so_single_unaveraged_inputs.plot(show_asymptote=False, show_error_bars=False)
        y = find_error_bars(result[0].axes[0], self.mcoll)
        assert len(y) == 0
        result = so_single_unaveraged_inputs.plot(show_asymptote=False)
        y = find_error_bars(result[0].axes[0], self.mcoll)
        assert len(y) == 2
        calculated_error = np.std(so_single_unaveraged_inputs.measures["test1"], axis=0)
        calculated_error = calculated_error[~np.isclose(calculated_error, 0)]
        assert np.allclose(
            y,
            calculated_error,
            rtol=0,
            atol=1e-16,
        )
        self.plt.close(result[0])

    def test_multiplot(self, so_mixed_averaged_inputs):
        """Tests that the plot is generated for multiple classes"""
        result = so_mixed_averaged_inputs.plot(show_asymptote=False, show_error_bars=False)
        assert len(result) == 3
        assert isinstance(result[0], Figure)
        self.plt.close("all")

    def test_averaged_inputs_error_bars(self, so_mixed_averaged_inputs):
        """Tests that user is warned for plotting error bars without full, unaveraged data"""
        with pytest.warns(UserWarning, match=r"Error bars cannot be plotted without full, unaveraged data"):
            result = so_mixed_averaged_inputs.plot(show_error_bars=True)
        assert len(result) == 3
        assert isinstance(result[0], Figure)
        self.plt.close("all")

    def test_unaveraged_inputs_multiplot(self, so_mixed_unaveraged_inputs):
        """Tests that the plot is generated"""
        result = so_mixed_unaveraged_inputs.plot()
        assert len(result) == 3
        assert isinstance(result[0], Figure)
        self.plt.close("all")

    def test_multiplot_asymptote(self, so_mixed_unaveraged_inputs):
        """Tests that asymptote is generated for each figure and class"""
        result = so_mixed_unaveraged_inputs.plot(show_asymptote=False, show_error_bars=False)
        y = find_horizontal_line(result[0].axes[0])
        assert y is None
        result = so_mixed_unaveraged_inputs.plot(show_asymptote=False)
        y = find_horizontal_line(result[0].axes[0])
        assert y is None
        result = so_mixed_unaveraged_inputs.plot(show_error_bars=False)
        y = find_horizontal_line(result[0].axes[0])
        assert y == 1 - so_mixed_unaveraged_inputs._params[so_mixed_unaveraged_inputs.n_iter]["test1"][0][2]
        self.plt.close("all")
        result = so_mixed_unaveraged_inputs.plot(show_asymptote=False, show_error_bars=False)
        y = find_horizontal_line(result[1].axes[0])
        assert y is None
        result = so_mixed_unaveraged_inputs.plot(show_asymptote=False)
        y = find_horizontal_line(result[1].axes[0])
        assert y is None
        result = so_mixed_unaveraged_inputs.plot(show_error_bars=False)
        y = find_horizontal_line(result[1].axes[0])
        assert y == 1 - so_mixed_unaveraged_inputs._params[so_mixed_unaveraged_inputs.n_iter]["test1"][1][2]
        self.plt.close("all")
        result = so_mixed_unaveraged_inputs.plot(show_asymptote=False, show_error_bars=False)
        y = find_horizontal_line(result[2].axes[0])
        assert y is None
        result = so_mixed_unaveraged_inputs.plot(show_asymptote=False)
        y = find_horizontal_line(result[2].axes[0])
        assert y is None
        result = so_mixed_unaveraged_inputs.plot(show_error_bars=False)
        y = find_horizontal_line(result[2].axes[0])
        assert y == 1 - so_mixed_unaveraged_inputs._params[so_mixed_unaveraged_inputs.n_iter]["test2"][2]
        self.plt.close("all")

    def test_multiplot_error_bars(self, so_mixed_unaveraged_inputs):
        """Tests that error bars are plotted for each figure and class"""
        result = so_mixed_unaveraged_inputs.plot(show_asymptote=False, show_error_bars=False)
        y = find_error_bars(result[0].axes[0], self.mcoll)
        assert len(y) == 0
        result = so_mixed_unaveraged_inputs.plot(show_asymptote=False, show_error_bars=True)
        y = find_error_bars(result[0].axes[0], self.mcoll)
        assert len(y) == 2
        calculated_error = np.std(so_mixed_unaveraged_inputs.measures["test1"][:, :, 0], axis=0)
        calculated_error = calculated_error[~np.isclose(calculated_error, 0)]
        assert np.allclose(
            y,
            calculated_error,
            rtol=0,
            atol=1e-16,
        )
        y = find_error_bars(result[1].axes[0], self.mcoll)
        assert len(y) == 2
        calculated_error = np.std(so_mixed_unaveraged_inputs.measures["test1"][:, :, 1], axis=0)
        calculated_error = calculated_error[~np.isclose(calculated_error, 0)]

        assert np.allclose(
            y,
            calculated_error,
            rtol=0,
            atol=1e-16,
        )
        y = find_error_bars(result[2].axes[0], self.mcoll)
        assert len(y) == 2
        calculated_error = np.std(so_mixed_unaveraged_inputs.measures["test2"], axis=0)
        calculated_error = calculated_error[~np.isclose(calculated_error, 0)]
        assert np.allclose(
            y,
            calculated_error,
            rtol=0,
            atol=1e-16,
        )
        self.plt.close("all")

    def test_multiplot_classwise(self, so_multi_averaged_inputs):
        result = so_multi_averaged_inputs.plot(show_asymptote=False, show_error_bars=False)
        assert len(result) == 2
        assert isinstance(result[0], Figure)
        self.plt.close("all")

    def test_unaveraged_inputs_multiplot_classwise(self, so_multi_unaveraged_inputs):
        result = so_multi_unaveraged_inputs.plot(show_asymptote=False, show_error_bars=False)
        assert len(result) == 2
        assert isinstance(result[0], Figure)
        self.plt.close("all")

    def test_multiplot_classwise_asymptote(self, so_multi_unaveraged_inputs):
        """Tests that asymptote is generated for each class"""
        for i in range(2):
            result = so_multi_unaveraged_inputs.plot(show_asymptote=False, show_error_bars=False)
            y = find_horizontal_line(result[i].axes[0])
            assert y is None
            result = so_multi_unaveraged_inputs.plot(show_asymptote=False)
            y = find_horizontal_line(result[i].axes[0])
            assert y is None
            result = so_multi_unaveraged_inputs.plot()
            y = find_horizontal_line(result[i].axes[0])
            assert y == 1 - so_multi_unaveraged_inputs._params[so_multi_unaveraged_inputs.n_iter]["test1"][i][2]
        self.plt.close("all")

    def test_multiplot_classwise_error_bars(self, so_multi_unaveraged_inputs):
        """Tests that error bars are plotted for each class"""
        for i in range(2):
            result = so_multi_unaveraged_inputs.plot(show_error_bars=False)
            y = find_error_bars(result[i].axes[0], self.mcoll)
            assert len(y) == 0
            result = so_multi_unaveraged_inputs.plot(show_error_bars=True)
            y = find_error_bars(result[i].axes[0], self.mcoll)
            assert len(y) == 2
            calculated_error = np.std(so_multi_unaveraged_inputs.measures["test1"][:, :, i], axis=0)
            calculated_error = calculated_error[~np.isclose(calculated_error, 0)]
            assert np.allclose(
                y,
                calculated_error,
                rtol=0,
                atol=1e-16,
            )
        self.plt.close("all")

    def test_multiplot_classwise_invalid_names(self, so_multi_averaged_inputs):
        with pytest.raises(IndexError):
            so_multi_averaged_inputs.plot(["A", "B", "C"], show_asymptote=False, show_error_bars=False)
        self.plt.close("all")

    def test_unaveraged_inputs_multiplot_classwise_invalid_names(self, so_multi_unaveraged_inputs):
        with pytest.raises(IndexError):
            so_multi_unaveraged_inputs.plot(["A", "B", "C"], show_asymptote=False, show_error_bars=False)
        self.plt.close("all")

    def test_multiplot_classwise_with_names(self, so_multi_averaged_inputs):
        result = so_multi_averaged_inputs.plot(["A", "B"], show_asymptote=False, show_error_bars=False)
        assert result[0].axes[0].get_title().startswith("test1_A")
        self.plt.close("all")

    def test_unaveraged_inputs_multiplot_classwise_with_names(self, so_multi_unaveraged_inputs):
        result = so_multi_unaveraged_inputs.plot(["A", "B"], show_asymptote=False, show_error_bars=False)
        assert result[0].axes[0].get_title().startswith("test1_A")
        self.plt.close("all")

    def test_unaveraged_inputs_multiplot_classwise_without_names(self, so_multi_unaveraged_inputs):
        result = so_multi_unaveraged_inputs.plot(show_asymptote=False, show_error_bars=False)
        assert result[0].axes[0].get_title().startswith("test1_0")
        self.plt.close("all")

    def test_multiplot_mixed(self, so_mixed_averaged_inputs):
        result = so_mixed_averaged_inputs.plot(show_asymptote=False, show_error_bars=False)
        assert len(result) == 3
        assert result[0].axes[0].get_title().startswith("test1_0")
        assert result[1].axes[0].get_title().startswith("test1_1")
        assert result[2].axes[0].get_title().startswith("test2")
        self.plt.close("all")

    def test_unaveraged_inputs_multiplot_mixed(self, so_mixed_unaveraged_inputs):
        result = so_mixed_unaveraged_inputs.plot(show_asymptote=False, show_error_bars=False)
        assert len(result) == 3
        assert result[0].axes[0].get_title().startswith("test1_0")
        assert result[1].axes[0].get_title().startswith("test1_1")
        self.plt.close("all")

    @pytest.mark.parametrize(
        "asymptote, error_bars, expected_num",
        [(False, False, 4), (True, False, 6), (False, True, 10), (True, True, 12)],
    )
    def test_plot_multiple_of_same_output(self, so_mixed_unaveraged_inputs, asymptote, error_bars, expected_num):
        """Tests expected number of elements for plotting identical SufficiencyOutputs"""
        result = so_mixed_unaveraged_inputs.plot(
            show_asymptote=asymptote, show_error_bars=error_bars, reference_outputs=so_mixed_unaveraged_inputs
        )
        assert len(result) == 3
        assert result[0].axes[0].get_title().startswith("test1_0")
        assert len(result[0].axes[0].lines) + len(result[0].axes[0].collections) == expected_num
        assert result[1].axes[0].get_title().startswith("test1_1")
        assert len(result[1].axes[0].lines) + len(result[1].axes[0].collections) == expected_num
        assert result[2].axes[0].get_title().startswith("test2")
        assert len(result[2].axes[0].lines) + len(result[2].axes[0].collections) == expected_num
        self.plt.close("all")

    @pytest.mark.parametrize("asymptote, error_bars, expected_num", [(False, False, 2), (True, False, 3)])
    def test_plot_multiple_outputs_incompatible_classes(
        self, so_mixed_averaged_inputs, so_single_averaged_inputs, asymptote, error_bars, expected_num
    ):
        """Tests expected number of elements for plotting SufficiencyOutputs with no compatible test classes"""
        result = so_mixed_averaged_inputs.plot(
            show_asymptote=asymptote, show_error_bars=error_bars, reference_outputs=so_single_averaged_inputs
        )
        assert len(result) == 3
        assert result[0].axes[0].get_title().startswith("test1_0")
        assert len(result[0].axes[0].lines) + len(result[0].axes[0].collections) == expected_num
        assert result[1].axes[0].get_title().startswith("test1_1")
        assert len(result[1].axes[0].lines) + len(result[1].axes[0].collections) == expected_num
        assert result[2].axes[0].get_title().startswith("test2")
        assert len(result[2].axes[0].lines) + len(result[2].axes[0].collections) == expected_num

        self.plt.close("all")

    @pytest.mark.parametrize(
        "asymptote, error_bars, expected_multiclass, expected_single_class", [(False, False, 4, 2), (True, False, 6, 3)]
    )
    def test_plot_multiple_outputs_mixed_classes(
        self,
        so_mixed_averaged_inputs,
        so_multi_averaged_inputs,
        asymptote,
        error_bars,
        expected_multiclass,
        expected_single_class,
    ):
        """Tests expected number of elements for plotting SufficiencyOutputs with one compatible test"""
        result = so_mixed_averaged_inputs.plot(
            show_asymptote=asymptote, show_error_bars=error_bars, reference_outputs=so_multi_averaged_inputs
        )
        assert len(result) == 3
        assert result[0].axes[0].get_title().startswith("test1_0")
        assert len(result[0].axes[0].lines) + len(result[0].axes[0].collections) == expected_multiclass
        assert result[1].axes[0].get_title().startswith("test1_1")
        assert len(result[1].axes[0].lines) + len(result[1].axes[0].collections) == expected_multiclass
        assert result[2].axes[0].get_title().startswith("test2")
        assert len(result[2].axes[0].lines) + len(result[2].axes[0].collections) == expected_single_class
        self.plt.close("all")

    @pytest.mark.parametrize(
        "asymptote, error_bars, expected_multiclass, expected_single_class", [(False, False, 6, 4), (True, False, 9, 6)]
    )
    def test_plot_multiple_outputs(
        self,
        so_mixed_averaged_inputs,
        so_multi_averaged_inputs,
        asymptote,
        error_bars,
        expected_multiclass,
        expected_single_class,
    ):
        """Tests expected number of elements for plotting three SufficiencyOutputs together"""
        result = so_mixed_averaged_inputs.plot(
            show_asymptote=asymptote,
            show_error_bars=error_bars,
            reference_outputs=[so_multi_averaged_inputs, so_mixed_averaged_inputs],
        )
        assert len(result) == 3
        assert result[0].axes[0].get_title().startswith("test1_0")
        assert len(result[0].axes[0].lines) + len(result[0].axes[0].collections) == expected_multiclass
        assert result[1].axes[0].get_title().startswith("test1_1")
        assert len(result[1].axes[0].lines) + len(result[1].axes[0].collections) == expected_multiclass
        assert result[2].axes[0].get_title().startswith("test2")
        assert len(result[2].axes[0].lines) + len(result[2].axes[0].collections) == expected_single_class
        self.plt.close("all")


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
        num_samples = np.arange(20, 80, step=10, dtype=np.uint32)
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

    def test_f_inv_out_unachievable_targets(self):
        """
        Verifies that f_inv_out handles unachievable targets
        """
        num_samples = np.arange(20, 80, step=10, dtype=np.uint32)
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
        with pytest.warns(
            UserWarning,
            match=r"Number of samples could not be determined for target\(s\): \[0\.99\] with asymptote of 0\.9369",
        ):
            needed_data = data.inv_project(desired_accuracies)["Accuracy"]
        target_needed_data = np.array([925, 6649, -1])
        npt.assert_array_equal(needed_data, target_needed_data)

        # all target accuracies unachievable, 0.9368 returns value greater than int64
        desired_accuracies = {"Accuracy": np.array([0.9369, 1, 1.01])}
        # expect warning for all targets
        with pytest.warns(
            UserWarning,
            match=r"Number of samples could not be determined for target\(s\): \[0\.9369, 1\.0, 1\.01\]",
        ):
            needed_data = data.inv_project(desired_accuracies)["Accuracy"]

        target_needed_data = np.array([-1, -1, -1])
        npt.assert_array_equal(needed_data, target_needed_data)
