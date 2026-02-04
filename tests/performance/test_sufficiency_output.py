import warnings

import numpy as np
import numpy.testing as npt
import polars as pl
import pytest

from dataeval.performance._output import (
    LRU_CACHE_SIZE,
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
        assert isinstance(result, pl.DataFrame)
        npt.assert_almost_equal(result["test1"].to_numpy(), [10.0], decimal=4)

    def test_project_invalid_steps(self, so_single_averaged_inputs):
        with pytest.raises(ValueError):
            so_single_averaged_inputs.project("not a number")  # type: ignore

    def test_project_classwise(self, so_multi_averaged_inputs):
        assert so_multi_averaged_inputs.averaged_measures["test1"].shape == (2, 3)
        result = so_multi_averaged_inputs.project([1000, 2000, 4000, 8000])
        assert isinstance(result, pl.DataFrame)
        # Multi-class produces test1_0, test1_1 columns
        assert "test1_0" in result.columns
        assert "test1_1" in result.columns
        assert len(result) == 4  # 4 projection steps

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
        assert isinstance(result, pl.DataFrame)
        assert "test1_0" in result.columns
        assert "test1_1" in result.columns
        assert len(result) == 4

    def test_project_mixed(self, so_mixed_averaged_inputs):
        assert so_mixed_averaged_inputs.averaged_measures["test1"].shape == (2, 3)
        assert so_mixed_averaged_inputs.averaged_measures["test2"].shape == (3,)
        result = so_mixed_averaged_inputs.project([1000, 2000, 4000, 8000])
        assert isinstance(result, pl.DataFrame)
        # test1 is multi-class, test2 is scalar
        assert "test1_0" in result.columns
        assert "test1_1" in result.columns
        assert "test2" in result.columns
        assert len(result) == 4

    def test_unaveraged_inputs_project_mixed(self, so_mixed_unaveraged_inputs):
        assert so_mixed_unaveraged_inputs.measures["test1"].shape == (3, 3, 2)
        assert so_mixed_unaveraged_inputs.measures["test2"].shape == (3, 3)
        assert so_mixed_unaveraged_inputs.averaged_measures["test1"].shape == (2, 3)
        assert so_mixed_unaveraged_inputs.averaged_measures["test2"].shape == (3,)
        result = so_mixed_unaveraged_inputs.project([1000, 2000, 4000, 8000])
        assert isinstance(result, pl.DataFrame)
        assert "test1_0" in result.columns
        assert "test1_1" in result.columns
        assert "test2" in result.columns
        assert len(result) == 4
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
        assert isinstance(result, pl.DataFrame)
        # test1 is multi-class (2 classes), test2 is scalar
        assert "test1_0" in result.columns
        assert "test1_1" in result.columns
        assert "test2" in result.columns
        assert len(result) == 4  # 4 target values

    def test_inv_project_ignore_unknown_measure(self, so_multi_averaged_inputs):
        targets = {"test1": np.array([0.6, 0.7, 0.8, 0.9]), "test2": np.array([0.6, 0.7, 0.8, 0.9])}

        result = so_multi_averaged_inputs.inv_project(targets)
        assert isinstance(result, pl.DataFrame)
        # Only test1 exists in averaged_measures, test2 is ignored
        assert "test1_0" in result.columns
        assert "test1_1" in result.columns
        assert "test2" not in result.columns


@pytest.mark.required
class TestSufficiencyInverseProject:
    def test_empty_data(self):
        """
        Verifies that inv_project returns empty DataFrame when fed empty data
        """
        data = SufficiencyOutput(np.array([]), measures={}, averaged_measures={})
        desired_accuracies = {}
        result = data.inv_project(desired_accuracies)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0

    def test_unaveraged_inputs_empty_data(self):
        """
        Verifies that inv_project returns empty DataFrame when fed empty data and initialized with unaveraged measures
        """
        data = SufficiencyOutput(np.array([]), measures={})
        desired_accuracies = {}
        result = data.inv_project(desired_accuracies)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0

    def test_can_invert_sufficiency(self):
        """
        Tests metric projection output can be inversed
        """
        num_samples = np.arange(20, 80, step=10, dtype=np.intp)
        accuracies = num_samples / 100.0

        data = SufficiencyOutput(steps=num_samples, measures={}, averaged_measures={"Accuracy": accuracies})
        data._params = {1000: {"Accuracy": np.array([-0.01, -1.0, 1.0])}}

        desired_accuracies = {"Accuracy": np.array([0.4, 0.6])}
        result = data.inv_project(desired_accuracies)
        needed_data = result["Accuracy"].to_numpy()

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
        result = data.inv_project(desired_accuracies)
        needed_data = result["Accuracy"].to_numpy()

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
            result = data.inv_project(desired_accuracies)
            needed_data = result["Accuracy"].to_numpy()

        # 0.90 and 0.93 targets achievable, 0.99 above curve upper bound
        desired_accuracies = {"Accuracy": np.array([0.90, 0.93, 0.99])}
        # expect warning for 0.99 target
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            result = data.inv_project(desired_accuracies)
            needed_data = result["Accuracy"].to_numpy()
        assert "Number of samples could not be determined for target(s): [0.99] with asymptote of 0.9369" in caplog.text
        target_needed_data = np.array([925, 6649, -1])
        npt.assert_array_equal(needed_data, target_needed_data)

        # all target accuracies unachievable, 0.9368 returns value greater than int64
        desired_accuracies = {"Accuracy": np.array([0.9369, 1, 1.01])}
        # expect warning for all targets
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            result = data.inv_project(desired_accuracies)
            needed_data = result["Accuracy"].to_numpy()
        assert "Number of samples could not be determined for target(s): [0.9369, 1.0, 1.01]" in caplog.text

        target_needed_data = np.array([-1, -1, -1])
        npt.assert_array_equal(needed_data, target_needed_data)


@pytest.mark.required
class TestParamsLRUCache:
    @pytest.fixture
    def so_for_cache(self) -> SufficiencyOutput:
        """Fresh SufficiencyOutput for cache testing (no pre-set _params)."""
        return SufficiencyOutput(
            steps=np.array([10, 100, 1000]),
            measures={},
            averaged_measures={"test1": np.array([0.2, 0.6, 0.9])},
        )

    def test_cache_stores_params(self, so_for_cache):
        """Verify that get_params stores computed params in cache."""
        assert so_for_cache._params is None
        assert so_for_cache._params_cache_keys == []

        # First call should compute and cache
        params = so_for_cache.get_params(n_iter=10)

        assert so_for_cache._params is not None
        assert 10 in so_for_cache._params
        assert so_for_cache._params_cache_keys == [10]
        assert "test1" in params

    def test_cache_hit_returns_same_object(self, so_for_cache):
        """Verify cache hit returns the same cached object."""
        params1 = so_for_cache.get_params(n_iter=10)
        params2 = so_for_cache.get_params(n_iter=10)

        # Should be the exact same object (not recomputed)
        assert params1 is params2

    def test_cache_hit_updates_access_order(self, so_for_cache):
        """Verify cache hit moves key to end (most recently used)."""
        so_for_cache.get_params(n_iter=10)
        so_for_cache.get_params(n_iter=20)
        so_for_cache.get_params(n_iter=30)

        assert so_for_cache._params_cache_keys == [10, 20, 30]

        # Access 10 again - should move to end
        so_for_cache.get_params(n_iter=10)
        assert so_for_cache._params_cache_keys == [20, 30, 10]

        # Access 20 again - should move to end
        so_for_cache.get_params(n_iter=20)
        assert so_for_cache._params_cache_keys == [30, 10, 20]

    def test_cache_eviction_when_full(self, so_for_cache):
        """Verify oldest entry is evicted when cache exceeds LRU_CACHE_SIZE."""
        # Fill the cache
        for i in range(LRU_CACHE_SIZE):
            so_for_cache.get_params(n_iter=i + 1)

        assert len(so_for_cache._params_cache_keys) == LRU_CACHE_SIZE
        assert so_for_cache._params_cache_keys[0] == 1  # Oldest
        assert 1 in so_for_cache._params

        # Add one more - should evict oldest (n_iter=1)
        so_for_cache.get_params(n_iter=LRU_CACHE_SIZE + 1)

        assert len(so_for_cache._params_cache_keys) == LRU_CACHE_SIZE
        assert 1 not in so_for_cache._params  # Evicted
        assert 1 not in so_for_cache._params_cache_keys
        assert LRU_CACHE_SIZE + 1 in so_for_cache._params
        assert so_for_cache._params_cache_keys[-1] == LRU_CACHE_SIZE + 1

    def test_cache_eviction_respects_access_order(self, so_for_cache):
        """Verify LRU eviction respects access order, not insertion order."""
        # Fill the cache
        for i in range(LRU_CACHE_SIZE):
            so_for_cache.get_params(n_iter=i + 1)

        # Access the oldest (n_iter=1) to make it most recently used
        so_for_cache.get_params(n_iter=1)
        assert so_for_cache._params_cache_keys[-1] == 1

        # Now n_iter=2 is the oldest
        assert so_for_cache._params_cache_keys[0] == 2

        # Add new entry - should evict n_iter=2, not n_iter=1
        so_for_cache.get_params(n_iter=LRU_CACHE_SIZE + 1)

        assert 1 in so_for_cache._params  # Still cached (was accessed)
        assert 2 not in so_for_cache._params  # Evicted (was oldest)

    def test_direct_params_manipulation_still_works(self):
        """Verify direct _params manipulation (used in tests) doesn't break cache."""
        data = SufficiencyOutput(
            steps=np.array([10, 100, 1000]),
            measures={},
            averaged_measures={"test1": np.array([0.2, 0.6, 0.9])},
        )
        # Directly set _params (as done in other test fixtures)
        data._params = {1000: {"test1": np.array([-0.1, -1.0, 1.0])}}

        # get_params should still work and add to cache_keys
        params = data.get_params(n_iter=1000)
        assert params is data._params[1000]
        assert 1000 in data._params_cache_keys
