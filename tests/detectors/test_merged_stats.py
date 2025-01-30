import numpy as np
import pytest

from dataeval.detectors.linters.merged_stats import add_stats, combine_stats, get_dataset_step_from_idx
from dataeval.metrics.stats import hashstats, pixelstats


def get_dataset(count: int, channels: int):
    return [np.random.random((channels, 16, 16)) for _ in range(count)]


@pytest.mark.required
class TestMergingStats:
    def test_image_stats_addition(self):
        data_a = get_dataset(10, 1)
        data_b = get_dataset(10, 1)
        results_a = pixelstats(data_a)
        results_b = pixelstats(data_b)
        results_added = add_stats(results_a, results_b)
        results_alldata = pixelstats(data_a + data_b)
        assert len(results_added.mean) == 20
        assert len(results_alldata.mean) == 20

    def test_channel_stats_addition(self):
        data_a = get_dataset(10, 3)
        data_b = get_dataset(10, 1)
        data_c = get_dataset(10, 3)
        results_a = pixelstats(data_a, per_channel=True)
        results_b = pixelstats(data_b, per_channel=True)
        results_c = pixelstats(data_c, per_channel=True)
        results_added, dataset_steps = combine_stats((results_a, results_b, results_c))
        results_alldata = pixelstats(data_a + data_b + data_c, per_channel=True)
        assert results_added is not None
        np.testing.assert_array_equal(results_added.mean, results_alldata.mean)
        assert dataset_steps == [30, 40, 70]

    def test_combine_different_stats_fail(self):
        stats_a = pixelstats(get_dataset(10, 1))
        stats_b = hashstats(get_dataset(10, 1))
        with pytest.raises(TypeError):
            add_stats(stats_a, stats_b)

    def test_merge_different_types(self):
        stats_a = pixelstats(get_dataset(10, 1))
        random_obj = "hello"
        with pytest.raises(TypeError):
            add_stats(stats_a, random_obj)  # type: ignore

    def test_get_dataset_step_from_idx_no_steps(self):
        results = get_dataset_step_from_idx(0, [])
        assert results == (-1, 0)

    def test_combine_stats_with_invalid_stat_types(self):
        stats = pixelstats(get_dataset(10, 1))
        with pytest.raises(TypeError):
            combine_stats([])  # type: ignore
        with pytest.raises(TypeError):
            combine_stats([None])  # type: ignore
        with pytest.raises(TypeError):
            combine_stats([1, None])  # type: ignore
        with pytest.raises(TypeError):
            combine_stats([stats, None])  # type: ignore
        with pytest.raises(TypeError):
            combine_stats([stats, 1, None])  # type: ignore
