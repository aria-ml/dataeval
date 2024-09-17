import numpy as np
import pytest

from dataeval._internal.detectors.merged_stats import add_stats, combine_stats
from dataeval._internal.flags import ImageStat
from dataeval._internal.metrics.stats import channelstats, imagestats


def get_dataset(count: int, channels: int):
    return [np.random.random((channels, 16, 16)) for _ in range(count)]


class TestMergingStats:
    def test_image_stats_addition(self):
        data_a = get_dataset(10, 1)
        data_b = get_dataset(10, 1)
        results_a = imagestats(data_a, ImageStat.ALL)
        results_b = imagestats(data_b, ImageStat.ALL)
        results_added = add_stats(results_a, results_b)
        results_alldata = imagestats(data_a + data_b)
        assert len(results_added.mean) == 20
        assert len(results_alldata.mean) == 20

    def test_channel_stats_addition(self):
        data_a = get_dataset(10, 3)
        data_b = get_dataset(10, 1)
        data_c = get_dataset(10, 3)
        results_a = channelstats(data_a)
        results_b = channelstats(data_b)
        results_c = channelstats(data_c)
        results_added, dataset_steps = combine_stats((results_a, results_b, results_c))
        results_alldata = channelstats(data_a + data_b + data_c)
        assert results_added is not None
        assert len(results_added.ch_idx_map[3]) == 20
        assert len(results_alldata.ch_idx_map[3]) == 20
        np.testing.assert_array_equal(results_added.mean[3], results_alldata.mean[3])
        assert dataset_steps == [10, 20, 30]

    def test_combine_image_channel_stats_fail(self):
        stats_a = imagestats(get_dataset(10, 1), ImageStat.ALL_PIXELSTATS)
        stats_b = channelstats(get_dataset(10, 1), ImageStat.ALL_PIXELSTATS)
        with pytest.raises(ValueError):
            add_stats(stats_a, stats_b)
        with pytest.raises(ValueError):
            add_stats(stats_b, stats_a)

    def test_combine_missing_stats_fail(self):
        stats_a = imagestats(get_dataset(10, 1))
        stats_b = imagestats(get_dataset(10, 1), ImageStat.MEAN)
        with pytest.raises(ValueError):
            add_stats(stats_a, stats_b)

    def test_combine_extra_stats_warn(self):
        stats_a = imagestats(get_dataset(10, 1))
        stats_b = imagestats(get_dataset(10, 1), ImageStat.MEAN)
        with pytest.warns():
            add_stats(stats_b, stats_a)

    def test_merge_different_types(self):
        stats_a = imagestats(get_dataset(10, 1))
        random_obj = "hello"
        with pytest.raises(TypeError):
            add_stats(stats_a, random_obj)  # type: ignore
