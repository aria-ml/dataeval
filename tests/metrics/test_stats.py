from enum import Flag, auto
from typing import Dict

import numpy as np
import pytest

from daml._internal.metrics.flags import ImageStatistics, auto_all
from daml._internal.metrics.stats import (
    BaseStatsMetric,
    ChannelStatisticsMetric,
    ChannelStats,
    ImageHashMetric,
    ImageStatisticsMetric,
    ImageStats,
)


class MockFlag(Flag):
    RED = auto()
    GREEN = auto()
    BLUE = auto()
    ALL = auto_all()


mock_func_map = {
    MockFlag.RED: lambda: "RED",
    MockFlag.GREEN: lambda: "GREEN",
    MockFlag.BLUE: lambda: "BLUE",
}

mock_bad_func_map = {
    MockFlag(0): lambda: "NONE",
}


def get_dataset(count: int, channels: int):
    return [np.random.random((channels, 16, 16)) for _ in range(count)]


class MockStatsMetric(BaseStatsMetric):
    def __init__(self, flags: MockFlag, func_map: Dict):
        super().__init__(flags)
        self.func_map = func_map

    def update(self, batch):
        results = []
        for _ in batch:
            results = self._map(self.func_map)
        self.results.append(results)


class TestFlags:
    def test_auto_all(self):
        assert MockFlag.RED & MockFlag.ALL
        assert MockFlag.GREEN & MockFlag.ALL
        assert MockFlag.BLUE & MockFlag.ALL


class TestBaseStatsMetric:
    def test_map_all(self):
        mm = MockStatsMetric(MockFlag.ALL, mock_func_map)
        mm.update([0])
        assert len(mm.results) == 1
        assert len(mm.results[0]) == 3

    def test_map_one(self):
        mm = MockStatsMetric(MockFlag.RED, mock_func_map)
        mm.update([0])
        assert len(mm.results) == 1
        assert len(mm.results[0]) == 1
        assert mm.results[0] == {"red": "RED"}

    def test_map_invalid_type(self):
        mm = MockStatsMetric(1000, mock_func_map)  # type: ignore
        with pytest.raises(TypeError):
            mm.update([0])

    def test_map_invalid_value(self):
        mm = MockStatsMetric(MockFlag.RED, mock_bad_func_map)
        with pytest.raises(ValueError):
            mm.update([0])


class TestImageStats:
    def run_stats(self, stat_class, count, channels, metrics=None):
        stats = stat_class(metrics)
        stats.update(get_dataset(count, channels))
        stats.compute()
        return stats

    def test_image_stats_single_channel(self):
        stats = self.run_stats(ImageStats, 100, 1)
        assert stats.length == 100
        assert len(stats.stats) == 19
        assert len(stats.stats["mean"]) == 100

    def test_image_stats_triple_channel(self):
        stats = self.run_stats(ImageStats, 100, 3)
        assert stats.length == 100
        assert len(stats.stats) == 19
        assert len(stats.stats["mean"]) == 100

    def test_image_stats_single_channel_hashes_only(self):
        stats = self.run_stats(ImageStats, 100, 1, [ImageHashMetric()])
        assert stats.length == 100
        assert len(stats.stats) == 2
        assert len(stats.stats["xxhash"]) == 100

    def test_image_stats_single_channel_mean_only(self):
        stats = self.run_stats(ImageStats, 100, 1, [ImageStatisticsMetric(ImageStatistics.MEAN)]).compute()
        assert stats.length == 100
        assert len(stats.stats) == 1
        assert len(stats.stats["mean"]) == 100

    def test_channel_stats_single_channel(self):
        stats = self.run_stats(ChannelStats, 100, 1).compute()
        assert len(stats.stats) == 9
        assert len(stats.stats["mean"]) == 1
        assert stats.stats["mean"][1].shape == (1, 100)

    def test_channel_stats_triple_channel(self):
        stats = self.run_stats(ChannelStats, 100, 3)
        assert stats.length == 100
        assert len(stats.stats) == 9
        assert len(stats.stats["mean"]) == 1
        assert stats.stats["mean"][3].shape == (3, 100)

    def test_channel_stats_triple_channel_mean_only(self):
        stats = self.run_stats(ChannelStats, 100, 3, ChannelStatisticsMetric(ImageStatistics.MEAN))
        assert stats.length == 100
        assert len(stats.stats) == 2
        assert stats.stats["mean"][3].shape == (3, 100)

    def test_channel_stats_mixed_channels(self):
        data_triple = [np.random.random((3, 16, 16)) for _ in range(50)]
        data_single = [np.random.random((1, 16, 16)) for _ in range(50)]
        stats = ChannelStats(data_triple + data_single)
        assert stats.length == 100
        assert len(stats.stats) == 9
        assert len(stats.stats["mean"]) == 2
        assert stats.stats["mean"][3].shape == (3, 50)
        assert stats.stats["mean"][1].shape == (1, 50)
