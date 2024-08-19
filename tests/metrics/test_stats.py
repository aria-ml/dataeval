from enum import Flag, auto
from typing import Dict, Tuple, TypeVar

import numpy as np
import pytest

from dataeval._internal.flags import ImageHash, ImageProperty, ImageStatistics, ImageVisuals, auto_all
from dataeval._internal.metrics.stats import BaseStatsMetric, ChannelStats, ImageStats


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

    def update(self, images):
        results = []
        for _ in images:
            results.append(self._map(self.func_map))
        self.results.extend(results)


class TestFlagAll:
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

    def test_compute_single(self):
        mm = MockStatsMetric(MockFlag.RED, mock_func_map)
        mm.update([0])
        results = mm.compute()

        assert len(results) == 1
        assert len(results["red"]) == 1
        assert results["red"] == ["RED"]

    def test_compute_all(self):
        mm = MockStatsMetric(MockFlag.ALL, mock_func_map)
        mm.update([0, 1])
        results = mm.compute()

        assert len(results) == 3
        assert results["red"] == ["RED", "RED"]
        assert results["green"] == ["GREEN", "GREEN"]
        assert results["blue"] == ["BLUE", "BLUE"]

    def test_reset(self):
        mm = MockStatsMetric(MockFlag.ALL, mock_func_map)
        mm.update([0])
        mm.reset()
        assert len(mm.results) == 0


TStatsClass = TypeVar("TStatsClass", ImageStats, ChannelStats)


def run_stats(stats: TStatsClass, count, channels) -> Tuple[TStatsClass, Dict]:
    stats.update(get_dataset(count, channels))
    results = stats.compute()
    return stats, results


class TestImageStats:
    def test_image_stats_single_channel(self):
        stats, results = run_stats(ImageStats(), 100, 1)
        assert stats._length == 100
        assert len(results) == len(ImageHash) + len(ImageStatistics) + len(ImageProperty) + len(ImageVisuals)
        assert len(results["mean"]) == 100

    def test_image_stats_triple_channel(self):
        stats, results = run_stats(ImageStats(), 100, 3)
        assert stats._length == 100
        assert len(results) == len(ImageHash) + len(ImageStatistics) + len(ImageProperty) + len(ImageVisuals)
        assert len(results["mean"]) == 100

    def test_image_stats_hashes_only(self):
        stats, results = run_stats(ImageStats(ImageHash.ALL), 100, 1)
        assert stats._length == 100
        assert len(results) == 2
        assert len(results["xxhash"]) == 100

    def test_image_stats_mean_only(self):
        stats, results = run_stats(ImageStats(ImageStatistics.MEAN), 100, 1)
        assert stats._length == 100
        assert len(results) == 1
        assert len(results["mean"]) == 100

    def test_image_stats_mean_xxhash(self):
        stats, results = run_stats(ImageStats([ImageStatistics.MEAN, ImageHash.XXHASH]), 100, 1)
        assert stats._length == 100
        assert len(results) == 2
        assert len(results["mean"]) == 100
        assert len(results["xxhash"]) == 100

    def test_image_stats_ignore_empty_flag(self):
        stats, results = run_stats(ImageStats([ImageStatistics(0), ImageHash.XXHASH]), 100, 1)
        assert stats._length == 100
        assert len(results) == 1
        assert len(results["xxhash"]) == 100

    def test_image_stats_merge_multiple_flags(self):
        stats, results = run_stats(
            ImageStats([ImageStatistics.MEAN, ImageHash.XXHASH, ImageStatistics.ENTROPY]), 100, 1
        )
        assert stats._length == 100
        assert len(results) == 3
        assert len(results["xxhash"]) == 100
        assert len(results["mean"]) == 100
        assert len(results["entropy"]) == 100

    def test_image_stats_reset(self):
        stats, _ = run_stats(ImageStats(ImageStatistics.MEAN), 100, 1)
        stats.reset()
        assert stats._length == 0
        assert len(stats._metrics_dict) == 1
        assert len(next(iter(stats._metrics_dict.items()))[1]) == 0


class TestChannelStats:
    def test_channel_stats_single_channel(self):
        stats, results = run_stats(ChannelStats(), 100, 1)
        assert len(results) == len(ImageStatistics) + 1
        assert len(results["mean"]) == 1
        assert results["mean"][1].shape == (1, 100)

    def test_channel_stats_triple_channel(self):
        stats, results = run_stats(ChannelStats(), 100, 3)
        assert len(results) == len(ImageStatistics) + 1
        assert len(results["mean"]) == 1
        assert results["mean"][3].shape == (3, 100)

    def test_channel_stats_triple_channel_mean_only(self):
        stats, results = run_stats(ChannelStats(ImageStatistics.MEAN), 100, 3)
        assert len(results) == 2
        assert results["mean"][3].shape == (3, 100)

    def test_channel_stats_mixed_channels(self):
        data_triple = get_dataset(50, 3)
        data_single = get_dataset(50, 1)
        stats = ChannelStats()
        stats.update(data_triple)
        stats.update(data_single)
        results = stats.compute()
        assert len(results) == len(ImageStatistics) + 1
        assert len(results["mean"]) == 2
        assert results["mean"][3].shape == (3, 50)
        assert results["mean"][1].shape == (1, 50)

    def test_channel_stats_reset(self):
        stats, _ = run_stats(ChannelStats(ImageStatistics.MEAN), 100, 3)
        stats.reset()
        assert len(stats._metrics_dict) == 1
        assert len(next(iter(stats._metrics_dict.items()))[1]) == 0
