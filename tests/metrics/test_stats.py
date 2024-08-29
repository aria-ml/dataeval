import numpy as np

from dataeval._internal.flags import ImageHash, ImageProperty, ImageStatistics, ImageVisuals
from dataeval._internal.metrics.stats import channelstats, imagestats


def get_dataset(count: int, channels: int):
    return [np.random.random((channels, 16, 16)) for _ in range(count)]


class TestImageStats:
    def test_image_stats_single_channel(self):
        results = imagestats(get_dataset(100, 1))
        assert len(results) == len(ImageHash) + len(ImageStatistics) + len(ImageProperty) + len(ImageVisuals)
        assert len(results["mean"]) == 100

    def test_image_stats_triple_channel(self):
        results = imagestats(get_dataset(100, 3))
        assert len(results) == len(ImageHash) + len(ImageStatistics) + len(ImageProperty) + len(ImageVisuals)
        assert len(results["mean"]) == 100

    def test_image_stats_hashes_only(self):
        results = imagestats(get_dataset(100, 1), ImageHash.ALL)
        assert len(results) == 2
        assert len(results["xxhash"]) == 100

    def test_image_stats_mean_only(self):
        results = imagestats(get_dataset(100, 1), ImageStatistics.MEAN)
        assert len(results) == 1
        assert len(results["mean"]) == 100

    def test_image_stats_mean_xxhash(self):
        results = imagestats(get_dataset(100, 1), (ImageStatistics.MEAN, ImageHash.XXHASH))
        assert len(results) == 2
        assert len(results["mean"]) == 100
        assert len(results["xxhash"]) == 100

    def test_image_stats_ignore_empty_flag(self):
        results = imagestats(get_dataset(100, 1), (ImageStatistics(0), ImageHash.XXHASH))
        assert len(results) == 1
        assert len(results["xxhash"]) == 100

    def test_image_stats_merge_multiple_flags(self):
        results = imagestats(get_dataset(100, 1), [ImageStatistics.MEAN, ImageHash.XXHASH, ImageStatistics.ENTROPY])
        assert len(results) == 3
        assert len(results["xxhash"]) == 100
        assert len(results["mean"]) == 100
        assert len(results["entropy"]) == 100


class TestChannelStats:
    def test_channel_stats_single_channel(self):
        results = channelstats(get_dataset(100, 1))
        assert len(results) == len(ImageStatistics) + 1
        assert len(results["mean"]) == 1
        assert results["mean"][1].shape == (1, 100)

    def test_channel_stats_triple_channel(self):
        results = channelstats(get_dataset(100, 3))
        assert len(results) == len(ImageStatistics) + 1
        assert len(results["mean"]) == 1
        assert results["mean"][3].shape == (3, 100)

    def test_channel_stats_triple_channel_mean_only(self):
        results = channelstats(get_dataset(100, 3), ImageStatistics.MEAN)
        assert len(results) == 2
        assert results["mean"][3].shape == (3, 100)

    def test_channel_stats_mixed_channels(self):
        data_triple = get_dataset(50, 3)
        data_single = get_dataset(50, 1)
        results = channelstats(data_triple + data_single)
        assert len(results) == len(ImageStatistics) + 1
        assert len(results["mean"]) == 2
        assert results["mean"][3].shape == (3, 50)
        assert results["mean"][1].shape == (1, 50)
