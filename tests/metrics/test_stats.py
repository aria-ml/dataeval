import numpy as np
import pytest

from dataeval._internal.flags import ImageStat, to_distinct
from dataeval.metrics import channelstats, imagestats


def get_dataset(count: int, channels: int):
    return [np.random.random((channels, 16, 16)) for _ in range(count)]


class TestImageStats:
    def test_image_stats_single_channel(self):
        results = imagestats(get_dataset(100, 1))
        assert len(results.dict()) == len(to_distinct(ImageStat.ALL_STATS))
        assert len(results.mean) == 100

    def test_image_stats_triple_channel(self):
        results = imagestats(get_dataset(100, 3))
        assert len(results.dict()) == len(to_distinct(ImageStat.ALL_STATS))
        assert len(results.mean) == 100

    def test_image_stats_hashes_only(self):
        results = imagestats(get_dataset(100, 1), ImageStat.ALL_HASHES)
        assert len(results.dict()) == 2
        assert len(results.xxhash) == 100

    def test_image_stats_mean_only(self):
        results = imagestats(get_dataset(100, 1), ImageStat.MEAN)
        assert len(results.dict()) == 1
        assert len(results.mean) == 100

    def test_image_stats_mean_xxhash(self):
        results = imagestats(get_dataset(100, 1), ImageStat.MEAN | ImageStat.XXHASH)
        assert len(results.dict()) == 2
        assert len(results.mean) == 100
        assert len(results.xxhash) == 100

    def test_image_stats_ignore_empty_flag(self):
        results = imagestats(get_dataset(100, 1), ImageStat(0) | ImageStat.XXHASH)
        assert len(results.dict()) == 1
        assert len(results.xxhash) == 100

    def test_image_stats_merge_multiple_flags(self):
        results = imagestats(get_dataset(100, 1), ImageStat.MEAN | ImageStat.XXHASH | ImageStat.ENTROPY)
        assert len(results.dict()) == 3
        assert len(results.xxhash) == 100
        assert len(results.mean) == 100
        assert len(results.entropy) == 100


class TestChannelStats:
    def test_channel_stats_single_channel(self):
        results = channelstats(get_dataset(100, 1))
        assert len(results.dict()) == len(to_distinct(ImageStat.ALL_PIXELSTATS)) + 1
        assert len(results.mean) == 1
        assert results.mean[1].shape == (1, 100)

    def test_channel_stats_triple_channel(self):
        results = channelstats(get_dataset(100, 3))
        assert len(results.dict()) == len(to_distinct(ImageStat.ALL_PIXELSTATS)) + 1
        assert len(results.mean) == 1
        assert results.mean[3].shape == (3, 100)

    def test_channel_stats_triple_channel_mean_only(self):
        results = channelstats(get_dataset(100, 3), ImageStat.MEAN)
        assert len(results.dict()) == 2
        assert results.mean[3].shape == (3, 100)

    def test_channel_stats_mixed_channels(self):
        data_triple = get_dataset(50, 3)
        data_single = get_dataset(50, 1)
        results = channelstats(data_triple + data_single)
        assert len(results.dict()) == len(to_distinct(ImageStat.ALL_PIXELSTATS)) + 1
        assert len(results.mean) == 2
        assert results.mean[3].shape == (3, 50)
        assert results.mean[1].shape == (1, 50)

    def test_channel_stats_value_error(self):
        with pytest.raises(ValueError):
            channelstats(get_dataset(1, 1), ImageStat.XXHASH)
